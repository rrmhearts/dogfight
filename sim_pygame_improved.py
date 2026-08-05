import pygame
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# --- INITIALIZATION & CONSTANTS ---
pygame.init()
GRAVITY = 98.0  # Scaled gravity for the simulation speed

class WeaponType(Enum):
    MACHINE_GUN = "machine_gun"
    MISSILE = "missile"
    CANNON = "cannon"
    BOMB = "bomb"

class ManeuverType(Enum):
    ORBIT = "orbit"
    DOGFIGHT = "dogfight"
    BOMB_TARGET = "bomb_target"
    FOLLOW = "follow"
    FLANK = "flank"
    ATTACK_RUN = "attack_run"
    CLIMB = "climb"
    DIVE = "dive"
    PATROL = "patrol"
    INTERCEPT = "intercept"
    RETREAT = "retreat"

@dataclass
class Maneuver:
    maneuver_type: ManeuverType
    duration: float  # Duration in seconds
    target: Optional[object] = None  # Target object (aircraft or ground target)
    parameters: dict = None  # Additional parameters for the maneuver
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class WeaponConfig:
    weapon_type: WeaponType
    damage: float
    range: float
    fire_rate: float  # rounds per second
    velocity: float
    ammo_count: int
    tracking: bool = False  # For missiles
    blast_radius: float = 0.0
    can_target_ground: bool = False  # Can target ground units

@dataclass
class AircraftConfig:
    max_speed: float
    acceleration: float
    turn_rate: float  # radians per second
    climb_rate: float
    max_altitude: float
    health: float
    weapons: List[WeaponConfig]

# --- VISUAL EFFECTS ---
class Effect:
    def __init__(self, effect_type, pos, magnitude):
        self.type = effect_type
        self.pos = np.array(pos, dtype=float)
        self.magnitude = magnitude
        self.lifetime = 1.0 if effect_type == 'explosion' else 2.0

    def update(self, dt):
        self.lifetime -= dt

    def draw(self, surface, project_fn):
        if self.lifetime <= 0: return
        x, y, z = project_fn(self.pos)
        
        if self.type == 'explosion':
            alpha = max(0, int(255 * self.lifetime))
            radius = int(self.magnitude * (1.0 - self.lifetime))
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (255, 165, 0, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (x - radius, y - radius))
        elif self.type == 'flares':
            for _ in range(5):
                offset = (random.uniform(-15, 15), random.uniform(-15, 15))
                pygame.draw.circle(surface, (255, 255, 0), (int(x + offset[0]), int(y + offset[1])), 2)

class GroundTarget:
    def __init__(self, x, y, target_type, health=100, team="neutral"):
        self.pos = np.array([x, y, 0], dtype=float)
        self.target_type = target_type  # "tank", "building", "aa_gun"
        self.health = health
        self.max_health = health
        self.team = team
        self.alive = True
        self.size = 15 if target_type == "tank" else 20
        
        # Anti-air capabilities for AA guns
        self.can_shoot = target_type == "aa_gun"
        self.last_shot_time = 0
        self.range = 400 if self.can_shoot else 0
        
        # Turret rotation for AA guns and tanks
        self.turret_angle = 0.0  # Current turret facing direction
        self.target_turret_angle = 0.0  # Desired turret direction
        self.turret_turn_rate = 2.0  # Radians per second
        
    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            self.health = 0
            
    def update(self, dt, aircraft_list, projectiles_list):
        if not self.alive:
            return
            
        # Update turret rotation
        self.update_turret_rotation(dt)
        self.update_tank_rotation(aircraft_list)
            
        if not self.can_shoot:
            return
            
        self.last_shot_time += dt
        
        # Find nearest enemy aircraft
        nearest_target = None
        min_distance = float('inf')
        
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                if distance < self.range and distance < min_distance:
                    min_distance = distance
                    nearest_target = aircraft
                    
        if nearest_target:
            # Calculate desired turret angle to target
            to_target = nearest_target.pos - self.pos
            self.target_turret_angle = math.atan2(to_target[1], to_target[0])
            
            # Only fire if turret is roughly pointing at target and enough time has passed
            angle_diff = abs(self.turret_angle - self.target_turret_angle)
            # Handle angle wrap-around
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
                
            # Fire if turret is aimed within 15 degrees and cooldown is ready
            if angle_diff < math.pi/12 and self.last_shot_time >= 2.0:
                self.fire_at_target(nearest_target, projectiles_list)
                self.last_shot_time = 0
        else:
            # No target - slowly return turret to neutral position (pointing north)
            self.target_turret_angle = math.pi/2  # Point upward
            
    def update_turret_rotation(self, dt):
        """Smoothly rotate turret toward target angle"""
        if abs(self.turret_angle - self.target_turret_angle) < 0.01:
            self.turret_angle = self.target_turret_angle
            return
            
        # Calculate shortest rotation direction
        angle_diff = self.target_turret_angle - self.turret_angle
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Apply rotation with rate limit
        max_rotation = self.turret_turn_rate * dt
        if abs(angle_diff) <= max_rotation:
            self.turret_angle = self.target_turret_angle
        else:
            self.turret_angle += max_rotation if angle_diff > 0 else -max_rotation
            
        # Normalize turret angle to [0, 2*pi]
        self.turret_angle = self.turret_angle % (2 * math.pi)
    
    def update_tank_rotation(self, aircraft_list):
        # Tanks also rotate turrets toward nearest enemies (but don't shoot)
        if self.target_type == "tank":
            nearest_enemy = None
            min_distance = float('inf')
            
            for aircraft in aircraft_list:
                if aircraft.team != self.team and aircraft.alive:
                    distance = np.linalg.norm(aircraft.pos - self.pos)
                    if distance < 300 and distance < min_distance:  # Tanks track at shorter range
                        min_distance = distance
                        nearest_enemy = aircraft
                        
            if nearest_enemy:
                # Point turret at enemy
                to_enemy = nearest_enemy.pos - self.pos
                self.target_turret_angle = math.atan2(to_enemy[1], to_enemy[0])
            else:
                # Return to neutral position
                self.target_turret_angle = math.pi/2

    def fire_at_target(self, target, projectiles_list):
        """Fire at the specified target"""
        to_target = target.pos - self.pos
        distance = np.linalg.norm(to_target)
        direction = to_target / distance
        
        aa_weapon = WeaponConfig(
            weapon_type=WeaponType.CANNON,
            damage=40,
            range=self.range,
            fire_rate=0.5,
            velocity=600,
            ammo_count=999,
            blast_radius=10
        )
        
        # Fire from turret position
        turret_offset = np.array([math.cos(self.turret_angle), math.sin(self.turret_angle), 0]) * 10
        fire_pos = self.pos + turret_offset
        
        projectile = Projectile(
            fire_pos[0], fire_pos[1], fire_pos[2],
            direction[0] * aa_weapon.velocity,
            direction[1] * aa_weapon.velocity,
            direction[2] * aa_weapon.velocity,
            aa_weapon, None, self.team
        )
        projectiles_list.append(projectile)

class Projectile:
    def __init__(self, x, y, z, vx, vy, vz, weapon_config, target=None, team=None):
        self.pos = np.array([x, y, z], dtype=float)
        self.velocity = np.array([vx, vy, vz], dtype=float)
        self.config = weapon_config
        self.target = target
        self.team = team
        self.lifetime = weapon_config.range / weapon_config.velocity if weapon_config.velocity > 0 else 8.0
        self.age = 0.0
        self.active = True

    def update(self, dt, aircraft_list, ground_targets, effects_list):
        if not self.active:
            return
            
        self.age += dt
        
        # Apply Gravity for Bombs
        if self.config.weapon_type == WeaponType.BOMB:
            self.velocity[2] -= GRAVITY * dt

        # Missile tracking
        if self.config.tracking and self.target and self.target.alive:
            target_pos = self.target.pos
            to_target = target_pos - self.pos
            distance = np.linalg.norm(to_target)
            
            if distance > 0:
                # Proportional navigation
                desired_velocity = (to_target / distance) * self.config.velocity
                turn_rate = 5.0  # radians per second for missiles
                max_turn = turn_rate * dt
                
                current_dir = self.velocity / np.linalg.norm(self.velocity)
                desired_dir = desired_velocity / np.linalg.norm(desired_velocity)
                
                # Smoothly turn towards target
                angle_diff = np.arccos(np.clip(np.dot(current_dir, desired_dir), -1, 1))
                if angle_diff > 0:
                    turn_amount = min(max_turn, angle_diff)
                    self.velocity = self.velocity + (desired_velocity - self.velocity) * (turn_amount / angle_diff)
                    self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * self.config.velocity

        # Update position
        self.pos += self.velocity * dt

        # Check for ground impact
        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.explode(aircraft_list, ground_targets, effects_list)
            return

        if self.age >= self.lifetime:
            self.active = False
            return

        # Check for proximity hits on aircraft
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                hit_distance = 15.0 if self.config.blast_radius > 0 else 8.0
                
                if distance < hit_distance:
                    self.explode(aircraft_list, ground_targets, effects_list)
                    return

    def explode(self, aircraft_list, ground_targets, effects_list):
        self.active = False
        radius = max(self.config.blast_radius, 20)
        effects_list.append(Effect('explosion', self.pos.copy(), radius))
        
        # Damage aircraft
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                dist = np.linalg.norm(aircraft.pos - self.pos)
                if dist < radius:
                    aircraft.take_damage(self.config.damage * (1 - dist/radius))
                    
        # Damage ground targets
        for target in ground_targets:
            if target.team != self.team and target.alive:
                dist = np.linalg.norm(target.pos - self.pos)
                if dist < radius + target.size:
                    target.take_damage(self.config.damage * (1 - dist/(radius + target.size)))


class Aircraft:
    def __init__(self, x, y, z, team, config, color):
        self.pos = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.heading = 0.0  # radians
        self.pitch = 0.0
        self.team = team
        self.config = config
        self.color = color
        self.health = config.health
        self.alive = True
        
        # Weapon systems & Countermeasures
        self.weapons = config.weapons.copy()
        self.current_weapon = 0
        self.weapon_cooldowns = [0.0] * len(self.weapons)
        self.flares = 10
        self.flare_cooldown = 0
        self.ccip = None
        
        # Maneuver system
        self.maneuver_queue = []
        self.current_maneuver = None
        self.maneuver_timer = 0.0
        self.maneuver_state = {}  # State variables for current maneuver
        
        # AI state
        self.target = None
        self.last_shot_time = 0
        self.default_behavior = True

    def add_maneuver(self, maneuver):
        """Add a maneuver to the queue"""
        self.maneuver_queue.append(maneuver)
        self.default_behavior = False

    def clear_maneuvers(self):
        """Clear all maneuvers and return to default AI"""
        self.maneuver_queue.clear()
        self.current_maneuver = None
        self.default_behavior = True

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            self.health = 0

    def deploy_flares(self, effects_list, projectiles_list):
        if self.flares > 0 and self.flare_cooldown <= 0:
            self.flares -= 1
            self.flare_cooldown = 2.0
            effects_list.append(Effect('flares', self.pos.copy(), 5))
            for p in projectiles_list:
                if p.config.weapon_type == WeaponType.MISSILE and p.target == self:
                    if random.random() < 0.8:
                        p.target = None

    def calculate_ccip(self):
        sim_pos = self.pos.copy()
        sim_vel = self.velocity.copy()
        dt_sim = 0.05
        for _ in range(200):
            sim_vel[2] -= GRAVITY * dt_sim
            sim_pos += sim_vel * dt_sim
            if sim_pos[2] <= 0:
                return sim_pos[:2]
        return None

    def find_nearest_enemy(self, aircraft_list):
        nearest_enemy = None
        min_distance = float('inf')
        
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_enemy = aircraft
                    
        return nearest_enemy, min_distance
        
    def find_nearest_ground_target(self, ground_targets):
        nearest_target = None
        min_distance = float('inf')
        
        for target in ground_targets:
            if target.team != self.team and target.alive:
                distance = np.linalg.norm(target.pos - self.pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = target
                    
        return nearest_target, min_distance

    def can_fire_at_target(self, target, is_ground=False):
        if not target or not target.alive:
            return False
            
        weapon = self.weapons[self.current_weapon]
        if is_ground and not weapon.can_target_ground:
            return False
            
        to_target = target.pos - self.pos
        distance = np.linalg.norm(to_target)
        
        # Allow bomb release if CCIP is close regardless of standard slant range
        if weapon.weapon_type == WeaponType.BOMB and self.ccip is not None:
            dist_2d = np.linalg.norm(self.ccip - target.pos[:2])
            return dist_2d < 45.0
        
        if distance > weapon.range:
            return False
            
        forward = np.array([math.cos(self.heading), math.sin(self.heading), 0])
        to_target_normalized = to_target / distance
        angle = math.acos(np.clip(np.dot(forward, to_target_normalized), -1, 1))
        
        max_angle = math.pi / 3 if is_ground else math.pi / 4
        return angle < max_angle

    def fire_weapon(self):
        if self.current_weapon >= len(self.weapons):
            return None
            
        weapon = self.weapons[self.current_weapon]
        
        # Check ammo and cooldown
        if weapon.ammo_count <= 0 or self.weapon_cooldowns[self.current_weapon] > 0:
            return None
            
        weapon.ammo_count -= 1
        self.weapon_cooldowns[self.current_weapon] = 1.0 / weapon.fire_rate
        
        # Calculate firing position and velocity
        forward = np.array([math.cos(self.heading), math.sin(self.heading), math.sin(self.pitch)])
        fire_pos = self.pos + forward * 20
        
        projectile_velocity = self.velocity.copy()
        if weapon.weapon_type != WeaponType.BOMB:
            projectile_velocity += forward * weapon.velocity
        
        target = self.target if weapon.tracking else None
        
        return Projectile(fire_pos[0], fire_pos[1], fire_pos[2],
                         projectile_velocity[0], projectile_velocity[1], projectile_velocity[2],
                         weapon, target, self.team)

    def execute_maneuver(self, dt, aircraft_list, ground_targets):
        """Execute the current maneuver"""
        if not self.current_maneuver:
            return None
            
        maneuver = self.current_maneuver
        
        if maneuver.maneuver_type == ManeuverType.ORBIT:
            self.execute_orbit(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.DOGFIGHT:
            return self.execute_dogfight(dt, maneuver, aircraft_list)
        elif maneuver.maneuver_type == ManeuverType.BOMB_TARGET:
            return self.execute_bomb_target(dt, maneuver, ground_targets)
        elif maneuver.maneuver_type == ManeuverType.FOLLOW:
            self.execute_follow(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.FLANK:
            return self.execute_flank(dt, maneuver, aircraft_list)
        elif maneuver.maneuver_type == ManeuverType.ATTACK_RUN:
            return self.execute_attack_run(dt, maneuver, aircraft_list, ground_targets)
        elif maneuver.maneuver_type == ManeuverType.CLIMB:
            self.execute_climb(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.DIVE:
            self.execute_dive(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.PATROL:
            self.execute_patrol(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.INTERCEPT:
            self.execute_intercept(dt, maneuver, aircraft_list)
        elif maneuver.maneuver_type == ManeuverType.RETREAT:
            self.execute_retreat(dt, maneuver, aircraft_list)
        return None

    def execute_orbit(self, dt, maneuver):
        """Orbit around a target or fixed point"""
        if maneuver.target and maneuver.target.alive:
            center = maneuver.target.pos
        else:
            center = maneuver.parameters.get('center', np.array([600, 400, 200]))
            
        radius = maneuver.parameters.get('radius', 150)
        orbit_speed = maneuver.parameters.get('speed', 1.0)
        
        # Calculate orbit position
        to_center = center - self.pos
        distance = np.linalg.norm(to_center[:2])  # 2D distance
        
        if distance > radius + 50:
            # Move towards orbit radius
            desired_heading = math.atan2(to_center[1], to_center[0])
        else:
            # Orbit around the target
            tangent_angle = math.atan2(to_center[1], to_center[0]) + math.pi/2 * orbit_speed
            desired_heading = tangent_angle
            
        self.turn_towards_heading(desired_heading, dt)
        
        # Maintain altitude close to target
        if maneuver.target:
            desired_alt = maneuver.target.pos[2] + 50
            self.adjust_altitude(desired_alt, dt)

    def execute_dogfight(self, dt, maneuver, aircraft_list):
        """Aggressive dogfighting maneuver"""
        if not maneuver.target or not maneuver.target.alive:
            # Find new target
            enemy, _ = self.find_nearest_enemy(aircraft_list)
            maneuver.target = enemy
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            to_target = target.pos - self.pos
            distance = np.linalg.norm(to_target)
            
            # Aggressive pursuit
            desired_heading = math.atan2(to_target[1], to_target[0])
            
            # Add lead calculation for moving targets
            target_velocity = getattr(target, 'velocity', np.zeros(3))
            time_to_intercept = distance / max(1, self.config.max_speed)
            predicted_pos = target.pos + target_velocity * time_to_intercept
            
            to_predicted = predicted_pos - self.pos
            desired_heading = math.atan2(to_predicted[1], to_predicted[0])
            
            self.turn_towards_heading(desired_heading, dt)
            
            # Altitude matching with slight advantage
            desired_alt = target.pos[2] + 30
            self.adjust_altitude(desired_alt, dt)
            
            # Fire when in range
            if self.can_fire_at_target(target):
                return self.try_fire_weapon()

    def execute_bomb_target(self, dt, maneuver, ground_targets):
        """Bomb run on ground targets utilizing CCIP"""
        if not maneuver.target or not maneuver.target.alive:
            target, _ = self.find_nearest_ground_target(ground_targets)
            maneuver.target = target
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            to_target = target.pos - self.pos
            distance_2d = np.linalg.norm(to_target[:2])
            
            if 'phase' not in self.maneuver_state:
                self.maneuver_state['phase'] = 'approach'
                # Ensure weapon is set to bomb if we have it
                for i, weapon in enumerate(self.weapons):
                    if weapon.weapon_type == WeaponType.BOMB and weapon.ammo_count > 0:
                        self.current_weapon = i
                        break
                
            if self.maneuver_state['phase'] == 'approach':
                # Climb to bombing altitude
                bomb_altitude = maneuver.parameters.get('altitude', 300)
                if self.pos[2] < bomb_altitude - 20:
                    self.pitch = min(0.3, self.pitch + dt * 0.5)
                else:
                    self.maneuver_state['phase'] = 'attack'
                    
                desired_heading = math.atan2(to_target[1], to_target[0])
                self.turn_towards_heading(desired_heading, dt)
                
            elif self.maneuver_state['phase'] == 'attack':
                self.pitch *= 0.9
                desired_heading = math.atan2(to_target[1], to_target[0])
                self.turn_towards_heading(desired_heading, dt)
                
                weapon = self.weapons[self.current_weapon]
                if weapon.weapon_type == WeaponType.BOMB and self.ccip is not None:
                    dist_to_impact = np.linalg.norm(self.ccip - target.pos[:2])
                    if dist_to_impact < 45 and self.can_fire_at_target(target, is_ground=True):
                        proj = self.try_fire_weapon()
                        if proj:
                            self.maneuver_state['phase'] = 'egress'
                            return proj
                elif distance_2d < 100 and self.can_fire_at_target(target, is_ground=True):
                    proj = self.try_fire_weapon()
                    if proj:
                        self.maneuver_state['phase'] = 'egress'
                        return proj

            elif self.maneuver_state['phase'] == 'egress':
                self.pitch = min(0.4, self.pitch + dt * 0.5)

    def execute_follow(self, dt, maneuver):
        """Follow another aircraft"""
        if not maneuver.target or not maneuver.target.alive: return
        target = maneuver.target
        follow_distance = maneuver.parameters.get('distance', 100)
        target_heading = getattr(target, 'heading', 0)
        offset_angle = maneuver.parameters.get('offset_angle', math.pi)
        
        follow_x = target.pos[0] + math.cos(target_heading + offset_angle) * follow_distance
        follow_y = target.pos[1] + math.sin(target_heading + offset_angle) * follow_distance
        follow_z = target.pos[2] + maneuver.parameters.get('altitude_offset', 0)
        
        to_follow_pos = np.array([follow_x, follow_y, follow_z]) - self.pos
        self.turn_towards_heading(math.atan2(to_follow_pos[1], to_follow_pos[0]), dt)
        self.adjust_altitude(follow_z, dt)

    def execute_flank(self, dt, maneuver, aircraft_list):
        """Flanking maneuver around target"""
        if not maneuver.target or not maneuver.target.alive:
            enemy, _ = self.find_nearest_enemy(aircraft_list)
            maneuver.target = enemy
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            flank_radius = maneuver.parameters.get('radius', 200)
            flank_side = maneuver.parameters.get('side', 'left')  # 'left' or 'right'
            
            # Calculate flanking position
            to_target = target.pos - self.pos
            target_heading = math.atan2(to_target[1], to_target[0])
            
            flank_angle = target_heading + (math.pi/2 if flank_side == 'left' else -math.pi/2)
            flank_x = target.pos[0] + math.cos(flank_angle) * flank_radius
            flank_y = target.pos[1] + math.sin(flank_angle) * flank_radius
            
            to_flank_pos = np.array([flank_x, flank_y, target.pos[2]]) - self.pos
            desired_heading = math.atan2(to_flank_pos[1], to_flank_pos[0])
            
            self.turn_towards_heading(desired_heading, dt)
            self.adjust_altitude(target.pos[2], dt)
            
            distance = np.linalg.norm(to_target)
            if distance < self.weapons[self.current_weapon].range:
                if self.can_fire_at_target(target):
                    return self.try_fire_weapon()
        return None

    def execute_attack_run(self, dt, maneuver, aircraft_list, ground_targets):
        if not maneuver.target or not maneuver.target.alive:
            return None
            
        target = maneuver.target
        to_target = target.pos - self.pos
        
        desired_heading = math.atan2(to_target[1], to_target[0])
        self.turn_towards_heading(desired_heading, dt, turn_rate_multiplier=2.0)
        
        if hasattr(target, 'target_type'):
            desired_alt = 100
        else:
            desired_alt = target.pos[2]
            
        self.adjust_altitude(desired_alt, dt)
        
        is_ground = hasattr(target, 'target_type')
        if self.can_fire_at_target(target, is_ground):
            return self.try_fire_weapon()
        return None

    def execute_climb(self, dt, maneuver):
        target_altitude = maneuver.parameters.get('altitude', 400)
        climb_rate = maneuver.parameters.get('rate', 1.0)
        if self.pos[2] < target_altitude:
            self.pitch = min(0.4, climb_rate)
        else:
            self.pitch *= 0.9

    def execute_dive(self, dt, maneuver):
        """Dive to specified altitude"""
        target_altitude = maneuver.parameters.get('altitude', 100)
        dive_rate = maneuver.parameters.get('rate', -1.0)
        if self.pos[2] > target_altitude:
            self.pitch = max(-0.4, dive_rate)
        else:
            self.pitch *= 0.9

    def execute_patrol(self, dt, maneuver):
        """Patrol between waypoints"""
        waypoints = maneuver.parameters.get('waypoints', [
            np.array([300, 300, 200]),
            np.array([900, 300, 200]),
            np.array([900, 600, 200]),
            np.array([300, 600, 200])
        ])
        
        if 'current_waypoint' not in self.maneuver_state:
            self.maneuver_state['current_waypoint'] = 0
            
        current_wp = waypoints[self.maneuver_state['current_waypoint']]
        to_waypoint = current_wp - self.pos
        distance = np.linalg.norm(to_waypoint)
        
        if distance < 50:  # Reached waypoint
            self.maneuver_state['current_waypoint'] = (self.maneuver_state['current_waypoint'] + 1) % len(waypoints)
            current_wp = waypoints[self.maneuver_state['current_waypoint']]
            to_waypoint = current_wp - self.pos
            
        desired_heading = math.atan2(to_waypoint[1], to_waypoint[0])
        self.turn_towards_heading(desired_heading, dt)
        self.adjust_altitude(current_wp[2], dt)

    def execute_intercept(self, dt, maneuver, aircraft_list):
        """Intercept enemy aircraft"""
        if not maneuver.target or not maneuver.target.alive:
            enemy, _ = self.find_nearest_enemy(aircraft_list)
            maneuver.target = enemy
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            
            # Calculate intercept point
            to_target = target.pos - self.pos
            target_velocity = getattr(target, 'velocity', np.zeros(3))
            relative_velocity = self.velocity - target_velocity
            
            # Simple intercept calculation
            time_to_intercept = np.linalg.norm(to_target) / max(1, np.linalg.norm(relative_velocity))
            intercept_point = target.pos + target_velocity * time_to_intercept
            
            to_intercept = intercept_point - self.pos
            desired_heading = math.atan2(to_intercept[1], to_intercept[0])
            
            self.turn_towards_heading(desired_heading, dt)
            self.adjust_altitude(intercept_point[2], dt)

    def execute_retreat(self, dt, maneuver, aircraft_list):
        """Retreat from enemies"""
        # Find nearest enemy
        enemy, distance = self.find_nearest_enemy(aircraft_list)
        if enemy:
            # Head away from enemy
            away_from_enemy = self.pos - enemy.pos
            desired_heading = math.atan2(away_from_enemy[1], away_from_enemy[0])
            self.turn_towards_heading(desired_heading, dt)
            # Climb for advantage
            desired_alt = min(self.config.max_altitude, self.pos[2] + 100)
            self.adjust_altitude(desired_alt, dt)

    def turn_towards_heading(self, desired_heading, dt, turn_rate_multiplier=1.0):
        """Smoothly turn towards desired heading"""
        heading_diff = desired_heading - self.heading
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        while heading_diff < -math.pi:
            heading_diff += 2 * math.pi
            
        max_turn = self.config.turn_rate * dt * turn_rate_multiplier
        turn_amount = max(-max_turn, min(max_turn, heading_diff))
        self.heading += turn_amount

    def adjust_altitude(self, desired_altitude, dt):
        """Adjust pitch to reach desired altitude"""
        alt_diff = desired_altitude - self.pos[2]
        max_pitch = 0.3
        
        if alt_diff > 20:
            self.pitch = min(max_pitch, self.pitch + dt * 0.5)
        elif alt_diff < -20:
            self.pitch = max(-max_pitch, self.pitch - dt * 0.5)
        else:
            self.pitch *= 0.95

    def try_fire_weapon(self):
        """Attempt to fire current weapon"""
        if self.weapon_cooldowns[self.current_weapon] <= 0:
            return self.fire_weapon()
        return None
        
    def manual_override(self, dt):
        """Handle manual flight control"""
        keys = pygame.key.get_pressed()
        self.clear_maneuvers()
        if keys[pygame.K_LEFT]:
            self.heading -= self.config.turn_rate * dt
        if keys[pygame.K_RIGHT]:
            self.heading += self.config.turn_rate * dt
        if keys[pygame.K_UP]:
            self.pitch = max(-0.5, self.pitch - dt)
        elif keys[pygame.K_DOWN]:
            self.pitch = min(0.5, self.pitch + dt)
        else:
            self.pitch *= 0.95
        
        if keys[pygame.K_SPACE]:
            return self.try_fire_weapon()
        return None

    def update_maneuvers(self, dt):
        """Update maneuver system"""
        # Update current maneuver timer
        if self.current_maneuver:
            self.maneuver_timer += dt
            
            # Check if maneuver is complete
            maneuver_complete = False
            if self.maneuver_timer >= self.current_maneuver.duration:
                maneuver_complete = True
                
            # Target-based completion (target destroyed)
            if (self.current_maneuver.target and 
                hasattr(self.current_maneuver.target, 'alive') and 
                not getattr(self.current_maneuver.target, 'alive', True)):
                maneuver_complete = True
                
            if maneuver_complete:
                self.current_maneuver = None
                self.maneuver_timer = 0.0
                self.maneuver_state.clear()
                
        # Start next maneuver if current one is finished
        if not self.current_maneuver and self.maneuver_queue:
            self.current_maneuver = self.maneuver_queue.pop(0)
            self.maneuver_timer = 0.0
            self.maneuver_state.clear()

    def ai_update(self, dt, aircraft_list, ground_targets, is_selected=False):
        if not self.alive: return None
        
        # Update cooldowns
        for i in range(len(self.weapon_cooldowns)):
            self.weapon_cooldowns[i] = max(0, self.weapon_cooldowns[i] - dt)
        if self.flare_cooldown > 0: self.flare_cooldown -= dt
        
        # Continuously Compute Impact Point for bombs
        if self.weapons[self.current_weapon].weapon_type == WeaponType.BOMB:
            self.ccip = self.calculate_ccip()
        else:
            self.ccip = None
            
        proj = None
        keys = pygame.key.get_pressed()
        
        # Check for manual override
        if is_selected and (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_SPACE]):
            proj = self.manual_override(dt)
        else:
            self.update_maneuvers(dt)
            if self.current_maneuver and not self.default_behavior:
                proj = self.execute_maneuver(dt, aircraft_list, ground_targets)
            else:
                # Default AI behavior
                enemy, enemy_distance = self.find_nearest_enemy(aircraft_list)
                self.target = enemy
                
                if enemy and enemy_distance < 800:
                    to_enemy = enemy.pos - self.pos
                    desired_heading = math.atan2(to_enemy[1], to_enemy[0])
                    self.turn_towards_heading(desired_heading, dt)
                    
                    if self.can_fire_at_target(enemy):
                        if random.random() < 0.02:
                            proj = self.try_fire_weapon()
                else:
                    self.pitch *= 0.95
                        
        # Physics Update (utilizing robust acceleration model)
        forward = np.array([math.cos(self.heading) * math.cos(self.pitch),
                           math.sin(self.heading) * math.cos(self.pitch),
                           math.sin(self.pitch)])
        desired_velocity = forward * self.config.max_speed
        self.velocity += (desired_velocity - self.velocity) * self.config.acceleration * dt
        self.pos += self.velocity * dt
        
        # Keep in bounds and above ground
        self.pos[0] = max(50, min(1150, self.pos[0]))
        self.pos[1] = max(50, min(750, self.pos[1]))
        self.pos[2] = max(50, min(self.config.max_altitude, self.pos[2]))
        
        return proj

class DogfightSimulation:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ultimate Hybrid Dogfight Simulation")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.aircraft = []
        self.projectiles = []
        self.ground_targets = []
        self.effects = []
        
        self.camera_height = 1000
        
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.selected_aircraft = None
        self.setup_default_scenario()

    def setup_default_scenario(self):
        """Setup a default dogfight scenario with ground targets"""
        machine_gun = WeaponConfig(WeaponType.MACHINE_GUN, 15, 300, 10, 600, 5000)
        missile = WeaponConfig(WeaponType.MISSILE, 100, 600, 0.5, 700, 6, tracking=True, blast_radius=20)
        cannon = WeaponConfig(WeaponType.CANNON, 50, 250, 3, 750, 100, can_target_ground=True)
        # Increased bomb range to 800 so high-altitude CCIP releases trigger correctly
        bomb = WeaponConfig(WeaponType.BOMB, 200, 800, 1, 0, 6, blast_radius=65, can_target_ground=True)
        
        fighter_config = AircraftConfig(120, 1.0, 1.5, 30, 500, 100, [machine_gun, missile])
        attack_config = AircraftConfig(100, 0.8, 1.2, 20, 400, 150, [cannon, bomb])
        
        self.aircraft = [
            Aircraft(200, 200, 200, "blue", fighter_config, (100, 150, 255)),
            Aircraft(300, 250, 180, "blue", attack_config, (100, 150, 255)),
            Aircraft(1000, 600, 220, "red", attack_config, (255, 100, 100)),
            Aircraft(900, 550, 200, "red", fighter_config, (255, 150, 150)),
        ]
        
        for aircraft in self.aircraft:
            if aircraft.team == "blue": aircraft.heading = 0 
            else: aircraft.heading = math.pi
                
        self.ground_targets = [
            GroundTarget(400, 300, "tank", 80, "red"),
            GroundTarget(450, 320, "tank", 80, "red"),
            GroundTarget(500, 300, "building", 150, "red"),
            GroundTarget(800, 500, "aa_gun", 100, "red"),
            GroundTarget(200, 600, "tank", 80, "blue"),
            GroundTarget(150, 580, "building", 150, "blue"),
        ]

    def create_sample_mission(self):
        if len(self.aircraft) >= 2:
            blue_fighter = self.aircraft[0]
            blue_attacker = self.aircraft[1]
            blue_fighter.add_maneuver(Maneuver(ManeuverType.CLIMB, 5.0, parameters={'altitude': 400}))
            blue_fighter.add_maneuver(Maneuver(ManeuverType.DOGFIGHT, 30.0))
            ground_target = next((gt for gt in self.ground_targets if gt.team == "red" and gt.alive), None)
            if ground_target:
                blue_attacker.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 20.0, target=ground_target))
            blue_attacker.add_maneuver(Maneuver(ManeuverType.FOLLOW, 15.0, target=blue_fighter, 
                                              parameters={'distance': 80, 'offset_angle': math.pi + 0.5}))

    def draw_ground_target(self, target):
        if not target.alive:
            return
        x, y = int(target.pos[0]), int(target.pos[1])
        
        if target.target_type == "tank":
            color = (100, 100, 200) if target.team == "blue" else (200, 100, 100)
            pygame.draw.rect(self.screen, color, (x - 8, y - 5, 16, 10))
            turret_color = tuple(min(255, c + 20) for c in color)
            pygame.draw.circle(self.screen, turret_color, (x, y), 6)
            barrel_end_x = x + math.cos(target.turret_angle) * 12
            barrel_end_y = y + math.sin(target.turret_angle) * 12
            pygame.draw.line(self.screen, turret_color, (x, y), (barrel_end_x, barrel_end_y), 3)
            
        elif target.target_type == "building":
            color = (80, 80, 150) if target.team == "blue" else (150, 80, 80)
            pygame.draw.rect(self.screen, color, (x - 12, y - 12, 24, 24))
            
        elif target.target_type == "aa_gun":
            color = (60, 60, 120) if target.team == "blue" else (120, 60, 60)
            pygame.draw.circle(self.screen, color, (x, y), 10)
            mount_color = tuple(min(255, c + 30) for c in color)
            pygame.draw.circle(self.screen, mount_color, (x, y), 6)
            barrel_end_x = x + math.cos(target.turret_angle) * 18
            barrel_end_y = y + math.sin(target.turret_angle) * 18
            pygame.draw.line(self.screen, mount_color, (x, y), (barrel_end_x, barrel_end_y), 4)
            if target.last_shot_time < 0.2:
                flash_end_x = barrel_end_x + math.cos(target.turret_angle) * 8
                flash_end_y = barrel_end_y + math.sin(target.turret_angle) * 8
                pygame.draw.line(self.screen, (255, 255, 100), (barrel_end_x, barrel_end_y), (flash_end_x, flash_end_y), 6)
                pygame.draw.circle(self.screen, (255, 200, 0), (int(barrel_end_x), int(barrel_end_y)), 4)
            
        # Health bar
        if target.health < target.max_health:
            bar_width = 20
            health_width = int(bar_width * (target.health / target.max_health))
            pygame.draw.rect(self.screen, (255, 0, 0), (x - bar_width // 2, y - target.size - 8, bar_width, 3))
            pygame.draw.rect(self.screen, (0, 255, 0), (x - bar_width // 2, y - target.size - 8, health_width, 3))
            
        # Targeting indicator
        if target.target_type == "aa_gun" and target.can_shoot and abs(target.turret_angle - target.target_turret_angle) > 0.1:
            start_angle = target.turret_angle
            end_angle = target.target_turret_angle
            for i in range(3):
                angle_step = (end_angle - start_angle) * (i + 1) / 4
                if abs(angle_step) > math.pi:
                    angle_step = angle_step - 2 * math.pi if angle_step > 0 else angle_step + 2 * math.pi
                dot_angle = start_angle + angle_step
                dot_x = x + math.cos(dot_angle) * 15
                dot_y = y + math.sin(dot_angle) * 15
                pygame.draw.circle(self.screen, (255, 255, 0), (int(dot_x), int(dot_y)), 1)

    def project_3d_to_2d(self, pos_3d):
        return int(pos_3d[0]), int(pos_3d[1]), pos_3d[2]

    def draw_aircraft(self, aircraft):
        if not aircraft.alive:
            return
        x, y, z = self.project_3d_to_2d(aircraft.pos)
        
        # Shadow
        pygame.draw.circle(self.screen, (50, 50, 50), (x, y), 3)
        
        # Aircraft body
        size = max(8, int(12 - z / 100))
        forward = np.array([math.cos(aircraft.heading), math.sin(aircraft.heading)]) * size
        left = np.array([-math.sin(aircraft.heading), math.cos(aircraft.heading)]) * size * 0.6
        points = [np.array([x, y]) + forward, np.array([x, y]) - forward * 0.3 + left, 
                  np.array([x, y]) - forward, np.array([x, y]) - forward * 0.3 - left]
        
        pygame.draw.polygon(self.screen, aircraft.color, points)
        
        # Altitude line
        if z > 100:
            line_color = tuple(max(50, c - 100) for c in aircraft.color)
            pygame.draw.line(self.screen, line_color, (x, y), (x, y - int(z / 10)), 2)
            
        # Health bar
        health_width = int(20 * (aircraft.health / aircraft.config.health))
        pygame.draw.rect(self.screen, (255, 0, 0), (x - 10, y - size - 10, 20, 4))
        pygame.draw.rect(self.screen, (0, 255, 0), (x - 10, y - size - 10, health_width, 4))
        
        # Team indicator
        team_color = (0, 0, 255) if aircraft.team == "blue" else (255, 0, 0)
        pygame.draw.circle(self.screen, team_color, (x - size - 5, y - size - 5), 3)
        
        # Labels
        if aircraft.current_maneuver:
            self.screen.blit(self.small_font.render(aircraft.current_maneuver.maneuver_type.value[:3].upper(), True, (255, 255, 0)), (x + size + 5, y + size + 5))
        self.screen.blit(self.small_font.render(f"{int(z)}m", True, (255, 255, 255)), (x + size + 5, y - size))

        # Draw CCIP ground crosshair if selected and bomb weapon is active
        if aircraft == self.selected_aircraft and aircraft.ccip is not None:
            cx, cy = int(aircraft.ccip[0]), int(aircraft.ccip[1])
            pygame.draw.circle(self.screen, (255, 0, 0), (cx, cy), 8, 1)
            pygame.draw.line(self.screen, (255, 0, 0), (cx - 10, cy), (cx + 10, cy))
            pygame.draw.line(self.screen, (255, 0, 0), (cx, cy - 10), (cx, cy + 10))

    def draw_projectile(self, projectile):
        if not projectile.active:
            return
        x, y, _ = self.project_3d_to_2d(projectile.pos)
        
        if projectile.config.weapon_type == WeaponType.MACHINE_GUN:
            color, size = (255, 255, 0), 2
        elif projectile.config.weapon_type == WeaponType.MISSILE:
            color, size = (255, 100, 0), 4
        elif projectile.config.weapon_type == WeaponType.BOMB:
            color, size = (255, 150, 100), 6
        else:
            color, size = (255, 200, 0), 3
            
        pygame.draw.circle(self.screen, color, (x, y), size)
        
        # Trail for missiles
        if projectile.config.weapon_type == WeaponType.MISSILE:
            trail_start = projectile.pos - (projectile.velocity / np.linalg.norm(projectile.velocity)) * 20
            trail_x, trail_y, _ = self.project_3d_to_2d(trail_start)
            pygame.draw.line(self.screen, (255, 150, 0), (trail_x, trail_y), (x, y), 2)

    def draw_ui(self):
        ui_surface = pygame.Surface((300, 310))
        ui_surface.set_alpha(180)
        ui_surface.fill((0, 0, 0))
        self.screen.blit(ui_surface, (10, 10))
        
        y_offset = 20
        self.screen.blit(self.font.render("Hybrid Dogfight Simulation", True, (255, 255, 255)), (20, y_offset))
        y_offset += 30
        
        blue_alive = sum(1 for a in self.aircraft if a.team == "blue" and a.alive)
        red_alive = sum(1 for a in self.aircraft if a.team == "red" and a.alive)
        
        self.screen.blit(self.font.render(f"Blue Team: {blue_alive}", True, (100, 150, 255)), (20, y_offset))
        y_offset += 25
        self.screen.blit(self.font.render(f"Red Team: {red_alive}", True, (255, 100, 100)), (20, y_offset))
        y_offset += 25
        
        self.screen.blit(self.font.render(f"Ground Targets: {sum(1 for gt in self.ground_targets if gt.alive)}", True, (200, 200, 200)), (20, y_offset))
        y_offset += 25
        
        self.screen.blit(self.font.render(f"Projectiles: {sum(1 for p in self.projectiles if p.active)}", True, (255, 255, 255)), (20, y_offset))
        y_offset += 25
        
        controls = [
            "P: Pause/Resume",
            "R: Reset | M: Mission",
            "1-4: Select Aircraft",
            "W/S: Switch Weapon | X: Flares",
            "Arrows: Manual Fly | Space: Manual Fire",
            "Q: Dogfight | E: Bomb Target",
            "F: Follow | O: Orbit",
            "G: Flank | I: Intercept",
            "T: Retreat | C: Clear Orders"
        ]
        
        for control in controls:
            self.screen.blit(self.small_font.render(control, True, (200, 200, 200)), (20, y_offset))
            y_offset += 18
            
        if self.selected_aircraft and self.selected_aircraft.alive:
            aircraft = self.selected_aircraft
            info_surface = pygame.Surface((280, 180))
            info_surface.set_alpha(180)
            info_surface.fill((0, 0, 50))
            self.screen.blit(info_surface, (self.width - 290, 10))
            
            info_y = 20
            self.screen.blit(self.font.render(f"{aircraft.team.upper()} Aircraft", True, (255, 255, 255)), (self.width - 280, info_y))
            info_y += 25
            
            stats = [
                f"Health: {aircraft.health:.0f}/{aircraft.config.health}",
                f"Altitude: {aircraft.pos[2]:.0f}m",
                f"Speed: {np.linalg.norm(aircraft.velocity):.0f}",
                f"Weapon: {aircraft.weapons[aircraft.current_weapon].weapon_type.value}",
                f"Ammo: {aircraft.weapons[aircraft.current_weapon].ammo_count}",
                f"Flares: {aircraft.flares}"
            ]
            for stat in stats:
                self.screen.blit(self.small_font.render(stat, True, (255, 255, 255)), (self.width - 280, info_y))
                info_y += 18
                
            if aircraft.current_maneuver:
                self.screen.blit(self.small_font.render(f"Maneuver: {aircraft.current_maneuver.maneuver_type.value}", True, (255, 255, 0)), (self.width - 280, info_y))
                info_y += 18
                self.screen.blit(self.small_font.render(f"Time Left: {aircraft.current_maneuver.duration - aircraft.maneuver_timer:.1f}s", True, (255, 255, 0)), (self.width - 280, info_y))
            else:
                self.screen.blit(self.small_font.render("Status: Default AI / Manual", True, (200, 200, 200)), (self.width - 280, info_y))

    def update(self, dt):
        if self.paused: return
            
        for target in self.ground_targets:
            target.update(dt, self.aircraft, self.projectiles)
            
        for aircraft in self.aircraft:
            if aircraft.alive:
                proj = aircraft.ai_update(dt, self.aircraft, self.ground_targets, is_selected=(aircraft == self.selected_aircraft))
                if proj: self.projectiles.append(proj)
                    
        for projectile in self.projectiles[:]:
            projectile.update(dt, self.aircraft, self.ground_targets, self.effects)
            if not projectile.active: self.projectiles.remove(projectile)
            
        for effect in self.effects[:]:
            effect.update(dt)
            if effect.lifetime <= 0: self.effects.remove(effect)

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.paused = not self.paused
            elif event.key == pygame.K_r:
                self.projectiles.clear()
                self.effects.clear()
                self.setup_default_scenario()
                self.selected_aircraft = None
            elif event.key == pygame.K_m: self.create_sample_mission()
            elif event.key == pygame.K_1 and len([a for a in self.aircraft if a.alive]) > 0: self.selected_aircraft = [a for a in self.aircraft if a.alive][0]
            elif event.key == pygame.K_2 and len([a for a in self.aircraft if a.alive]) > 1: self.selected_aircraft = [a for a in self.aircraft if a.alive][1]
            elif event.key == pygame.K_3 and len([a for a in self.aircraft if a.alive]) > 2: self.selected_aircraft = [a for a in self.aircraft if a.alive][2]
            elif event.key == pygame.K_4 and len([a for a in self.aircraft if a.alive]) > 3: self.selected_aircraft = [a for a in self.aircraft if a.alive][3]
            elif event.key == pygame.K_w and self.selected_aircraft and self.selected_aircraft.alive:
                self.selected_aircraft.current_weapon = (self.selected_aircraft.current_weapon + 1) % len(self.selected_aircraft.weapons)
            elif event.key == pygame.K_s and self.selected_aircraft and self.selected_aircraft.alive:
                self.selected_aircraft.current_weapon = (self.selected_aircraft.current_weapon - 1) % len(self.selected_aircraft.weapons)
            elif event.key == pygame.K_x and self.selected_aircraft and self.selected_aircraft.alive:
                self.selected_aircraft.deploy_flares(self.effects, self.projectiles)
            
            # AI Maneuver Queueing
            elif event.key == pygame.K_q and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_enemy(self.aircraft)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.DOGFIGHT, 30.0, target=tgt))
            elif event.key == pygame.K_e and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_ground_target(self.ground_targets)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    for i, w in enumerate(self.selected_aircraft.weapons):
                        if w.weapon_type == WeaponType.BOMB: self.selected_aircraft.current_weapon = i
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 25.0, target=tgt, parameters={'altitude': 300}))
            elif event.key == pygame.K_f and self.selected_aircraft:
                friendlies = [a for a in self.aircraft if a.team == self.selected_aircraft.team and a != self.selected_aircraft and a.alive]
                if friendlies:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.FOLLOW, 20.0, target=friendlies[0], parameters={'distance': 100, 'offset_angle': math.pi + 0.5}))
            elif event.key == pygame.K_o and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_enemy(self.aircraft)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.ORBIT, 20.0, target=tgt, parameters={'radius': 200, 'speed': 1.0}))
            elif event.key == pygame.K_g and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_enemy(self.aircraft)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.FLANK, 15.0, target=tgt, parameters={'radius': 250, 'side': random.choice(['left', 'right'])}))
            elif event.key == pygame.K_i and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_enemy(self.aircraft)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.INTERCEPT, 15.0, target=tgt))
            elif event.key == pygame.K_t and self.selected_aircraft:
                self.selected_aircraft.clear_maneuvers()
                self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.RETREAT, 10.0))
            elif event.key == pygame.K_c and self.selected_aircraft:
                self.selected_aircraft.clear_maneuvers()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                else: self.handle_input(event)
            
            self.update(dt)
            self.screen.fill((20, 40, 80))
            
            for i in range(0, self.width, 100): pygame.draw.line(self.screen, (40, 60, 100), (i, 0), (i, self.height))
            for i in range(0, self.height, 100): pygame.draw.line(self.screen, (40, 60, 100), (0, i), (self.width, i))
            
            for target in self.ground_targets: self.draw_ground_target(target)
            for projectile in self.projectiles: self.draw_projectile(projectile)
            for aircraft in self.aircraft: self.draw_aircraft(aircraft)
            for effect in self.effects: effect.draw(self.screen, self.project_3d_to_2d)
                
            if self.selected_aircraft and self.selected_aircraft.alive:
                x, y, z = self.project_3d_to_2d(self.selected_aircraft.pos)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 25, 2)
                if self.selected_aircraft.maneuver_queue:
                    self.screen.blit(self.small_font.render(f"Queued: {len(self.selected_aircraft.maneuver_queue)}", True, (255, 255, 0)), (x + 30, y + 20))
            
            self.draw_ui()
            if self.paused: self.screen.blit(self.font.render("PAUSED", True, (255, 255, 0)), (self.width // 2 - 40, 50))
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    simulation = DogfightSimulation()
    simulation.run()