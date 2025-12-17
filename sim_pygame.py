import pygame
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Initialize Pygame
pygame.init()

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
        # Add this to the GroundTarget.update method, right after the turret rotation update:

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
        self.lifetime = weapon_config.range / weapon_config.velocity
        self.age = 0.0
        self.active = True

    def update(self, dt, aircraft_list, ground_targets=None):
        if not self.active:
            return
            
        self.age += dt
        if self.age >= self.lifetime:
            self.active = False
            return

        # Missile tracking
        if self.config.tracking and self.target and self.target.alive:
            target_pos = self.target.pos
            to_target = target_pos - self.pos
            distance = np.linalg.norm(to_target)
            
            if distance > 0:
                # Simple proportional navigation
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

        # Check for hits on aircraft
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                hit_distance = 15.0 if self.config.blast_radius > 0 else 8.0
                
                if distance < hit_distance:
                    aircraft.take_damage(self.config.damage)
                    self.active = False
                    return
                    
        # Check for hits on ground targets
        if ground_targets and self.config.can_target_ground:
            for target in ground_targets:
                if target.team != self.team and target.alive:
                    # Check if projectile is close to ground level and near target
                    if self.pos[2] < 50:  # Near ground
                        distance_2d = np.linalg.norm(self.pos[:2] - target.pos[:2])
                        hit_distance = target.size + (self.config.blast_radius or 10)
                        
                        if distance_2d < hit_distance:
                            target.take_damage(self.config.damage)
                            self.active = False
                            return

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
        
        # Weapon systems
        self.weapons = config.weapons.copy()
        self.current_weapon = 0
        self.weapon_cooldowns = [0.0] * len(self.weapons)
        
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
        
        # Check if weapon can target this type
        if is_ground and not weapon.can_target_ground:
            return False
            
        # Check if target is in range
        to_target = target.pos - self.pos
        distance = np.linalg.norm(to_target)
        
        if distance > weapon.range:
            return False
            
        # Check angle to target
        forward = np.array([math.cos(self.heading), math.sin(self.heading), 0])
        to_target_normalized = to_target / distance
        angle = math.acos(np.clip(np.dot(forward, to_target_normalized), -1, 1))
        
        max_angle = math.pi / 3 if is_ground else math.pi / 4  # Wider angle for ground targets
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
        
        projectile_velocity = self.velocity + forward * weapon.velocity
        
        # For bombs, add gravity effect
        if weapon.weapon_type == WeaponType.BOMB:
            projectile_velocity[2] -= 50  # Initial downward velocity
        
        target = self.target if weapon.tracking else None
        
        return Projectile(fire_pos[0], fire_pos[1], fire_pos[2],
                         projectile_velocity[0], projectile_velocity[1], projectile_velocity[2],
                         weapon, target, self.team)

    def execute_maneuver(self, dt, aircraft_list, ground_targets):
        """Execute the current maneuver"""
        if not self.current_maneuver:
            return
            
        maneuver = self.current_maneuver
        
        if maneuver.maneuver_type == ManeuverType.ORBIT:
            self.execute_orbit(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.DOGFIGHT:
            self.execute_dogfight(dt, maneuver, aircraft_list)
        elif maneuver.maneuver_type == ManeuverType.BOMB_TARGET:
            self.execute_bomb_target(dt, maneuver, ground_targets)
        elif maneuver.maneuver_type == ManeuverType.FOLLOW:
            self.execute_follow(dt, maneuver)
        elif maneuver.maneuver_type == ManeuverType.FLANK:
            self.execute_flank(dt, maneuver, aircraft_list)
        elif maneuver.maneuver_type == ManeuverType.ATTACK_RUN:
            self.execute_attack_run(dt, maneuver, aircraft_list, ground_targets)
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
            time_to_intercept = distance / self.config.max_speed
            predicted_pos = target.pos + target_velocity * time_to_intercept
            
            to_predicted = predicted_pos - self.pos
            desired_heading = math.atan2(to_predicted[1], to_predicted[0])
            
            self.turn_towards_heading(desired_heading, dt)
            
            # Altitude matching with slight advantage
            desired_alt = target.pos[2] + 30
            self.adjust_altitude(desired_alt, dt)
            
            # Fire when in range
            if self.can_fire_at_target(target):
                self.try_fire_weapon()

    def execute_bomb_target(self, dt, maneuver, ground_targets):
        """Bomb run on ground targets"""
        if not maneuver.target or not maneuver.target.alive:
            # Find new ground target
            target, _ = self.find_nearest_ground_target(ground_targets)
            maneuver.target = target
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            to_target = target.pos - self.pos
            distance_2d = np.linalg.norm(to_target[:2])
            
            # Approach phase
            if 'phase' not in self.maneuver_state:
                self.maneuver_state['phase'] = 'approach'
                
            if self.maneuver_state['phase'] == 'approach':
                # Climb to bombing altitude
                bomb_altitude = maneuver.parameters.get('altitude', 300)
                if self.pos[2] < bomb_altitude - 20:
                    self.pitch = min(0.3, self.pitch + dt * 0.5)
                else:
                    self.maneuver_state['phase'] = 'attack'
                    
                # Head towards target
                desired_heading = math.atan2(to_target[1], to_target[0])
                self.turn_towards_heading(desired_heading, dt)
                
            elif self.maneuver_state['phase'] == 'attack':
                # Level flight towards target
                self.pitch *= 0.9
                desired_heading = math.atan2(to_target[1], to_target[0])
                self.turn_towards_heading(desired_heading, dt)
                
                # Switch to bomb weapon if available
                for i, weapon in enumerate(self.weapons):
                    if weapon.weapon_type == WeaponType.BOMB and weapon.ammo_count > 0:
                        self.current_weapon = i
                        break
                
                # Drop bombs when close
                if distance_2d < 100 and self.can_fire_at_target(target, is_ground=True):
                    self.try_fire_weapon()

    def execute_follow(self, dt, maneuver):
        """Follow another aircraft"""
        if not maneuver.target or not maneuver.target.alive:
            return
            
        target = maneuver.target
        follow_distance = maneuver.parameters.get('distance', 100)
        
        # Calculate follow position (behind and slightly offset)
        target_heading = getattr(target, 'heading', 0)
        offset_angle = maneuver.parameters.get('offset_angle', math.pi)
        
        follow_x = target.pos[0] + math.cos(target_heading + offset_angle) * follow_distance
        follow_y = target.pos[1] + math.sin(target_heading + offset_angle) * follow_distance
        follow_z = target.pos[2] + maneuver.parameters.get('altitude_offset', 0)
        
        follow_pos = np.array([follow_x, follow_y, follow_z])
        to_follow_pos = follow_pos - self.pos
        
        desired_heading = math.atan2(to_follow_pos[1], to_follow_pos[0])
        self.turn_towards_heading(desired_heading, dt)
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
            
            # Attack when in position
            distance = np.linalg.norm(to_target)
            if distance < self.weapons[self.current_weapon].range:
                if self.can_fire_at_target(target):
                    self.try_fire_weapon()

    def execute_attack_run(self, dt, maneuver, aircraft_list, ground_targets):
        """High-speed attack run"""
        if not maneuver.target or not maneuver.target.alive:
            return
            
        target = maneuver.target
        to_target = target.pos - self.pos
        distance = np.linalg.norm(to_target)
        
        # Head straight for target at high speed
        desired_heading = math.atan2(to_target[1], to_target[0])
        self.turn_towards_heading(desired_heading, dt, turn_rate_multiplier=2.0)
        
        # Maintain low altitude for ground targets, match altitude for air targets
        if hasattr(target, 'target_type'):  # Ground target
            desired_alt = 100
        else:  # Air target
            desired_alt = target.pos[2]
            
        self.adjust_altitude(desired_alt, dt)
        
        # Fire when in range
        is_ground = hasattr(target, 'target_type')
        if self.can_fire_at_target(target, is_ground):
            self.try_fire_weapon()

    def execute_climb(self, dt, maneuver):
        """Climb to specified altitude"""
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
            time_to_intercept = np.linalg.norm(to_target) / np.linalg.norm(relative_velocity) if np.linalg.norm(relative_velocity) > 0 else 1.0
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
            self.pitch *= 0.95  # Level out

    def try_fire_weapon(self):
        """Attempt to fire current weapon"""
        if self.weapon_cooldowns[self.current_weapon] <= 0:
            return self.fire_weapon()
        return None

    def update_maneuvers(self, dt):
        """Update maneuver system"""
        # Update current maneuver timer
        if self.current_maneuver:
            self.maneuver_timer += dt
            
            # Check if maneuver is complete
            maneuver_complete = False
            
            # Time-based completion
            if self.maneuver_timer >= self.current_maneuver.duration:
                maneuver_complete = True
                
            # Target-based completion (target destroyed)
            if (self.current_maneuver.target and 
                hasattr(self.current_maneuver.target, 'alive') and 
                not self.current_maneuver.target.alive):
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

    def ai_update(self, dt, aircraft_list, ground_targets=None):
        if not self.alive:
            return
            
        # Update weapon cooldowns
        for i in range(len(self.weapon_cooldowns)):
            self.weapon_cooldowns[i] = max(0, self.weapon_cooldowns[i] - dt)
            
        # Update maneuver system
        self.update_maneuvers(dt)
        
        if self.current_maneuver and not self.default_behavior:
            # Execute current maneuver
            self.execute_maneuver(dt, aircraft_list, ground_targets or [])
        else:
            # Default AI behavior
            enemy, enemy_distance = self.find_nearest_enemy(aircraft_list)
            self.target = enemy
            
            if enemy and enemy_distance < 800:
                # Basic engagement
                to_enemy = enemy.pos - self.pos
                desired_heading = math.atan2(to_enemy[1], to_enemy[0])
                self.turn_towards_heading(desired_heading, dt)
                
                if self.can_fire_at_target(enemy):
                    if random.random() < 0.02:
                        projectile = self.fire_weapon()
                        return projectile
                        
        # Update physics
        forward = np.array([math.cos(self.heading) * math.cos(self.pitch),
                           math.sin(self.heading) * math.cos(self.pitch),
                           math.sin(self.pitch)])
        
        desired_velocity = forward * self.config.max_speed
        acceleration = (desired_velocity - self.velocity) * self.config.acceleration * dt
        self.velocity += acceleration
        
        # Update position
        self.pos += self.velocity * dt
        
        # Keep in bounds and above ground
        self.pos[0] = max(50, min(1150, self.pos[0]))
        self.pos[1] = max(50, min(750, self.pos[1]))
        self.pos[2] = max(50, min(self.config.max_altitude, self.pos[2]))
        
        return None

class DogfightSimulation:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Dogfight Simulation with Maneuvers")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.aircraft = []
        self.projectiles = []
        self.ground_targets = []
        
        # Camera for 3D projection
        self.camera_height = 1000
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Selected aircraft for giving orders
        self.selected_aircraft = None
        
        self.setup_default_scenario()

    def setup_default_scenario(self):
        """Setup a default dogfight scenario with ground targets"""
        # Define weapon configurations
        machine_gun = WeaponConfig(
            weapon_type=WeaponType.MACHINE_GUN,
            damage=15,
            range=300,
            fire_rate=10,
            velocity=600,
            ammo_count=5000
        )
        
        missile = WeaponConfig(
            weapon_type=WeaponType.MISSILE,
            damage=100,
            range=600,
            fire_rate=0.5,
            velocity=700,
            ammo_count=6,
            tracking=True,
            blast_radius=20
        )
        
        cannon = WeaponConfig(
            weapon_type=WeaponType.CANNON,
            damage=50,
            range=250,
            fire_rate=3,
            velocity=750,
            ammo_count=100,
            can_target_ground=True
        )
        
        bomb = WeaponConfig(
            weapon_type=WeaponType.BOMB,
            damage=200,
            range=150,
            fire_rate=1,
            velocity=100,
            ammo_count=6,
            blast_radius=40,
            can_target_ground=True
        )
        
        # Fighter aircraft config
        fighter_config = AircraftConfig(
            max_speed=120,
            acceleration=1.0,
            turn_rate=1.5,
            climb_rate=30,
            max_altitude=500,
            health=100,
            weapons=[machine_gun, missile]
        )
        
        # Attack aircraft config
        attack_config = AircraftConfig(
            max_speed=100,
            acceleration=0.8,
            turn_rate=1.2,
            climb_rate=20,
            max_altitude=400,
            health=150,
            weapons=[cannon, bomb]
        )
        
        # Create aircraft
        self.aircraft = [
            Aircraft(200, 200, 200, "blue", fighter_config, (100, 150, 255)),
            Aircraft(300, 250, 180, "blue", attack_config, (100, 150, 255)),
            Aircraft(1000, 600, 220, "red", attack_config, (255, 100, 100)),
            Aircraft(900, 550, 200, "red", fighter_config, (255, 150, 150)),
        ]
        
        # Set initial headings
        for aircraft in self.aircraft:
            if aircraft.team == "blue":
                aircraft.heading = 0  # Face right
            else:
                aircraft.heading = math.pi  # Face left
                
        # Create ground targets
        self.ground_targets = [
            GroundTarget(400, 300, "tank", 80, "red"),
            GroundTarget(450, 320, "tank", 80, "red"),
            GroundTarget(500, 300, "building", 150, "red"),
            GroundTarget(800, 500, "aa_gun", 100, "red"),
            GroundTarget(200, 600, "tank", 80, "blue"),
            GroundTarget(150, 580, "building", 150, "blue"),
        ]

    def add_maneuver_to_aircraft(self, aircraft_index, maneuver):
        """Add a maneuver to specified aircraft"""
        if 0 <= aircraft_index < len(self.aircraft):
            self.aircraft[aircraft_index].add_maneuver(maneuver)

    def create_sample_mission(self):
        """Create a sample mission with predefined maneuvers"""
        if len(self.aircraft) >= 2:
            # Blue team attack mission
            blue_fighter = self.aircraft[0]
            blue_attacker = self.aircraft[1]
            
            # Fighter: Climb, then dogfight
            blue_fighter.add_maneuver(Maneuver(ManeuverType.CLIMB, 5.0, parameters={'altitude': 400}))
            blue_fighter.add_maneuver(Maneuver(ManeuverType.DOGFIGHT, 30.0))
            
            # Attacker: Bomb ground targets, then support
            ground_target = next((gt for gt in self.ground_targets if gt.team == "red" and gt.alive), None)
            if ground_target:
                blue_attacker.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 20.0, target=ground_target))
            blue_attacker.add_maneuver(Maneuver(ManeuverType.FOLLOW, 15.0, target=blue_fighter, 
                                              parameters={'distance': 80, 'offset_angle': math.pi + 0.5}))

    def draw_ground_target(self, target):
        """Draw a ground target with rotating turret if applicable"""
        if not target.alive:
            return
            
        x, y = int(target.pos[0]), int(target.pos[1])
        
        # Different shapes for different types
        if target.target_type == "tank":
            # Draw tank as rectangle with turret
            color = (100, 100, 100)
            if target.team == "blue":
                color = (100, 100, 200)
            elif target.team == "red":
                color = (200, 100, 100)
                
            # Tank body
            pygame.draw.rect(self.screen, color, (x - 8, y - 5, 16, 10))
            
            # Rotating turret
            turret_color = tuple(min(255, c + 20) for c in color)  # Slightly brighter
            pygame.draw.circle(self.screen, turret_color, (x, y), 6)
            
            # Turret barrel
            barrel_length = 12
            barrel_end_x = x + math.cos(target.turret_angle) * barrel_length
            barrel_end_y = y + math.sin(target.turret_angle) * barrel_length
            pygame.draw.line(self.screen, turret_color, (x, y), (barrel_end_x, barrel_end_y), 3)
            
        elif target.target_type == "building":
            # Draw building as larger rectangle
            color = (80, 80, 80)
            if target.team == "blue":
                color = (80, 80, 150)
            elif target.team == "red":
                color = (150, 80, 80)
                
            pygame.draw.rect(self.screen, color, (x - 12, y - 12, 24, 24))
            
        elif target.target_type == "aa_gun":
            # Draw AA gun as circle with rotating barrel
            color = (60, 60, 60)
            if target.team == "blue":
                color = (60, 60, 120)
            elif target.team == "red":
                color = (120, 60, 60)
            
            # Base platform
            pygame.draw.circle(self.screen, color, (x, y), 10)
            
            # Rotating gun mount
            mount_color = tuple(min(255, c + 30) for c in color)
            pygame.draw.circle(self.screen, mount_color, (x, y), 6)
            
            # Gun barrel (longer than tank barrel)
            barrel_length = 18
            barrel_end_x = x + math.cos(target.turret_angle) * barrel_length
            barrel_end_y = y + math.sin(target.turret_angle) * barrel_length
            pygame.draw.line(self.screen, mount_color, (x, y), (barrel_end_x, barrel_end_y), 4)
            
            # Muzzle flash effect when recently fired
            if target.last_shot_time < 0.2:  # Show muzzle flash for 0.2 seconds after firing
                flash_length = 8
                flash_end_x = barrel_end_x + math.cos(target.turret_angle) * flash_length
                flash_end_y = barrel_end_y + math.sin(target.turret_angle) * flash_length
                # Bright yellow/orange muzzle flash
                pygame.draw.line(self.screen, (255, 255, 100), (barrel_end_x, barrel_end_y), (flash_end_x, flash_end_y), 6)
                pygame.draw.circle(self.screen, (255, 200, 0), (int(barrel_end_x), int(barrel_end_y)), 4)
            
        # Health bar for ground targets
        if target.health < target.max_health:
            bar_width = 20
            bar_height = 3
            health_ratio = target.health / target.max_health
            health_width = int(bar_width * health_ratio)
            
            pygame.draw.rect(self.screen, (255, 0, 0), 
                            (x - bar_width//2, y - target.size - 8, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0, 255, 0), 
                            (x - bar_width//2, y - target.size - 8, health_width, bar_height))
            
        # Show targeting indicator for AA guns when they're tracking
        if (target.target_type == "aa_gun" and target.can_shoot and 
            abs(target.turret_angle - target.target_turret_angle) > 0.1):
            # Draw a small arc showing the turret is turning
            arc_radius = 15
            start_angle = target.turret_angle
            end_angle = target.target_turret_angle
            
            # Draw small dots to show rotation direction
            for i in range(3):
                angle_step = (end_angle - start_angle) * (i + 1) / 4
                if abs(angle_step) > math.pi:
                    angle_step = angle_step - 2*math.pi if angle_step > 0 else angle_step + 2*math.pi
                dot_angle = start_angle + angle_step
                dot_x = x + math.cos(dot_angle) * arc_radius
                dot_y = y + math.sin(dot_angle) * arc_radius
                pygame.draw.circle(self.screen, (255, 255, 0), (int(dot_x), int(dot_y)), 1)

    def project_3d_to_2d(self, pos_3d):
        """Project 3D position to 2D screen coordinates"""
        x, y, z = pos_3d
        # Simple orthographic projection with height indication
        screen_x = x
        screen_y = y
        return int(screen_x), int(screen_y), z

    def draw_aircraft(self, aircraft):
        """Draw an aircraft with 3D visual cues"""
        if not aircraft.alive:
            return
            
        x, y, z = self.project_3d_to_2d(aircraft.pos)
        
        # Draw altitude shadow
        shadow_color = (50, 50, 50)
        pygame.draw.circle(self.screen, shadow_color, (x, y), 3)
        
        # Draw aircraft at projected height
        size = max(8, int(12 - z / 100))
        
        # Draw aircraft body
        forward = np.array([math.cos(aircraft.heading), math.sin(aircraft.heading)]) * size
        left = np.array([-math.sin(aircraft.heading), math.cos(aircraft.heading)]) * size * 0.6
        
        nose = np.array([x, y]) + forward
        left_wing = np.array([x, y]) - forward * 0.3 + left
        right_wing = np.array([x, y]) - forward * 0.3 - left
        tail = np.array([x, y]) - forward
        
        points = [nose, left_wing, tail, right_wing]
        pygame.draw.polygon(self.screen, aircraft.color, points)
        
        # Draw altitude line
        if z > 100:
            line_color = tuple(max(50, c - 100) for c in aircraft.color)
            pygame.draw.line(self.screen, line_color, (x, y), (x, y - int(z / 10)), 2)
            
        # Draw health bar
        bar_width = 20
        bar_height = 4
        health_ratio = aircraft.health / aircraft.config.health
        health_width = int(bar_width * health_ratio)
        
        pygame.draw.rect(self.screen, (255, 0, 0), 
                        (x - bar_width//2, y - size - 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 255, 0), 
                        (x - bar_width//2, y - size - 10, health_width, bar_height))
        
        # Draw team indicator
        team_color = (0, 0, 255) if aircraft.team == "blue" else (255, 0, 0)
        pygame.draw.circle(self.screen, team_color, (x - size - 5, y - size - 5), 3)
        
        # Draw current maneuver indicator
        if aircraft.current_maneuver:
            maneuver_text = aircraft.current_maneuver.maneuver_type.value[:3].upper()
            text_surface = self.small_font.render(maneuver_text, True, (255, 255, 0))
            self.screen.blit(text_surface, (x + size + 5, y + size + 5))
        
        # Draw altitude text
        alt_text = f"{int(z)}m"
        text_surface = self.small_font.render(alt_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (x + size + 5, y - size))

    def draw_projectile(self, projectile):
        """Draw a projectile"""
        if not projectile.active:
            return
            
        x, y, z = self.project_3d_to_2d(projectile.pos)
        
        # Different colors for different weapon types
        if projectile.config.weapon_type == WeaponType.MACHINE_GUN:
            color = (255, 255, 0)
            size = 2
        elif projectile.config.weapon_type == WeaponType.MISSILE:
            color = (255, 100, 0)
            size = 4
        elif projectile.config.weapon_type == WeaponType.BOMB:
            color = (255, 150, 100)
            size = 6
        else:  # Cannon
            color = (255, 200, 0)
            size = 3
            
        pygame.draw.circle(self.screen, color, (x, y), size)
        
        # Draw trail for missiles
        if projectile.config.weapon_type == WeaponType.MISSILE:
            trail_length = 20
            vel_normalized = projectile.velocity / np.linalg.norm(projectile.velocity)
            trail_start = projectile.pos - vel_normalized * trail_length
            trail_x, trail_y, _ = self.project_3d_to_2d(trail_start)
            pygame.draw.line(self.screen, (255, 150, 0), (trail_x, trail_y), (x, y), 2)

    def draw_ui(self):
        """Draw user interface"""
        # Background for UI
        ui_surface = pygame.Surface((300, 280))
        ui_surface.set_alpha(180)
        ui_surface.fill((0, 0, 0))
        self.screen.blit(ui_surface, (10, 10))
        
        y_offset = 20
        
        # Title
        title = self.font.render("Dogfight Simulation", True, (255, 255, 255))
        self.screen.blit(title, (20, y_offset))
        y_offset += 30
        
        # Team status
        blue_alive = sum(1 for a in self.aircraft if a.team == "blue" and a.alive)
        red_alive = sum(1 for a in self.aircraft if a.team == "red" and a.alive)
        
        blue_text = self.font.render(f"Blue Team: {blue_alive}", True, (100, 150, 255))
        red_text = self.font.render(f"Red Team: {red_alive}", True, (255, 100, 100))
        
        self.screen.blit(blue_text, (20, y_offset))
        y_offset += 25
        self.screen.blit(red_text, (20, y_offset))
        y_offset += 25
        
        # Ground targets status
        ground_alive = sum(1 for gt in self.ground_targets if gt.alive)
        ground_text = self.font.render(f"Ground Targets: {ground_alive}", True, (200, 200, 200))
        self.screen.blit(ground_text, (20, y_offset))
        y_offset += 25
        
        # Active projectiles
        active_projectiles = sum(1 for p in self.projectiles if p.active)
        proj_text = self.font.render(f"Projectiles: {active_projectiles}", True, (255, 255, 255))
        self.screen.blit(proj_text, (20, y_offset))
        y_offset += 25
        
        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Simulation",
            "M: Sample Mission",
            "1-4: Select Aircraft",
            "W/S: Change Weapon"
            "Q: Dogfight Order",
            "E: Bomb Target Order",
            "F: Follow Order",
            "C: Clear Orders"
        ]
        
        for control in controls:
            control_text = self.small_font.render(control, True, (200, 200, 200))
            self.screen.blit(control_text, (20, y_offset))
            y_offset += 18
            
        # Selected aircraft info
        if self.selected_aircraft and self.selected_aircraft.alive:
            aircraft = self.selected_aircraft
            
            info_surface = pygame.Surface((280, 160))
            info_surface.set_alpha(180)
            info_surface.fill((0, 0, 50))
            self.screen.blit(info_surface, (self.width - 290, 10))
            
            info_y = 20
            name_text = self.font.render(f"{aircraft.team.upper()} Aircraft", True, (255, 255, 255))
            self.screen.blit(name_text, (self.width - 280, info_y))
            info_y += 25
            
            health_text = self.small_font.render(f"Health: {aircraft.health:.0f}/{aircraft.config.health}", True, (255, 255, 255))
            self.screen.blit(health_text, (self.width - 280, info_y))
            info_y += 18
            
            alt_text = self.small_font.render(f"Altitude: {aircraft.pos[2]:.0f}m", True, (255, 255, 255))
            self.screen.blit(alt_text, (self.width - 280, info_y))
            info_y += 18
            
            speed = np.linalg.norm(aircraft.velocity)
            speed_text = self.small_font.render(f"Speed: {speed:.0f}", True, (255, 255, 255))
            self.screen.blit(speed_text, (self.width - 280, info_y))
            info_y += 18
            
            weapon = aircraft.weapons[aircraft.current_weapon]
            weapon_text = self.small_font.render(f"Weapon: {weapon.weapon_type.value}", True, (255, 255, 255))
            self.screen.blit(weapon_text, (self.width - 280, info_y))
            info_y += 18
            
            ammo_text = self.small_font.render(f"Ammo: {weapon.ammo_count}", True, (255, 255, 255))
            self.screen.blit(ammo_text, (self.width - 280, info_y))
            info_y += 18
            
            # Current maneuver
            if aircraft.current_maneuver:
                maneuver_name = aircraft.current_maneuver.maneuver_type.value
                time_left = aircraft.current_maneuver.duration - aircraft.maneuver_timer
                maneuver_text = self.small_font.render(f"Maneuver: {maneuver_name}", True, (255, 255, 0))
                time_text = self.small_font.render(f"Time Left: {time_left:.1f}s", True, (255, 255, 0))
                self.screen.blit(maneuver_text, (self.width - 280, info_y))
                info_y += 18
                self.screen.blit(time_text, (self.width - 280, info_y))
            else:
                status_text = self.small_font.render("Status: Default AI", True, (200, 200, 200))
                self.screen.blit(status_text, (self.width - 280, info_y))

    def update(self, dt):
        """Update simulation"""
        if self.paused:
            return
            
        # Update ground targets
        for target in self.ground_targets:
            target.update(dt, self.aircraft, self.projectiles)
            
        # Update aircraft
        for aircraft in self.aircraft:
            if aircraft.alive:
                projectile = aircraft.ai_update(dt, self.aircraft, self.ground_targets)
                if projectile:
                    self.projectiles.append(projectile)
                    
        # Update projectiles
        for projectile in self.projectiles[:]:
            projectile.update(dt, self.aircraft, self.ground_targets)
            if not projectile.active:
                self.projectiles.remove(projectile)

    def find_nearest_enemy_aircraft(self, from_aircraft):
        """Find nearest enemy aircraft to given aircraft"""
        nearest = None
        min_dist = float('inf')
        
        for aircraft in self.aircraft:
            if aircraft.team != from_aircraft.team and aircraft.alive:
                dist = np.linalg.norm(aircraft.pos - from_aircraft.pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = aircraft
                    
        return nearest

    def find_nearest_ground_target(self, from_aircraft):
        """Find nearest enemy ground target"""
        nearest = None
        min_dist = float('inf')
        
        for target in self.ground_targets:
            if target.team != from_aircraft.team and target.alive:
                dist = np.linalg.norm(target.pos - from_aircraft.pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = target
                    
        return nearest

    def handle_input(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
                
            elif event.key == pygame.K_r:
                # Reset simulation
                self.projectiles.clear()
                self.setup_default_scenario()
                self.selected_aircraft = None
                
            elif event.key == pygame.K_m:
                # Run sample mission
                self.create_sample_mission()
                
            elif event.key == pygame.K_1:
                alive_aircraft = [a for a in self.aircraft if a.alive]
                if len(alive_aircraft) > 0:
                    self.selected_aircraft = alive_aircraft[0]
                    
            elif event.key == pygame.K_2:
                alive_aircraft = [a for a in self.aircraft if a.alive]
                if len(alive_aircraft) > 1:
                    self.selected_aircraft = alive_aircraft[1]
                    
            elif event.key == pygame.K_3:
                alive_aircraft = [a for a in self.aircraft if a.alive]
                if len(alive_aircraft) > 2:
                    self.selected_aircraft = alive_aircraft[2]
                    
            elif event.key == pygame.K_4:
                alive_aircraft = [a for a in self.aircraft if a.alive]
                if len(alive_aircraft) > 3:
                    self.selected_aircraft = alive_aircraft[3]
            elif event.key == pygame.K_w:
                if hasattr(self, 'selected_aircraft') and self.selected_aircraft and self.selected_aircraft.alive:
                    self.selected_aircraft.current_weapon = (self.selected_aircraft.current_weapon + 1) % len(self.selected_aircraft.weapons)
                    
            elif event.key == pygame.K_s:
                if hasattr(self, 'selected_aircraft') and self.selected_aircraft and self.selected_aircraft.alive:
                    self.selected_aircraft.current_weapon = (self.selected_aircraft.current_weapon - 1) % len(self.selected_aircraft.weapons)
            # Maneuver commands for selected aircraft
            elif event.key == pygame.K_q and self.selected_aircraft:
                # Dogfight order
                target = self.find_nearest_enemy_aircraft(self.selected_aircraft)
                if target:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(
                        Maneuver(ManeuverType.DOGFIGHT, 30.0, target=target)
                    )
                    
            elif event.key == pygame.K_e and self.selected_aircraft:
                # Bomb target order
                target = self.find_nearest_ground_target(self.selected_aircraft)
                if target:
                    self.selected_aircraft.clear_maneuvers()
                    # Switch to bomb if available
                    for i, weapon in enumerate(self.selected_aircraft.weapons):
                        if weapon.weapon_type == WeaponType.BOMB:
                            self.selected_aircraft.current_weapon = i
                            break
                    self.selected_aircraft.add_maneuver(
                        Maneuver(ManeuverType.BOMB_TARGET, 25.0, target=target, 
                                parameters={'altitude': 300})
                    )
                    
            elif event.key == pygame.K_f and self.selected_aircraft:
                # Follow order - follow another friendly aircraft
                friendly_aircraft = [a for a in self.aircraft 
                                   if a.team == self.selected_aircraft.team and 
                                   a != self.selected_aircraft and a.alive]
                if friendly_aircraft:
                    target = friendly_aircraft[0]
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(
                        Maneuver(ManeuverType.FOLLOW, 20.0, target=target,
                                parameters={'distance': 100, 'offset_angle': math.pi + 0.5})
                    )
                    
            elif event.key == pygame.K_c and self.selected_aircraft:
                # Clear all maneuvers
                self.selected_aircraft.clear_maneuvers()
                
            # Additional maneuver hotkeys
            elif event.key == pygame.K_o and self.selected_aircraft:
                # Orbit maneuver
                enemy = self.find_nearest_enemy_aircraft(self.selected_aircraft)
                if enemy:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(
                        Maneuver(ManeuverType.ORBIT, 20.0, target=enemy,
                                parameters={'radius': 200, 'speed': 1.0})
                    )
                    
            elif event.key == pygame.K_g and self.selected_aircraft:
                # Flank maneuver
                enemy = self.find_nearest_enemy_aircraft(self.selected_aircraft)
                if enemy:
                    self.selected_aircraft.clear_maneuvers()
                    side = random.choice(['left', 'right'])
                    self.selected_aircraft.add_maneuver(
                        Maneuver(ManeuverType.FLANK, 15.0, target=enemy,
                                parameters={'radius': 250, 'side': side})
                    )
                    
            elif event.key == pygame.K_i and self.selected_aircraft:
                # Intercept maneuver
                enemy = self.find_nearest_enemy_aircraft(self.selected_aircraft)
                if enemy:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(
                        Maneuver(ManeuverType.INTERCEPT, 15.0, target=enemy)
                    )
                    
            elif event.key == pygame.K_t and self.selected_aircraft:
                # Retreat maneuver
                self.selected_aircraft.clear_maneuvers()
                self.selected_aircraft.add_maneuver(
                    Maneuver(ManeuverType.RETREAT, 10.0)
                )

    def run(self):
        """Main simulation loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.handle_input(event)
            
            # Update simulation
            self.update(dt)
            
            # Draw everything
            self.screen.fill((20, 40, 80))  # Sky blue background
            
            # Draw grid for reference
            grid_color = (40, 60, 100)
            for i in range(0, self.width, 100):
                pygame.draw.line(self.screen, grid_color, (i, 0), (i, self.height))
            for i in range(0, self.height, 100):
                pygame.draw.line(self.screen, grid_color, (0, i), (self.width, i))
            
            # Draw ground targets
            for target in self.ground_targets:
                self.draw_ground_target(target)
                
            # Draw projectiles
            for projectile in self.projectiles:
                self.draw_projectile(projectile)
                
            # Draw aircraft
            for aircraft in self.aircraft:
                self.draw_aircraft(aircraft)
                
            # Highlight selected aircraft
            if self.selected_aircraft and self.selected_aircraft.alive:
                x, y, z = self.project_3d_to_2d(self.selected_aircraft.pos)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 25, 2)
                
                # Draw maneuver queue
                if self.selected_aircraft.maneuver_queue:
                    queue_text = f"Queued: {len(self.selected_aircraft.maneuver_queue)}"
                    text_surface = self.small_font.render(queue_text, True, (255, 255, 0))
                    self.screen.blit(text_surface, (x + 30, y + 20))
            
            # Draw UI
            self.draw_ui()
            
            # Pause indicator
            if self.paused:
                pause_text = self.font.render("PAUSED", True, (255, 255, 0))
                self.screen.blit(pause_text, (self.width // 2 - 40, 50))
            
            pygame.display.flip()
        
        pygame.quit()

def create_complex_scenario():
    """Example of creating a complex scenario with ground targets and maneuvers"""
    sim = DogfightSimulation()
    sim.aircraft.clear()
    sim.ground_targets.clear()
    
    # Create specialized weapon configs
    air_to_air_missile = WeaponConfig(
        weapon_type=WeaponType.MISSILE,
        damage=120,
        range=700,
        fire_rate=0.4,
        velocity=800,
        ammo_count=6,
        tracking=True,
        blast_radius=25
    )
    
    heavy_bomb = WeaponConfig(
        weapon_type=WeaponType.BOMB,
        damage=300,
        range=100,
        fire_rate=0.5,
        velocity=80,
        ammo_count=4,
        blast_radius=50,
        can_target_ground=True
    )
    
    # Fighter-bomber config
    fighter_bomber = AircraftConfig(
        max_speed=110,
        acceleration=0.9,
        turn_rate=1.4,
        climb_rate=25,
        max_altitude=550,
        health=120,
        weapons=[WeaponConfig(WeaponType.CANNON, 60, 300, 4, 700, 80, can_target_ground=True), 
                heavy_bomb, air_to_air_missile]
    )
    
    # Add aircraft with initial maneuvers
    blue_lead = Aircraft(150, 200, 250, "blue", fighter_bomber, (80, 120, 255))
    blue_wing = Aircraft(200, 250, 230, "blue", fighter_bomber, (120, 150, 255))
    
    # Create ground assault force
    sim.ground_targets.extend([
        GroundTarget(600, 400, "tank", 100, "red"),
        GroundTarget(650, 420, "tank", 100, "red"),
        GroundTarget(620, 380, "aa_gun", 120, "red"),
        GroundTarget(700, 400, "building", 200, "red"),
    ])
    
    sim.aircraft = [blue_lead, blue_wing]
    
    # Setup coordinated attack mission
    ground_target = sim.ground_targets[0]
    blue_lead.add_maneuver(Maneuver(ManeuverType.CLIMB, 8.0, parameters={'altitude': 400}))
    blue_lead.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 20.0, target=ground_target, 
                                  parameters={'altitude': 350}))
    blue_lead.add_maneuver(Maneuver(ManeuverType.ORBIT, 15.0, target=ground_target,
                                  parameters={'radius': 300, 'speed': 1.0}))
    
    # Wingman follows then flanks
    blue_wing.add_maneuver(Maneuver(ManeuverType.FOLLOW, 10.0, target=blue_lead,
                                  parameters={'distance': 80, 'offset_angle': math.pi/4}))
    blue_wing.add_maneuver(Maneuver(ManeuverType.FLANK, 15.0, target=ground_target,
                                  parameters={'radius': 200, 'side': 'right'}))
    
    return sim

if __name__ == "__main__":
    # Run default scenario
    simulation = DogfightSimulation()
    
    # Or run complex scenario
    # simulation = create_complex_scenario()
    
    print("Dogfight Simulation Controls:")
    print("=============================")
    print("SPACE: Pause/Resume")
    print("R: Reset simulation")
    print("M: Run sample mission")
    print("1-4: Select aircraft")
    print("W/S: Change Weapon")
    print("")
    print("Selected Aircraft Commands:")
    print("Q: Order dogfight")
    print("E: Order bomb attack")
    print("F: Order follow friendly")
    print("O: Order orbit target")
    print("G: Order flank attack")
    print("I: Order intercept")
    print("T: Order retreat")
    print("C: Clear all orders")
    print("")
    print("Additional Features:")
    print("- Ground targets: tanks, buildings, AA guns")
    print("- AA guns will shoot at aircraft")
    print("- Maneuver queue system")
    print("- Multiple weapon types including bombs")
    print("")
    
    simulation.run()