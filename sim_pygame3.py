import pygame
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Optional
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
    duration: float
    target: Optional[object] = None
    parameters: dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class WeaponConfig:
    weapon_type: WeaponType
    damage: float
    range: float
    fire_rate: float
    velocity: float
    ammo_count: int
    tracking: bool = False
    blast_radius: float = 0.0
    can_target_ground: bool = False

@dataclass
class AircraftConfig:
    max_speed: float
    acceleration: float
    turn_rate: float
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

# --- ENTITIES ---
class GroundTarget:
    def __init__(self, x, y, target_type, health=100, team="neutral"):
        self.pos = np.array([x, y, 0], dtype=float)
        self.target_type = target_type
        self.health = health
        self.max_health = health
        self.team = team
        self.alive = True
        self.size = 15 if target_type == "tank" else 20
        
        self.can_shoot = target_type == "aa_gun"
        self.last_shot_time = 0
        self.range = 400 if self.can_shoot else 0
        self.turret_angle = 0.0
        self.target_turret_angle = 0.0
        self.turret_turn_rate = 2.0
        
    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            self.health = 0
            
    def update(self, dt, aircraft_list, sim):
        if not self.alive: return
        self.update_turret_rotation(dt)
        self.update_tank_rotation(aircraft_list)
            
        if not self.can_shoot: return
        self.last_shot_time += dt
        
        nearest_target = None
        min_distance = float('inf')
        
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                if distance < self.range and distance < min_distance:
                    min_distance = distance
                    nearest_target = aircraft
                    
        if nearest_target:
            to_target = nearest_target.pos - self.pos
            self.target_turret_angle = math.atan2(to_target[1], to_target[0])
            angle_diff = abs(self.turret_angle - self.target_turret_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
                
            if angle_diff < math.pi/12 and self.last_shot_time >= 2.0:
                self.fire_at_target(nearest_target, sim.projectiles)
                self.last_shot_time = 0
        else:
            self.target_turret_angle = math.pi/2 
            
    def update_turret_rotation(self, dt):
        if abs(self.turret_angle - self.target_turret_angle) < 0.01:
            self.turret_angle = self.target_turret_angle
            return
        angle_diff = self.target_turret_angle - self.turret_angle
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
            
        max_rotation = self.turret_turn_rate * dt
        if abs(angle_diff) <= max_rotation:
            self.turret_angle = self.target_turret_angle
        else:
            self.turret_angle += max_rotation if angle_diff > 0 else -max_rotation
        self.turret_angle = self.turret_angle % (2 * math.pi)
    
    def update_tank_rotation(self, aircraft_list):
        if self.target_type == "tank":
            nearest_enemy = None
            min_distance = float('inf')
            for aircraft in aircraft_list:
                if aircraft.team != self.team and aircraft.alive:
                    distance = np.linalg.norm(aircraft.pos - self.pos)
                    if distance < 300 and distance < min_distance:
                        min_distance = distance
                        nearest_enemy = aircraft
            if nearest_enemy:
                to_enemy = nearest_enemy.pos - self.pos
                self.target_turret_angle = math.atan2(to_enemy[1], to_enemy[0])
            else:
                self.target_turret_angle = math.pi/2

    def fire_at_target(self, target, projectiles_list):
        to_target = target.pos - self.pos
        distance = np.linalg.norm(to_target)
        direction = to_target / distance
        
        aa_weapon = WeaponConfig(
            weapon_type=WeaponType.CANNON, damage=40, range=self.range, fire_rate=0.5,
            velocity=600, ammo_count=999, blast_radius=10
        )
        turret_offset = np.array([math.cos(self.turret_angle), math.sin(self.turret_angle), 0]) * 10
        fire_pos = self.pos + turret_offset
        
        projectiles_list.append(Projectile(
            fire_pos[0], fire_pos[1], fire_pos[2],
            direction[0] * aa_weapon.velocity, direction[1] * aa_weapon.velocity, direction[2] * aa_weapon.velocity,
            aa_weapon, None, self.team
        ))

class Projectile:
    def __init__(self, x, y, z, vx, vy, vz, weapon_config, target=None, team=None):
        self.pos = np.array([x, y, z], dtype=float)
        self.velocity = np.array([vx, vy, vz], dtype=float)
        self.config = weapon_config
        self.target = target
        self.team = team
        self.lifetime = weapon_config.range / weapon_config.velocity if weapon_config.velocity > 0 else 5.0
        self.age = 0.0
        self.active = True

    def update(self, dt, sim):
        if not self.active: return
        self.age += dt
        
        # Realistic Gravity for Bombs
        if self.config.weapon_type == WeaponType.BOMB:
            self.velocity[2] -= GRAVITY * dt
        
        # Missile tracking
        if self.config.tracking and self.target and self.target.alive:
            to_target = self.target.pos - self.pos
            distance = np.linalg.norm(to_target)
            if distance > 0:
                desired_velocity = (to_target / distance) * self.config.velocity
                turn_rate = 5.0
                max_turn = turn_rate * dt
                current_dir = self.velocity / np.linalg.norm(self.velocity)
                desired_dir = desired_velocity / np.linalg.norm(desired_velocity)
                angle_diff = np.arccos(np.clip(np.dot(current_dir, desired_dir), -1, 1))
                if angle_diff > 0:
                    turn_amount = min(max_turn, angle_diff)
                    self.velocity = self.velocity + (desired_velocity - self.velocity) * (turn_amount / angle_diff)
                    self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * self.config.velocity

        self.pos += self.velocity * dt

        # Ground impact
        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.explode(sim)
            return

        if self.age >= self.lifetime:
            self.active = False
            return

        # Aircraft proximity
        for aircraft in sim.aircraft:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                hit_distance = 15.0 if self.config.blast_radius > 0 else 8.0
                if distance < hit_distance:
                    self.explode(sim)
                    return

    def explode(self, sim):
        self.active = False
        radius = max(self.config.blast_radius, 15)
        sim.effects.append(Effect('explosion', self.pos.copy(), radius))
        
        # Damage aircraft
        for aircraft in sim.aircraft:
            if aircraft.team != self.team and aircraft.alive:
                dist = np.linalg.norm(aircraft.pos - self.pos)
                if dist < radius:
                    aircraft.take_damage(self.config.damage * (1 - dist/radius))
                    
        # Damage ground targets
        for target in sim.ground_targets:
            if target.team != self.team and target.alive:
                dist = np.linalg.norm(target.pos - self.pos)
                if dist < radius + target.size:
                    target.take_damage(self.config.damage * (1 - dist/(radius + target.size)))


class Aircraft:
    def __init__(self, x, y, z, team, config, color):
        self.pos = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.heading = 0.0
        self.pitch = 0.0
        self.team = team
        self.config = config
        self.color = color
        self.health = config.health
        self.alive = True
        
        self.weapons = config.weapons.copy()
        self.current_weapon = 0
        self.weapon_cooldowns = [0.0] * len(self.weapons)
        self.flares = 10
        self.flare_cooldown = 0
        self.ccip = None
        
        self.maneuver_queue = []
        self.current_maneuver = None
        self.maneuver_timer = 0.0
        self.maneuver_state = {}
        
        self.target = None
        self.default_behavior = True

    def add_maneuver(self, maneuver):
        self.maneuver_queue.append(maneuver)
        self.default_behavior = False

    def clear_maneuvers(self):
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

    def deploy_flares(self, sim):
        if self.flares > 0 and self.flare_cooldown <= 0:
            self.flares -= 1
            self.flare_cooldown = 2.0
            sim.effects.append(Effect('flares', self.pos.copy(), 5))
            for p in sim.projectiles:
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

    def fire_weapon(self):
        if self.current_weapon >= len(self.weapons): return None
        weapon = self.weapons[self.current_weapon]
        if weapon.ammo_count <= 0 or self.weapon_cooldowns[self.current_weapon] > 0: return None
            
        weapon.ammo_count -= 1
        self.weapon_cooldowns[self.current_weapon] = 1.0 / weapon.fire_rate
        
        forward = np.array([math.cos(self.heading), math.sin(self.heading), math.sin(self.pitch)])
        fire_pos = self.pos + forward * 20
        
        projectile_velocity = self.velocity.copy()
        if weapon.weapon_type != WeaponType.BOMB:
            projectile_velocity += forward * weapon.velocity
        
        target = self.target if weapon.tracking else None
        return Projectile(fire_pos[0], fire_pos[1], fire_pos[2],
                         projectile_velocity[0], projectile_velocity[1], projectile_velocity[2],
                         weapon, target, self.team)

    def execute_maneuver(self, dt, sim):
        if not self.current_maneuver: return
        maneuver = self.current_maneuver
        
        if maneuver.maneuver_type == ManeuverType.BOMB_TARGET:
            self.execute_bomb_target(dt, maneuver, sim)
        elif maneuver.maneuver_type == ManeuverType.DOGFIGHT:
            self.execute_dogfight(dt, maneuver, sim.aircraft)
        elif maneuver.maneuver_type == ManeuverType.FOLLOW:
            self.execute_follow(dt, maneuver)
        else:
            # Fallback basic patrol for other types in this streamlined merge
            self.pitch *= 0.95

    def execute_dogfight(self, dt, maneuver, aircraft_list):
        if not maneuver.target or not maneuver.target.alive:
            enemy, _ = self.find_nearest_enemy(aircraft_list)
            maneuver.target = enemy
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            to_target = target.pos - self.pos
            distance = np.linalg.norm(to_target)
            
            target_velocity = getattr(target, 'velocity', np.zeros(3))
            time_to_intercept = distance / max(1, self.config.max_speed)
            predicted_pos = target.pos + target_velocity * time_to_intercept
            
            to_predicted = predicted_pos - self.pos
            desired_heading = math.atan2(to_predicted[1], to_predicted[0])
            self.turn_towards_heading(desired_heading, dt)
            
            self.adjust_altitude(target.pos[2] + 30, dt)
            
            forward = np.array([math.cos(self.heading), math.sin(self.heading), 0])
            to_tgt_norm = to_target / max(1, distance)
            if distance < self.weapons[self.current_weapon].range and np.dot(forward, to_tgt_norm) > 0.8:
                return self.try_fire_weapon()

    def execute_bomb_target(self, dt, maneuver, sim):
        if not maneuver.target or not maneuver.target.alive:
            target, _ = self.find_nearest_ground_target(sim.ground_targets)
            maneuver.target = target
            
        if maneuver.target and maneuver.target.alive:
            target = maneuver.target
            to_target = target.pos - self.pos
            
            if 'phase' not in self.maneuver_state:
                self.maneuver_state['phase'] = 'approach'
                for i, w in enumerate(self.weapons):
                    if w.weapon_type == WeaponType.BOMB:
                        self.current_weapon = i
                
            if self.maneuver_state['phase'] == 'approach':
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
                
                # Check realistic CCIP firing solution
                if self.ccip is not None:
                    dist_to_impact = np.linalg.norm(self.ccip - target.pos[:2])
                    if dist_to_impact < 30:
                        proj = self.try_fire_weapon()
                        if proj:
                            self.maneuver_state['phase'] = 'egress'
                        return proj

            elif self.maneuver_state['phase'] == 'egress':
                self.pitch = min(0.4, self.pitch + dt * 0.5)

    def execute_follow(self, dt, maneuver):
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

    def turn_towards_heading(self, desired_heading, dt, turn_rate_multiplier=1.0):
        heading_diff = desired_heading - self.heading
        while heading_diff > math.pi: heading_diff -= 2 * math.pi
        while heading_diff < -math.pi: heading_diff += 2 * math.pi
            
        max_turn = self.config.turn_rate * dt * turn_rate_multiplier
        turn_amount = max(-max_turn, min(max_turn, heading_diff))
        self.heading += turn_amount

    def adjust_altitude(self, desired_altitude, dt):
        alt_diff = desired_altitude - self.pos[2]
        if alt_diff > 20: self.pitch = min(0.3, self.pitch + dt * 0.5)
        elif alt_diff < -20: self.pitch = max(-0.3, self.pitch - dt * 0.5)
        else: self.pitch *= 0.95

    def try_fire_weapon(self):
        if self.weapon_cooldowns[self.current_weapon] <= 0:
            return self.fire_weapon()
        return None

    def manual_override(self, dt, sim):
        keys = pygame.key.get_pressed()
        self.clear_maneuvers() # Clear AI if manually flying
        if keys[pygame.K_LEFT]: self.heading -= self.config.turn_rate * dt
        if keys[pygame.K_RIGHT]: self.heading += self.config.turn_rate * dt
        if keys[pygame.K_UP]: self.pitch = max(-0.5, self.pitch - dt)
        elif keys[pygame.K_DOWN]: self.pitch = min(0.5, self.pitch + dt)
        else: self.pitch *= 0.95
        
        if keys[pygame.K_SPACE]:
            proj = self.try_fire_weapon()
            if proj: sim.projectiles.append(proj)

    def ai_update(self, dt, sim, is_selected=False):
        if not self.alive: return
        
        for i in range(len(self.weapon_cooldowns)):
            self.weapon_cooldowns[i] = max(0, self.weapon_cooldowns[i] - dt)
        if self.flare_cooldown > 0: self.flare_cooldown -= dt
            
        if self.weapons[self.current_weapon].weapon_type == WeaponType.BOMB:
            self.ccip = self.calculate_ccip()
        else:
            self.ccip = None
            
        keys = pygame.key.get_pressed()
        if is_selected and (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_SPACE]):
            self.manual_override(dt, sim)
        else:
            if self.current_maneuver:
                self.maneuver_timer += dt
                if self.maneuver_timer >= self.current_maneuver.duration or (self.current_maneuver.target and not getattr(self.current_maneuver.target, 'alive', True)):
                    self.current_maneuver = None
                    self.maneuver_timer = 0.0
                    self.maneuver_state.clear()
            if not self.current_maneuver and self.maneuver_queue:
                self.current_maneuver = self.maneuver_queue.pop(0)
                self.maneuver_timer = 0.0
                self.maneuver_state.clear()
                
            if self.current_maneuver and not self.default_behavior:
                proj = self.execute_maneuver(dt, sim)
                if proj: sim.projectiles.append(proj)
            else:
                self.pitch *= 0.95
                
        # Physics Update
        forward = np.array([math.cos(self.heading) * math.cos(self.pitch),
                           math.sin(self.heading) * math.cos(self.pitch),
                           math.sin(self.pitch)])
        desired_velocity = forward * self.config.max_speed
        self.velocity += (desired_velocity - self.velocity) * self.config.acceleration * dt
        self.pos += self.velocity * dt
        
        self.pos[0] = max(50, min(1150, self.pos[0]))
        self.pos[1] = max(50, min(750, self.pos[1]))
        self.pos[2] = max(50, min(self.config.max_altitude, self.pos[2]))

# --- SIMULATION CORE ---
class DogfightSimulation:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hybrid Dogfight Simulator")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.aircraft = []
        self.projectiles = []
        self.ground_targets = []
        self.effects = []
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.selected_aircraft = None
        self.setup_scenario()

    def setup_scenario(self):
        machine_gun = WeaponConfig(WeaponType.MACHINE_GUN, 15, 300, 10, 600, 5000)
        missile = WeaponConfig(WeaponType.MISSILE, 100, 600, 0.5, 700, 6, tracking=True, blast_radius=20)
        bomb = WeaponConfig(WeaponType.BOMB, 200, 150, 1, 0, 6, blast_radius=60, can_target_ground=True)
        
        fighter_cfg = AircraftConfig(120, 1.0, 1.5, 30, 500, 100, [machine_gun, missile])
        attack_cfg = AircraftConfig(100, 0.8, 1.2, 20, 400, 150, [machine_gun, bomb])
        
        self.aircraft = [
            Aircraft(200, 200, 200, "blue", fighter_cfg, (100, 150, 255)),
            Aircraft(300, 250, 180, "blue", attack_cfg, (100, 150, 255)),
            Aircraft(1000, 600, 220, "red", attack_cfg, (255, 100, 100)),
        ]
        self.aircraft[2].heading = math.pi
        self.selected_aircraft = self.aircraft[1]
        
        self.ground_targets = [
            GroundTarget(600, 400, "aa_gun", 100, "red"),
            GroundTarget(650, 420, "tank", 80, "red"),
            GroundTarget(550, 380, "building", 150, "red"),
        ]

    def project_3d_to_2d(self, pos_3d):
        return int(pos_3d[0]), int(pos_3d[1]), pos_3d[2]

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.selected_aircraft is None: self.paused = not self.paused
            elif event.key == pygame.K_1 and len(self.aircraft) > 0: self.selected_aircraft = self.aircraft[0]
            elif event.key == pygame.K_2 and len(self.aircraft) > 1: self.selected_aircraft = self.aircraft[1]
            elif event.key == pygame.K_w and self.selected_aircraft: 
                self.selected_aircraft.current_weapon = (self.selected_aircraft.current_weapon + 1) % len(self.selected_aircraft.weapons)
            elif event.key == pygame.K_x and self.selected_aircraft:
                self.selected_aircraft.deploy_flares(self)
            elif event.key == pygame.K_q and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_enemy(self.aircraft)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.DOGFIGHT, 30.0, target=tgt))
            elif event.key == pygame.K_e and self.selected_aircraft:
                tgt, _ = self.selected_aircraft.find_nearest_ground_target(self.ground_targets)
                if tgt:
                    self.selected_aircraft.clear_maneuvers()
                    self.selected_aircraft.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 25.0, target=tgt, parameters={'altitude': 300}))
            elif event.key == pygame.K_c and self.selected_aircraft:
                self.selected_aircraft.clear_maneuvers()

    def update(self, dt):
        if self.paused: return
        for t in self.ground_targets: t.update(dt, self.aircraft, self)
        for a in self.aircraft: a.ai_update(dt, self, is_selected=(a == self.selected_aircraft))
        for p in self.projectiles[:]: 
            p.update(dt, self)
            if not p.active: self.projectiles.remove(p)
        for e in self.effects[:]:
            e.update(dt)
            if e.lifetime <= 0: self.effects.remove(e)

    def draw(self):
        self.screen.fill((20, 40, 80))
        for i in range(0, self.width, 100): pygame.draw.line(self.screen, (40, 60, 100), (i, 0), (i, self.height))
        for i in range(0, self.height, 100): pygame.draw.line(self.screen, (40, 60, 100), (0, i), (self.width, i))
        
        # Draw Entities
        for t in self.ground_targets:
            if not t.alive: continue
            x, y = int(t.pos[0]), int(t.pos[1])
            pygame.draw.rect(self.screen, (200, 100, 100), (x-8, y-8, 16, 16))
            pygame.draw.line(self.screen, (255,150,150), (x, y), (x + math.cos(t.turret_angle)*12, y + math.sin(t.turret_angle)*12), 3)
            
        for a in self.aircraft:
            if not a.alive: continue
            x, y, z = self.project_3d_to_2d(a.pos)
            size = max(8, int(12 - z / 100))
            fwd = np.array([math.cos(a.heading), math.sin(a.heading)]) * size
            left = np.array([-math.sin(a.heading), math.cos(a.heading)]) * size * 0.6
            pts = [np.array([x, y])+fwd, np.array([x, y])-fwd*0.3+left, np.array([x, y])-fwd, np.array([x, y])-fwd*0.3-left]
            pygame.draw.circle(self.screen, (50, 50, 50), (x, y), 3)
            pygame.draw.line(self.screen, tuple(max(50, c-100) for c in a.color), (x, y), (x, y - int(z/10)), 2)
            pygame.draw.polygon(self.screen, a.color, pts)
            
            # Draw CCIP
            if a == self.selected_aircraft and a.ccip is not None:
                cx, cy = int(a.ccip[0]), int(a.ccip[1])
                pygame.draw.circle(self.screen, (255, 0, 0), (cx, cy), 8, 1)
                pygame.draw.line(self.screen, (255, 0, 0), (cx-10, cy), (cx+10, cy))
                pygame.draw.line(self.screen, (255, 0, 0), (cx, cy-10), (cx, cy+10))
                
        for p in self.projectiles:
            if not p.active: continue
            x, y, _ = self.project_3d_to_2d(p.pos)
            pygame.draw.circle(self.screen, (255, 255, 0), (x, y), 3)
            
        for e in self.effects: e.draw(self.screen, self.project_3d_to_2d)
        
        # UI Setup
        ui_surface = pygame.Surface((300, 280))
        ui_surface.set_alpha(180)
        self.screen.blit(ui_surface, (10, 10))
        self.screen.blit(self.font.render("Hybrid Simulation Controls", True, (255,255,255)), (20, 20))
        
        controls = ["1-2: Select Aircraft", "Arrows: Manual Fly", "Space: Manual Fire", "X: Deploy Flares", "W: Switch Weapon", "Q: AI Dogfight", "E: AI Bomb", "C: Clear AI"]
        for idx, txt in enumerate(controls):
            self.screen.blit(self.small_font.render(txt, True, (200, 200, 200)), (20, 60 + idx*20))

        if self.selected_aircraft and self.selected_aircraft.alive:
            wpn = self.selected_aircraft.weapons[self.selected_aircraft.current_weapon]
            stat_text = f"Alt: {self.selected_aircraft.pos[2]:.0f} | Flares: {self.selected_aircraft.flares} | Wpn: {wpn.weapon_type.value} ({wpn.ammo_count})"
            self.screen.blit(self.font.render(stat_text, True, (255,255,0)), (self.width - 400, 20))
            if self.selected_aircraft.current_maneuver:
                self.screen.blit(self.font.render(f"AI: {self.selected_aircraft.current_maneuver.maneuver_type.value}", True, (255,100,100)), (self.width - 400, 50))

        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                else: self.handle_input(event)
            self.update(dt)
            self.draw()
        pygame.quit()

if __name__ == "__main__":
    DogfightSimulation().run()