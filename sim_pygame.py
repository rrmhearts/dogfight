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

@dataclass
class AircraftConfig:
    max_speed: float
    acceleration: float
    turn_rate: float  # radians per second
    climb_rate: float
    max_altitude: float
    health: float
    weapons: List[WeaponConfig]

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

    def update(self, dt, aircraft_list):
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
                    # Simple spherical interpolation
                    self.velocity = self.velocity + (desired_velocity - self.velocity) * (turn_amount / angle_diff)
                    self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * self.config.velocity

        # Update position
        self.pos += self.velocity * dt

        # Check for hits
        for aircraft in aircraft_list:
            if aircraft.team != self.team and aircraft.alive:
                distance = np.linalg.norm(aircraft.pos - self.pos)
                hit_distance = 15.0 if self.config.blast_radius > 0 else 8.0
                
                if distance < hit_distance:
                    aircraft.take_damage(self.config.damage)
                    self.active = False
                    break

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
        self.target = None
        
        # Weapon systems
        self.weapons = config.weapons.copy()
        self.current_weapon = 0
        self.weapon_cooldowns = [0.0] * len(self.weapons)
        
        # AI state
        self.ai_state = "patrol"
        self.last_shot_time = 0
        self.maneuver_timer = 0
        self.maneuver_type = None

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

    def can_fire_at_target(self, target):
        if not target or not target.alive:
            return False
            
        # Check if target is in range and roughly in front
        to_target = target.pos - self.pos
        distance = np.linalg.norm(to_target)
        
        if distance > self.weapons[self.current_weapon].range:
            return False
            
        # Check angle to target
        forward = np.array([math.cos(self.heading), math.sin(self.heading), 0])
        to_target_normalized = to_target / distance
        angle = math.acos(np.clip(np.dot(forward, to_target_normalized), -1, 1))
        
        return angle < math.pi / 4  # 45 degree cone

    def fire_weapon(self):
        if self.current_weapon >= len(self.weapons):
            return None
            
        weapon = self.weapons[self.current_weapon]
        
        # Check ammo
        if weapon.ammo_count <= 0:
            return None
            
        # Check cooldown
        if self.weapon_cooldowns[self.current_weapon] > 0:
            return None
            
        # Create projectile
        weapon.ammo_count -= 1
        self.weapon_cooldowns[self.current_weapon] = 1.0 / weapon.fire_rate
        
        # Calculate firing position and velocity
        forward = np.array([math.cos(self.heading), math.sin(self.heading), 0])
        fire_pos = self.pos + forward * 20
        
        projectile_velocity = self.velocity + forward * weapon.velocity
        
        target = self.target if weapon.tracking else None
        
        return Projectile(fire_pos[0], fire_pos[1], fire_pos[2],
                         projectile_velocity[0], projectile_velocity[1], projectile_velocity[2],
                         weapon, target, self.team)

    def ai_update(self, dt, aircraft_list):
        if not self.alive:
            return
            
        # Update weapon cooldowns
        for i in range(len(self.weapon_cooldowns)):
            self.weapon_cooldowns[i] = max(0, self.weapon_cooldowns[i] - dt)
            
        # Find target
        enemy, enemy_distance = self.find_nearest_enemy(aircraft_list)
        self.target = enemy
        
        if enemy and enemy_distance < 800:  # Engagement range
            self.ai_state = "engage"
        else:
            self.ai_state = "patrol"
            
        # AI behavior
        if self.ai_state == "engage" and enemy:
            # Point towards enemy
            to_enemy = enemy.pos - self.pos
            desired_heading = math.atan2(to_enemy[1], to_enemy[0])
            
            # Smooth turning
            heading_diff = desired_heading - self.heading
            while heading_diff > math.pi:
                heading_diff -= 2 * math.pi
            while heading_diff < -math.pi:
                heading_diff += 2 * math.pi
                
            max_turn = self.config.turn_rate * dt
            turn_amount = max(-max_turn, min(max_turn, heading_diff))
            self.heading += turn_amount
            
            # Evasive maneuvers
            self.maneuver_timer -= dt
            if self.maneuver_timer <= 0 or not self.maneuver_type:
                self.maneuver_timer = random.uniform(2.0, 5.0)
                self.maneuver_type = random.choice(["climb", "dive", "bank_left", "bank_right"])
                
            # Apply maneuvers
            if self.maneuver_type == "climb" and self.pos[2] < self.config.max_altitude:
                self.pitch = min(0.3, self.pitch + dt * 0.5)
            elif self.maneuver_type == "dive" and self.pos[2] > 50:
                self.pitch = max(-0.3, self.pitch - dt * 0.5)
            elif self.maneuver_type == "bank_left":
                self.heading += self.config.turn_rate * dt * 0.5
            elif self.maneuver_type == "bank_right":
                self.heading -= self.config.turn_rate * dt * 0.5
                
        else:  # Patrol
            # Simple patrol pattern
            self.pitch *= 0.95  # Level out
            
        # Maintain speed
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

class DogfightSimulation:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Dogfight Simulation")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.aircraft = []
        self.projectiles = []
        
        # Camera for 3D projection
        self.camera_height = 1000
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.setup_default_scenario()

    def setup_default_scenario(self):
        """Setup a default dogfight scenario"""
        # Define weapon configurations
        machine_gun = WeaponConfig(
            weapon_type=WeaponType.MACHINE_GUN,
            damage=15,
            range=300,
            fire_rate=10,
            velocity=800,
            ammo_count=500
        )
        
        missile = WeaponConfig(
            weapon_type=WeaponType.MISSILE,
            damage=100,
            range=600,
            fire_rate=0.5,
            velocity=900,
            ammo_count=4,
            tracking=True,
            blast_radius=20
        )
        
        cannon = WeaponConfig(
            weapon_type=WeaponType.CANNON,
            damage=50,
            range=250,
            fire_rate=3,
            velocity=1000,
            ammo_count=100
        )
        
        # Fighter aircraft config
        fighter_config = AircraftConfig(
            max_speed=300,
            acceleration=2.0,
            turn_rate=2.0,
            climb_rate=50,
            max_altitude=500,
            health=100,
            weapons=[machine_gun, missile]
        )
        
        # Attack aircraft config
        attack_config = AircraftConfig(
            max_speed=250,
            acceleration=1.5,
            turn_rate=1.5,
            climb_rate=30,
            max_altitude=400,
            health=150,
            weapons=[cannon, machine_gun]
        )
        
        # Create aircraft
        self.aircraft = [
            Aircraft(200, 200, 200, "blue", fighter_config, (100, 150, 255)),
            Aircraft(300, 250, 180, "blue", fighter_config, (100, 150, 255)),
            Aircraft(1000, 600, 220, "red", attack_config, (255, 100, 100)),
            Aircraft(900, 550, 200, "red", fighter_config, (255, 150, 150)),
        ]
        
        # Set initial headings
        for aircraft in self.aircraft:
            if aircraft.team == "blue":
                aircraft.heading = 0  # Face right
            else:
                aircraft.heading = math.pi  # Face left

    def add_aircraft(self, x, y, z, team, config, color):
        """Add a new aircraft to the simulation"""
        aircraft = Aircraft(x, y, z, team, config, color)
        self.aircraft.append(aircraft)
        return aircraft

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
        size = max(8, int(12 - z / 100))  # Smaller when higher
        
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
        ui_surface = pygame.Surface((300, 200))
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
        
        # Active projectiles
        active_projectiles = sum(1 for p in self.projectiles if p.active)
        proj_text = self.font.render(f"Projectiles: {active_projectiles}", True, (255, 255, 255))
        self.screen.blit(proj_text, (20, y_offset))
        y_offset += 25
        
        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Simulation",
            "1-4: Select Aircraft",
            "W/S: Change Weapon"
        ]
        
        for control in controls:
            control_text = self.small_font.render(control, True, (200, 200, 200))
            self.screen.blit(control_text, (20, y_offset))
            y_offset += 18
            
        # Selected aircraft info
        if hasattr(self, 'selected_aircraft') and self.selected_aircraft and self.selected_aircraft.alive:
            aircraft = self.selected_aircraft
            
            info_surface = pygame.Surface((250, 120))
            info_surface.set_alpha(180)
            info_surface.fill((0, 0, 50))
            self.screen.blit(info_surface, (self.width - 260, 10))
            
            info_y = 20
            name_text = self.font.render(f"{aircraft.team.upper()} Aircraft", True, (255, 255, 255))
            self.screen.blit(name_text, (self.width - 250, info_y))
            info_y += 25
            
            health_text = self.small_font.render(f"Health: {aircraft.health:.0f}/{aircraft.config.health}", True, (255, 255, 255))
            self.screen.blit(health_text, (self.width - 250, info_y))
            info_y += 18
            
            alt_text = self.small_font.render(f"Altitude: {aircraft.pos[2]:.0f}m", True, (255, 255, 255))
            self.screen.blit(alt_text, (self.width - 250, info_y))
            info_y += 18
            
            speed = np.linalg.norm(aircraft.velocity)
            speed_text = self.small_font.render(f"Speed: {speed:.0f}", True, (255, 255, 255))
            self.screen.blit(speed_text, (self.width - 250, info_y))
            info_y += 18
            
            weapon = aircraft.weapons[aircraft.current_weapon]
            weapon_text = self.small_font.render(f"Weapon: {weapon.weapon_type.value}", True, (255, 255, 255))
            self.screen.blit(weapon_text, (self.width - 250, info_y))
            info_y += 18
            
            ammo_text = self.small_font.render(f"Ammo: {weapon.ammo_count}", True, (255, 255, 255))
            self.screen.blit(ammo_text, (self.width - 250, info_y))

    def update(self, dt):
        """Update simulation"""
        if self.paused:
            return
            
        # Update aircraft
        for aircraft in self.aircraft:
            if aircraft.alive:
                aircraft.ai_update(dt, self.aircraft)
                
                # AI firing logic
                if aircraft.target and aircraft.can_fire_at_target(aircraft.target):
                    if random.random() < 0.02:  # 2% chance per frame to fire
                        projectile = aircraft.fire_weapon()
                        if projectile:
                            self.projectiles.append(projectile)
                            
        # Update projectiles
        for projectile in self.projectiles[:]:
            projectile.update(dt, self.aircraft)
            if not projectile.active:
                self.projectiles.remove(projectile)

    def handle_input(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
                
            elif event.key == pygame.K_r:
                # Reset simulation
                self.projectiles.clear()
                self.setup_default_scenario()
                
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
            
            # Draw projectiles first (so they appear behind aircraft)
            for projectile in self.projectiles:
                self.draw_projectile(projectile)
                
            # Draw aircraft
            for aircraft in self.aircraft:
                self.draw_aircraft(aircraft)
                
            # Highlight selected aircraft
            if hasattr(self, 'selected_aircraft') and self.selected_aircraft and self.selected_aircraft.alive:
                x, y, z = self.project_3d_to_2d(self.selected_aircraft.pos)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 20, 2)
            
            # Draw UI
            self.draw_ui()
            
            # Pause indicator
            if self.paused:
                pause_text = self.font.render("PAUSED", True, (255, 255, 0))
                self.screen.blit(pause_text, (self.width // 2 - 40, 50))
            
            pygame.display.flip()
        
        pygame.quit()

def create_custom_scenario():
    """Example of how to create a custom scenario"""
    sim = DogfightSimulation()
    
    # Clear default aircraft
    sim.aircraft.clear()
    
    # Create custom weapon configs
    heavy_cannon = WeaponConfig(
        weapon_type=WeaponType.CANNON,
        damage=80,
        range=400,
        fire_rate=2,
        velocity=1200,
        ammo_count=50
    )
    
    homing_missile = WeaponConfig(
        weapon_type=WeaponType.MISSILE,
        damage=150,
        range=800,
        fire_rate=0.3,
        velocity=1000,
        ammo_count=8,
        tracking=True,
        blast_radius=30
    )
    
    # Create heavy fighter config
    heavy_fighter = AircraftConfig(
        max_speed=280,
        acceleration=1.8,
        turn_rate=1.8,
        climb_rate=40,
        max_altitude=600,
        health=200,
        weapons=[heavy_cannon, homing_missile]
    )
    
    # Create interceptor config
    interceptor = AircraftConfig(
        max_speed=350,
        acceleration=2.5,
        turn_rate=2.5,
        climb_rate=60,
        max_altitude=550,
        health=80,
        weapons=[WeaponConfig(WeaponType.MACHINE_GUN, 20, 350, 15, 900, 800)]
    )
    
    # Add aircraft in formation
    sim.add_aircraft(200, 300, 250, "blue", heavy_fighter, (50, 100, 255))
    sim.add_aircraft(150, 250, 230, "blue", interceptor, (100, 150, 255))
    sim.add_aircraft(250, 350, 270, "blue", interceptor, (100, 150, 255))
    
    sim.add_aircraft(1000, 500, 280, "red", heavy_fighter, (255, 50, 50))
    sim.add_aircraft(950, 450, 260, "red", heavy_fighter, (255, 100, 100))
    
    return sim

if __name__ == "__main__":
    # Run default scenario
    simulation = DogfightSimulation()
    
    # Or run custom scenario
    # simulation = create_custom_scenario()
    
    simulation.run()