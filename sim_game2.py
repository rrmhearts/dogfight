import pygame
import math
import random
import numpy as np

# --- PYGAME INITIALIZATION ---
pygame.init()
pygame.font.init()

# --- CONSTANTS AND CONFIGURATION ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 800
SKY_BLUE = (135, 206, 235)
GROUND_GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# World-to-Screen mapping
# This scale means 1 pixel on screen = 10 units in the game world
WORLD_SCALE = 10.0 

# Physics
GRAVITY = 9.8

# --- FONT & UI SETUP ---
FONT_S = pygame.font.SysFont('Consolas', 16)
FONT_M = pygame.font.SysFont('Consolas', 20)

# --- HELPER FUNCTIONS ---
def world_to_screen(pos, camera_pos):
    """Converts 3D world coordinates to 2D screen coordinates."""
    screen_x = (pos[0] - camera_pos[0]) / WORLD_SCALE + SCREEN_WIDTH / 2
    screen_y = (pos[1] - camera_pos[1]) / WORLD_SCALE + SCREEN_HEIGHT / 2
    return int(screen_x), int(screen_y)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def heading_to_vector(heading_deg):
    rad = math.radians(heading_deg)
    return np.array([math.sin(rad), math.cos(rad)])

# --- BASE ENTITY CLASS ---
class BaseEntity:
    def __init__(self, pos, team):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.team = team
        self.is_destroyed = False

    def update(self, dt, world):
        raise NotImplementedError

    def draw(self, surface, camera_pos):
        raise NotImplementedError

# --- PROJECTILE CLASS ---
class Projectile(BaseEntity):
    def __init__(self, pos, team, target, weapon_config):
        super().__init__(pos, team)
        self.target = target
        self.config = weapon_config
        self.type = self.config['type']
        self.blast_radius = self.config['blast_radius']
        self.damage = self.config['damage']

        self.thrust_time = self.config.get('thrust_time', 0)
        self.lifetime = self.config['range'] / self.config['velocity'] if self.config['velocity'] > 0 else 5.0

    def update(self, dt, world):
        if self.is_destroyed: return

        self.lifetime -= dt
        
        # --- Physics-based trajectory ---
        if self.type == 'bomb':
            self.velocity[2] -= GRAVITY * dt
        elif self.type == 'missile':
            # --- Missile Fuel/Thrust and Seeker Logic ---
            if self.thrust_time > 0:
                self.thrust_time -= dt
                if self.target and not self.target.is_destroyed:
                    # Proportional navigation guidance
                    target_dir = normalize(self.target.pos - self.pos)
                    # Simple turn logic
                    self.velocity = normalize(self.velocity + target_dir * self.config.get('turn_rate', 2.0) * dt)
                # Maintain speed
                self.velocity = normalize(self.velocity) * self.config['velocity']
            else: # Gliding phase
                self.velocity[2] -= GRAVITY * dt * 0.5 

        self.pos += self.velocity * dt

        # --- Ground Impact Detection ---
        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.explode(world)
            return

        if self.lifetime <= 0:
            self.is_destroyed = True
            return

        # Proximity fuse check
        for entity in world.get_all_entities():
            if not entity.is_destroyed and entity.team != self.team:
                if np.linalg.norm(self.pos - entity.pos) < 20: # Collision radius
                    self.explode(world)
                    return

    def explode(self, world):
        self.is_destroyed = True
        world.add_effect('explosion', self.pos, self.blast_radius)
        for entity in world.get_all_entities():
            dist_3d = np.linalg.norm(self.pos - entity.pos)
            if not entity.is_destroyed and dist_3d < self.blast_radius:
                # Damage falls off with distance
                damage_dealt = self.damage * (1 - (dist_3d / self.blast_radius))
                entity.take_damage(damage_dealt)

    def draw(self, surface, camera_pos):
        if self.is_destroyed: return
        screen_pos = world_to_screen(self.pos, camera_pos)
        pygame.draw.circle(surface, BLACK, screen_pos, 2)
        
# --- AIRCRAFT CLASS ---
class Aircraft(BaseEntity):
    def __init__(self, pos, team, aircraft_type, is_player=False):
        super().__init__(pos, team)
        self.type_config = AIRCRAFT_TYPES[aircraft_type]
        self.speed = self.type_config['speed']
        self.heading = 0.0
        self.health = 100
        self.is_player = is_player

        # AI & State
        self.ai_maneuver = {'type': 'PATROL'}
        self.ai_target = None
        
        # Weapons & Systems
        self.weapon_cooldown = 0
        self.lock_on_target = None
        self.lock_on_timer = 0
        self.is_targeted_by_missile = False
        self.flares = 5
        self.flare_cooldown = 0
        self.ccip = None

    def take_damage(self, amount):
        if self.is_destroyed: return
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.is_destroyed = True

    def update(self, dt, world):
        if self.is_destroyed: return
            
        if self.is_player:
            self.player_update(dt, world)
        else:
            self.ai_decision_update(dt, world)
        
        self.physics_update(dt)

        if self.weapon_cooldown > 0: self.weapon_cooldown -= dt
        if self.flare_cooldown > 0: self.flare_cooldown -= dt
        
        # Calculate CCIP for bombs if equipped
        if self.type_config['weapons'][0]['type'] == 'bomb':
            self.ccip = self.calculate_ccip()

    def player_update(self, dt, world):
        """Handle player inputs."""
        keys = pygame.key.get_pressed()
        turn_rate = self.type_config['turn_rate']
        if keys[pygame.K_LEFT]: self.heading -= turn_rate * dt
        if keys[pygame.K_RIGHT]: self.heading += turn_rate * dt
        if keys[pygame.K_UP]: self.velocity[2] += 50 * dt
        if keys[pygame.K_DOWN]: self.velocity[2] -= 50 * dt

        self.heading %= 360

        # Find a target by pressing 't'
        if keys[pygame.K_t]:
            self.ai_target = self.find_closest_enemy(world, lockable_only=True)
            self.lock_on_target = self.ai_target
            self.lock_on_timer = 0

        # Fire weapon
        if keys[pygame.K_SPACE]:
            self.fire_on_target(world)

        # Deploy flares
        if keys[pygame.K_f]:
            self.deploy_flares(world)

    def ai_decision_update(self, dt, world):
        """Handles AI logic."""
        # Simple state machine
        if self.ai_maneuver['type'] == 'PATROL':
            if not self.ai_target or self.ai_target.is_destroyed:
                self.ai_target = self.find_closest_enemy(world)
            if self.ai_target:
                weapon = self.type_config['weapons'][0]
                if weapon['type'] == 'bomb' and isinstance(self.ai_target, GroundTarget):
                    self.ai_maneuver = {'type': 'BOMB_TARGET', 'stage': 'APPROACH'}
                else:
                    self.ai_maneuver = {'type': 'ATTACK_TARGET'}
        
        elif self.ai_maneuver['type'] in ['ATTACK_TARGET', 'BOMB_TARGET']:
            if not self.ai_target or self.ai_target.is_destroyed:
                self.ai_maneuver = {'type': 'PATROL'}
                self.ai_target = None
            else:
                self.fire_on_target(world)

    def physics_update(self, dt):
        """Handles all movement."""
        if not self.is_player:
            self.execute_ai_maneuver(dt)

        # Update velocity vector based on heading and speed
        heading_vec_2d = heading_to_vector(self.heading)
        self.velocity[0] = heading_vec_2d[0] * self.speed
        self.velocity[1] = heading_vec_2d[1] * self.speed
        
        self.pos += self.velocity * dt

        # Prevent flying into the ground
        min_alt = 50
        if self.pos[2] < min_alt:
            self.pos[2] = min_alt
            self.velocity[2] = max(0, self.velocity[2])

    def execute_ai_maneuver(self, dt):
        """AI-driven movement based on current maneuver."""
        maneuver = self.ai_maneuver
        target = self.ai_target
        if not target: return

        target_pos = target.pos
        direction_to_target = target_pos - self.pos
        
        if maneuver['type'] == 'ATTACK_TARGET':
            target_heading = math.degrees(math.atan2(direction_to_target[0], direction_to_target[1]))
            self.turn_towards(target_heading, dt)
            
        elif maneuver['type'] == 'BOMB_TARGET':
            stage = maneuver.get('stage', 'APPROACH')
            target_heading = math.degrees(math.atan2(direction_to_target[0], direction_to_target[1]))
            self.turn_towards(target_heading, dt)
            dist_to_target_2d = np.linalg.norm(self.pos[:2] - target.pos[:2])
            
            if stage == 'APPROACH':
                if dist_to_target_2d < 1500: maneuver['stage'] = 'DIVE'
            elif stage == 'DIVE':
                self.velocity[2] = -80 # Dive
                if self.pos[2] < 500: maneuver['stage'] = 'RELEASE'
            elif stage == 'RELEASE':
                self.fire_on_target(self.find_closest_enemy(world)) # world needs to be passed here
                maneuver['stage'] = 'EGRESS'
            elif stage == 'EGRESS':
                self.velocity[2] = 100 # Pull up
                if self.pos[2] > 800: self.ai_maneuver = {'type': 'PATROL'}


    def turn_towards(self, target_heading, dt):
        turn_rate = self.type_config['turn_rate']
        angle_diff = (target_heading - self.heading + 180) % 360 - 180
        max_turn = turn_rate * dt
        turn = np.clip(angle_diff, -max_turn, max_turn)
        self.heading = (self.heading + turn) % 360

    def find_closest_enemy(self, world, lockable_only=False):
        closest_target = None
        min_dist = float('inf')
        for target in world.get_all_entities():
            if target.team != self.team and not target.is_destroyed:
                if lockable_only and isinstance(target, GroundTarget):
                    continue
                dist = np.linalg.norm(self.pos - target.pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_target = target
        return closest_target

    def fire_on_target(self, world):
        if not self.ai_target or self.weapon_cooldown > 0:
            return

        weapon = self.type_config['weapons'][0]
        can_fire = self.calculate_firing_solution(self.ai_target, weapon)
        
        if can_fire:
            initial_velocity = np.copy(self.velocity)
            if weapon['type'] != 'bomb':
                direction = normalize(self.ai_target.pos - self.pos)
                initial_velocity = direction * weapon['velocity']
            
            proj = Projectile(self.pos.copy(), self.team, self.ai_target, weapon)
            proj.velocity = initial_velocity
            world.projectiles.append(proj)
            self.weapon_cooldown = weapon['fire_rate']
            self.lock_on_target = None
            self.lock_on_timer = 0

    def calculate_firing_solution(self, target, weapon):
        dist_to_target = np.linalg.norm(self.pos - target.pos)
        if dist_to_target > weapon['range']:
            return False

        if weapon['type'] == 'missile':
            if self.lock_on_target != target:
                self.lock_on_target = target
                self.lock_on_timer = 0
                return False
            else:
                self.lock_on_timer += 1/60.0 # dt
                return self.lock_on_timer >= 1.5 # Lock-on time

        if weapon['type'] == 'bomb':
             # Fire if CCIP is close to the target
            return self.ccip is not None and np.linalg.norm(self.ccip - target.pos[:2]) < 50
        
        return True

    def calculate_ccip(self):
        """Continuously Computed Impact Point for bombs."""
        sim_pos = self.pos.copy()
        sim_vel = self.velocity.copy()
        dt = 0.05 # simulation step
        for _ in range(500): # max simulation steps
            sim_vel[2] -= GRAVITY * dt
            sim_pos += sim_vel * dt
            if sim_pos[2] <= 0:
                return sim_pos[:2]
        return None

    def deploy_flares(self, world):
        if self.flares > 0 and self.flare_cooldown <= 0:
            self.flares -= 1
            self.flare_cooldown = 5
            world.add_effect('flares', self.pos, 5)
            # Break missile locks
            for p in world.projectiles:
                if p.type == 'missile' and p.target == self:
                    if random.random() < 0.8: # 80% chance to break lock
                        p.target = None

    def draw(self, surface, camera_pos):
        if self.is_destroyed: return
        screen_pos = world_to_screen(self.pos, camera_pos)
        
        # Draw aircraft body (triangle)
        size = 15
        angle = math.radians(self.heading)
        p1 = (screen_pos[0] + size * math.sin(angle), screen_pos[1] + size * math.cos(angle))
        p2 = (screen_pos[0] + size/2 * math.sin(angle + math.pi*0.8), screen_pos[1] + size/2 * math.cos(angle + math.pi*0.8))
        p3 = (screen_pos[0] + size/2 * math.sin(angle - math.pi*0.8), screen_pos[1] + size/2 * math.cos(angle - math.pi*0.8))
        color = (0,0,255) if self.team == 1 else (200,0,0)
        pygame.draw.polygon(surface, color, [p1, p2, p3])
        if self.is_player: # Add a white outline for player
             pygame.draw.polygon(surface, WHITE, [p1, p2, p3], 1)

        # Draw lock-on indicator
        if self.is_player and self.lock_on_target:
            target_screen_pos = world_to_screen(self.lock_on_target.pos, camera_pos)
            lock_progress = self.lock_on_timer / 1.5
            if lock_progress < 1.0:
                color = YELLOW
                pygame.draw.rect(surface, color, (*target_screen_pos, 20, 20), 1)
            else:
                color = RED
                pygame.draw.rect(surface, color, (*target_screen_pos, 20, 20), 2)
        
        # Draw CCIP
        if self.is_player and self.ccip is not None:
            ccip_screen_pos = world_to_screen(np.append(self.ccip, 0), camera_pos)
            pygame.draw.circle(surface, RED, ccip_screen_pos, 8, 1)
            pygame.draw.line(surface, RED, (ccip_screen_pos[0]-12, ccip_screen_pos[1]), (ccip_screen_pos[0]+12, ccip_screen_pos[1]), 1)
            pygame.draw.line(surface, RED, (ccip_screen_pos[0], ccip_screen_pos[1]-12), (ccip_screen_pos[0], ccip_screen_pos[1]+12), 1)

# --- GROUND TARGET CLASS ---
class GroundTarget(BaseEntity):
    def __init__(self, pos, team):
        super().__init__(pos, team)
        self.health = 200
        self.pos[2] = 0 # Ensure it's on the ground

    def take_damage(self, amount):
        if self.is_destroyed: return
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.is_destroyed = True
    
    def update(self, dt, world):
        pass # It's stationary

    def draw(self, surface, camera_pos):
        if self.is_destroyed: return
        screen_pos = world_to_screen(self.pos, camera_pos)
        color = (0,0,150) if self.team == 1 else (150,0,0)
        pygame.draw.rect(surface, color, (screen_pos[0]-8, screen_pos[1]-8, 16, 16))

# --- VISUAL EFFECTS ---
class Effect:
    def __init__(self, type, pos, magnitude):
        self.type = type
        self.pos = pos
        self.magnitude = magnitude
        self.lifetime = 1.0 if type == 'explosion' else 2.0

    def update(self, dt):
        self.lifetime -= dt

    def draw(self, surface, camera_pos):
        screen_pos = world_to_screen(self.pos, camera_pos)
        if self.type == 'explosion':
            alpha = max(0, 255 * self.lifetime)
            radius = int((self.magnitude / WORLD_SCALE) * (1.0 - self.lifetime))
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (255, 165, 0, alpha), (radius, radius), radius)
            surface.blit(temp_surf, (screen_pos[0] - radius, screen_pos[1] - radius))
        elif self.type == 'flares':
            for _ in range(5):
                offset = (random.uniform(-20, 20), random.uniform(-20, 20))
                pygame.draw.circle(surface, YELLOW, (screen_pos[0] + offset[0], screen_pos[1] + offset[1]), 3)

# --- SIMULATION/GAME CLASS ---
class DogfightSimulation:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.aircraft = []
        self.ground_targets = []
        self.projectiles = []
        self.effects = []
        
        # Setup scenario
        player = Aircraft(pos=[-2000, -5000, 1000], team=1, aircraft_type='A-10', is_player=True)
        self.aircraft.append(player)
        self.player_ref = player
        self.camera_pos = player.pos.copy()
        
        self.aircraft.append(Aircraft(pos=[1000, 5000, 1200], team=2, aircraft_type='F-16'))
        self.ground_targets.append(GroundTarget(pos=[0, 0, 0], team=2))

    def get_all_entities(self):
        return self.aircraft + self.ground_targets

    def add_effect(self, type, pos, magnitude):
        self.effects.append(Effect(type, pos, magnitude))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self, dt):
        for a in self.aircraft: a.update(dt, self)
        for g in self.ground_targets: g.update(dt, self)
        for p in self.projectiles: p.update(dt, self)
        for e in self.effects: e.update(dt)
        
        # Check for missile targeting on player
        self.player_ref.is_targeted_by_missile = False
        for p in self.projectiles:
            if p.type == 'missile' and p.target == self.player_ref:
                self.player_ref.is_targeted_by_missile = True
                break

        # Cleanup destroyed entities
        self.aircraft = [a for a in self.aircraft if not a.is_destroyed]
        self.ground_targets = [g for g in self.ground_targets if not g.is_destroyed]
        self.projectiles = [p for p in self.projectiles if not p.is_destroyed]
        self.effects = [e for e in self.effects if e.lifetime > 0]
        
        # Update camera to follow player smoothly
        self.camera_pos += (self.player_ref.pos - self.camera_pos) * 0.1

    def draw(self):
        # Draw ground and sky
        self.screen.fill(GROUND_GREEN)
        sky_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT / 2 - (self.camera_pos[2] / WORLD_SCALE))
        self.screen.fill(SKY_BLUE, sky_rect)
        
        # Draw entities
        all_drawables = sorted(self.aircraft + self.ground_targets + self.projectiles, key=lambda e: e.pos[2])
        for entity in all_drawables:
            entity.draw(self.screen, self.camera_pos)
        
        for effect in self.effects:
            effect.draw(self.screen, self.camera_pos)

        # Draw HUD
        self.draw_hud()

        pygame.display.flip()

    def draw_hud(self):
        # Player Status
        alt_text = FONT_M.render(f"ALT: {self.player_ref.pos[2]:.0f} m", True, WHITE)
        spd_text = FONT_M.render(f"SPD: {np.linalg.norm(self.player_ref.velocity):.0f} kph", True, WHITE)
        self.screen.blit(alt_text, (10, 10))
        self.screen.blit(spd_text, (10, 35))
        
        # Weapon/System Status
        flares_text = FONT_M.render(f"Flares: {self.player_ref.flares}", True, YELLOW)
        self.screen.blit(flares_text, (10, SCREEN_HEIGHT - 30))
        
        # Missile Warning
        if self.player_ref.is_targeted_by_missile:
            warning_text = FONT_M.render("INCOMING MISSILE", True, RED)
            self.screen.blit(warning_text, (SCREEN_WIDTH/2 - warning_text.get_width()/2, SCREEN_HEIGHT - 50))

# --- CONFIGURATION DATA ---
WEAPON_CONFIG = {
    'sidewinder': {'type': 'missile', 'range': 8000, 'velocity': 850, 'blast_radius': 75, 'damage': 100, 'fire_rate': 5, 'turn_rate': 3.0, 'thrust_time': 4},
    'mk82': {'type': 'bomb', 'range': 0, 'velocity': 0, 'blast_radius': 150, 'damage': 250, 'fire_rate': 1}
}

AIRCRAFT_TYPES = {
    'F-16': {'speed': 250, 'turn_rate': 21, 'weapons': [WEAPON_CONFIG['sidewinder']]},
    'A-10': {'speed': 190, 'turn_rate': 15, 'weapons': [WEAPON_CONFIG['mk82']]}
}

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Dogfight Simulator")
    
    sim = DogfightSimulation(screen)
    sim.run()
    
    pygame.quit()
