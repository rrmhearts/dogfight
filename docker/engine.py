import pygame
import socket
import json
import subprocess
import math
import uuid
import time
import os

# ==========================================
# CONSTANTS & SETUP
# ==========================================
WIDTH, HEIGHT = 1024, 768
FPS = 60
BG_COLOR = (10, 15, 20)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Podman Distributed Flight Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 14)

# Governing Socket Server (Receives container telemetry)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5000))
sock.setblocking(False)

# Assuming host network or docker0 bridge. 
# Use your machine's actual IP if running Podman in a VM (like on Mac/Win).
HOST_IP = "127.0.0.1" 

# State Management
# entities maps ID -> {data, address, health}
entities = {}
players_config = {
    "P1": {
        "color": (0, 255, 0), 
        "radar_active": 0, 
        "target": None, 
        "keys": {
            "fwd": pygame.K_w, 
            "rev": pygame.K_s, 
            "left": pygame.K_a, 
            "right": pygame.K_d, 
            "fire": pygame.K_SPACE, 
            "radar": pygame.K_r
        }
    },
    "P2": {
        "color": (0, 200, 255), 
        "radar_active": 0, 
        "target": None, 
        "keys": {
            "fwd": pygame.K_UP, 
            "rev": pygame.K_DOWN, 
            "left": pygame.K_LEFT, 
            "right": pygame.K_RIGHT, 
            "fire": pygame.K_RETURN, 
            "radar": pygame.K_p
        }
    }
}

# ==========================================
# CONTAINER ORCHESTRATION
# ==========================================
def spawn_container(entity_id, role, start_x, start_y, start_az=0):
    """Executes a Podman command to spawn a new entity container."""
    cmd = [
        "podman", "run", "-d", "--rm", "--network=host",
        f"-e", f"ENTITY_ID={entity_id}",
        f"-e", f"ROLE={role}",
        f"-e", f"ENGINE_IP={HOST_IP}",
        f"-e", f"START_X={start_x}",
        f"-e", f"START_Y={start_y}",
        f"-e", f"START_AZIMUTH={start_az}",
        "flight_sim_entity"
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Register in engine with default health
    health = 100 if role == 'player' else 1
    entities[entity_id] = {"health": health, "address": None, "state": None}

def destroy_container(entity_id):
    """Executes a Podman command to stop/destroy a container."""
    subprocess.Popen(["podman", "stop", "-t", "0", entity_id], stdout=subprocess.DEVNULL)
    if entity_id in entities:
        del entities[entity_id]
        
    # Clear targets if destroyed
    for p in players_config.values():
        if p["target"] == entity_id:
            p["target"] = None

# Spawn initial players
spawn_container("P1", "player", 200, HEIGHT//2, 0)
spawn_container("P2", "player", 800, HEIGHT//2, 180)

# ==========================================
# MAIN LOOP
# ==========================================
running = True
while running:
    # 1. HANDLE EVENTS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Laser Designation: Click near an entity to lock on for P1
            mx, my = pygame.mouse.get_pos()
            closest_id = None
            min_dist = 50
            for eid, edata in entities.items():
                if eid == "P1" or not edata["state"]: continue
                ex, ey = edata["state"]["x"], edata["state"]["y"]
                dist = math.hypot(mx - ex, my - ey)
                if dist < min_dist:
                    closest_id = eid
                    min_dist = dist
            players_config["P1"]["target"] = closest_id

        elif event.type == pygame.KEYDOWN:
            # Handle Single-Press Actions (Radar, Firing)
            for pid, config in players_config.items():
                if event.key == config["keys"]["radar"]:
                    config["radar_active"] = time.time() + 2.0 # Radar ping lasts 2 seconds
                
                if event.key == config["keys"]["fire"]:
                    if pid in entities and entities[pid]["state"] and config["target"]:
                        # Fire missile! Spawn a new missile container
                        m_id = f"M_{uuid.uuid4().hex[:6]}"
                        px = entities[pid]["state"]["x"]
                        py = entities[pid]["state"]["y"]
                        p_az = entities[pid]["state"]["azimuth"]
                        spawn_container(m_id, "missile", px, py, p_az)
                        
                        # Store who fired it and what its target is in Engine
                        entities[m_id] = {"health": 1, "address": None, "state": None, "target": config["target"], "owner": pid}

    # 2. HANDLE NETWORK TELEMETRY (RECEIVE)
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            state = json.loads(data.decode('utf-8'))
            eid = state["id"]
            if eid in entities:
                entities[eid]["state"] = state
                entities[eid]["address"] = addr # Save container's ephemeral port for 2-way comms
        except (BlockingIOError, json.JSONDecodeError):
            break

    # 3. GOVERNING LOGIC & SEND INSTRUCTIONS (SEND)
    keys = pygame.key.get_pressed()
    
    # Process Player Continuous Inputs
    for pid, config in players_config.items():
        if pid in entities and entities[pid]["address"]:
            accel = 1 if keys[config["keys"]["fwd"]] else (-1 if keys[config["keys"]["rev"]] else 0)
            turn = 1 if keys[config["keys"]["right"]] else (-1 if keys[config["keys"]["left"]] else 0)
            
            cmd = {"accel": accel, "turn": turn}
            sock.sendto(json.dumps(cmd).encode('utf-8'), entities[pid]["address"])

    # Process Missile Updates & Collision Detection
    for eid, edata in list(entities.items()):
        if not edata["state"]: continue
        
        if edata["state"]["role"] == "missile":
            target_id = edata.get("target")
            if target_id and target_id in entities and entities[target_id]["state"]:
                # 3a. Send Target Coordinates to Missile Container
                tx = entities[target_id]["state"]["x"]
                ty = entities[target_id]["state"]["y"]
                if edata["address"]:
                    cmd = {"target_x": tx, "target_y": ty}
                    sock.sendto(json.dumps(cmd).encode('utf-8'), edata["address"])
                
                # 3b. Collision Detection
                mx, my = edata["state"]["x"], edata["state"]["y"]
                dist = math.hypot(tx - mx, ty - my)
                if dist < 15: # Impact!
                    print(f"IMPACT! Missile {eid} hit {target_id}!")
                    destroy_container(eid)
                    entities[target_id]["health"] -= 35
                    if entities[target_id]["health"] <= 0:
                        print(f"{target_id} DESTROYED!")
                        destroy_container(target_id)
            else:
                # Target lost/destroyed
                destroy_container(eid)

    # 4. RENDER UI
    screen.fill(BG_COLOR)
    current_time = time.time()
    
    for eid, edata in entities.items():
        if not edata["state"]: continue
        
        x, y = edata["state"]["x"], edata["state"]["y"]
        az = edata["state"]["azimuth"]
        role = edata["state"]["role"]
        
        # Visibility Logic:
        # P1 is always visible. Other entities are visible if they belong to P1, 
        # or if P1's radar is active and they are within range.
        is_visible = True 
        if eid != "P1":
            dist_to_p1 = math.hypot(x - entities["P1"]["state"]["x"], y - entities["P1"]["state"]["y"]) if "P1" in entities and entities["P1"]["state"] else float('inf')
            is_visible = (players_config["P1"]["radar_active"] > current_time and dist_to_p1 < 400) or role == "missile"

        if is_visible:
            # Draw Entity Symbol
            color = players_config.get(eid, {}).get("color", (255, 100, 100) if role == 'player' else (255, 255, 0))
            
            # Simple Triangle for Heading
            rad = math.radians(az)
            tip = (x + math.cos(rad) * 15, y + math.sin(rad) * 15)
            left = (x + math.cos(rad - 2.5) * 10, y + math.sin(rad - 2.5) * 10)
            right = (x + math.cos(rad + 2.5) * 10, y + math.sin(rad + 2.5) * 10)
            pygame.draw.polygon(screen, color, [tip, left, right], 2)
            
            # Draw Data tag
            lbl = font.render(f"{eid} [H:{edata['health']}]", True, (200, 200, 200))
            screen.blit(lbl, (x + 15, y - 10))

    # Draw Radar Ping & Targeting for P1
    if "P1" in entities and entities["P1"]["state"]:
        px, py = entities["P1"]["state"]["x"], entities["P1"]["state"]["y"]
        
        # Radar Ring
        if players_config["P1"]["radar_active"] > current_time:
            pygame.draw.circle(screen, (0, 255, 0), (int(px), int(py)), 400, 1)
            
        # Laser Designator Line
        t_id = players_config["P1"]["target"]
        if t_id and t_id in entities and entities[t_id]["state"]:
            tx, ty = entities[t_id]["state"]["x"], entities[t_id]["state"]["y"]
            pygame.draw.line(screen, (255, 0, 0), (px, py), (tx, ty), 1)
            pygame.draw.circle(screen, (255, 0, 0), (int(tx), int(ty)), 20, 1)

    pygame.display.flip()
    clock.tick(FPS)

# Cleanup on exit
for eid in list(entities.keys()):
    destroy_container(eid)
pygame.quit()