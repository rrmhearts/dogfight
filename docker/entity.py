import socket
import json
import os
import time
import math
import select

# Environment Variables passed by Podman
ENTITY_ID = os.getenv("ENTITY_ID", "unknown")
ROLE = os.getenv("ROLE", "player") # 'player' or 'missile'
ENGINE_IP = os.getenv("ENGINE_IP", "127.0.0.1")
ENGINE_PORT = int(os.getenv("ENGINE_PORT", 5000))

# Initial Kinematic State
x = float(os.getenv("START_X", 400))
y = float(os.getenv("START_Y", 300))
azimuth = float(os.getenv("START_AZIMUTH", 0)) # Degrees
elevation = float(os.getenv("START_ELEVATION", 10000))
speed = float(os.getenv("START_SPEED", 0)) if ROLE == 'player' else 15.0

# Missile specific state
target_x = None
target_y = None

# Networking Setup (UDP)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 0)) # Bind to any available ephemeral port inside container
sock.setblocking(False)

print(f"[{ENTITY_ID}] Started as {ROLE}. Engine IP: {ENGINE_IP}:{ENGINE_PORT}")

def send_state():
    """Broadcasts current kinematic state to the governing engine."""
    state = {
        "id": ENTITY_ID,
        "role": ROLE,
        "x": x,
        "y": y,
        "azimuth": azimuth,
        "elevation": elevation,
        "speed": speed
    }
    try:
        sock.sendto(json.dumps(state).encode('utf-8'), (ENGINE_IP, ENGINE_PORT))
    except Exception as e:
        pass

def process_messages():
    """Reads incoming inputs/target updates from the Engine."""
    global speed, azimuth, target_x, target_y
    try:
        ready = select.select([sock], [], [], 0.01)
        if ready[0]:
            data, _ = sock.recvfrom(1024)
            msg = json.loads(data.decode('utf-8'))
            
            if ROLE == 'player':
                # Update player physics based on Engine input commands
                accel = msg.get('accel', 0)
                turn = msg.get('turn', 0)
                speed = max(0.0, min(20.0, speed + accel * 0.5))
                azimuth = (azimuth + turn * 3.0) % 360
                
            elif ROLE == 'missile':
                # Missiles receive live target coordinates to update trajectory
                target_x = msg.get('target_x')
                target_y = msg.get('target_y')
                
    except (BlockingIOError, json.JSONDecodeError):
        pass

# Main Physics/Self-Propulsion Loop
last_time = time.time()
while True:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    process_messages()

    # Self-Propelled Missile Guidance Logic (Proportional Navigation)
    if ROLE == 'missile' and target_x is not None and target_y is not None:
        desired_az = math.degrees(math.atan2(target_y - y, target_x - x))
        
        # Calculate shortest turn direction
        diff = (desired_az - azimuth + 180) % 360 - 180
        turn_rate = 5.0 # Max missile turn rate per frame
        
        if diff > turn_rate:
            azimuth += turn_rate
        elif diff < -turn_rate:
            azimuth -= turn_rate
        else:
            azimuth = desired_az

    # Update Position (Dead Reckoning)
    rad_az = math.radians(azimuth)
    x += speed * math.cos(rad_az)
    y += speed * math.sin(rad_az)

    # Broadcast state to Engine 30 times a second
    send_state()
    time.sleep(1/30.0)