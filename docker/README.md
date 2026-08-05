# Podman Distributed Flight Simulator

This project is a microservice-based flight simulator. The `engine.py` script acts as the server, UI renderer, and orchestrator. It uses subprocess to trigger podman commands that spin up completely independent containers for players and missiles.

### Prerequisites

1. Python 3.11+  
2. pygame installed on your host (pip install pygame)  
3. **Podman** installed and running on your system.

### Step 1: Build the Entity Container

The Engine requires a pre-built container image to spawn. In the same directory as the files above, run:

`podman build -f Containerfile -t flight_sim_entity .`

### Step 2: Ensure Networking setup

* **On Linux:** The script uses `--network=host`, meaning the containers and your host share the same localhost, and UDP packets will route perfectly.  
* **On macOS/Windows:** Podman runs inside a hidden VM. `--network=host` binds to the *VM's* localhost, not your Mac/Windows host.  
  * **Fix:** Open `engine.py` and change `HOST_IP = "127.0.0.1"` to your actual local IPv4 address (e.g., `192.168.1.100`), or host.containers.internal.

### Step 3: Run the Engine

`python engine.py`

*Note: You may see terminal popups as Podman spins containers up and down in the background.*

### Controls

* **Player 1 (Green):**  
  * W / S: Accelerate / Decelerate  
  * A / D: Yaw Left / Yaw Right  
  * R: Radar Ping (Reveals targets within a 400px radius for 2 seconds)  
  * Mouse Left Click: Laser Designate (Click near an enemy to lock on)  
  * SPACE: Fire Missile at Designated Target  
* **Player 2 (Blue):**  
  * Up / Down / Left / Right arrows: Movement  
  * Enter: Fire (if target logic is added to player 2\)