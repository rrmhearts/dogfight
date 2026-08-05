## What the Green and Blue Are

The green and blue represent a **dynamic artificial horizon** (sky and ground) rendered in a top-down/orthographic pseudo-3D view:

* **`SKY_BLUE` (Top):** Represents the open sky above your aircraft.
* **`GROUND_GREEN` (Bottom):** Represents the earth below.

### How it works dynamically:

Instead of a static background, the script calculates a moving horizon line based on your camera's altitude (`self.camera_pos[2]`):

```python
sky_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT / 2 - (self.camera_pos[2] / WORLD_SCALE))

```

* As you **climb to higher altitudes**, the horizon line shifts downward, revealing more sky (`SKY_BLUE`) and less ground.
* As you **dive toward the earth**, the horizon line rises, making the grassy ground (`GROUND_GREEN`) fill more of your screen.

---

## How to Control It

You control your A-10 Warthog using the keyboard. Here are your flight and combat controls defined in `player_update`:

| Key | Action | What It Does |
| --- | --- | --- |
| **LEFT / RIGHT Arrows** | **Turn / Yaw** | Rotates your aircraft's heading left or right (`15°/sec` turn rate for the A-10). |
| **UP / DOWN Arrows** | **Climb / Dive** | Changes your altitude (`+50 m/s` upward or `-50 m/s` downward). |
| **T** | **Target / Lock-On** | Cycles your target lock to the closest enemy (turns a yellow HUD box red after 1.5s of tracking). |
| **SPACE** | **Fire Weapon** | Drops an **Mk82 bomb** at your calculated CCIP ground target or fires your primary weapon. |
| **F** | **Deploy Flares** | Releases countermeasures (`5 flares max`, 5s cooldown) with an **80% chance** to break incoming enemy missile locks. |

---

## Key HUD & Targeting Features to Watch For

1. **CCIP Bombing Reticle (Red Crosshair):** Because your A-10 is armed with **Mk82 bombs**, the game runs a 500-step gravity physics simulation (`calculate_ccip`) every frame to predict where your bomb will land. A red circle with crosshairs marks the exact impact point on the ground.
2. **Lock-On Progress Box:** Pressing **T** puts a **Yellow** square over an enemy. Keep facing them for 1.5 seconds and it turns **Red**, indicating a full target lock.
3. **Incoming Missile Warning:** If the AI F-16 fires a Sidewinder missile at you, **`INCOMING MISSILE`** flashes in red at the bottom of the screen—press **F** immediately to deploy flares.