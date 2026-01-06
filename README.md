# 3D Dogfight Simulation

A real-time 3D aerial combat simulation with advanced AI maneuvers, ground targets, and tactical mission planning. Built with Python and Pygame, featuring an overhead tactical view of fully 3D flight physics.

![Dogfight Simulation](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Features

### **Aircraft Systems**

The simulation features comprehensive aircraft systems built around full 3D flight physics that provide realistic movement and flight dynamics. Multiple aircraft types are available including specialized fighters optimized for air-to-air combat, dedicated attack aircraft for ground operations, and versatile fighter-bombers capable of mixed missions. Each aircraft is equipped with authentic weapon systems ranging from rapid-fire machine guns and heat-seeking missiles to heavy cannons and area-effect bombs. The combat system includes detailed health and damage modeling with visual health bars and real-time combat feedback, while team-based warfare creates dynamic Blue vs Red force engagements.

### **Advanced Maneuver System**

A sophisticated maneuver system provides over ten distinct tactical maneuvers ranging from basic patrol routes to complex multi-phase flanking attacks. The maneuver queue system allows players to chain multiple orders together, enabling coordinated operations across multiple aircraft simultaneously. Smart AI drives context-aware behavior with intelligent target selection and tactical decision-making that adapts to changing battlefield conditions. Precision formation flying capabilities include accurate wingman positioning and following behaviors with configurable distances and offset angles for realistic military formations.

### **Ground Warfare**

Comprehensive ground warfare features multiple target types including armored tanks, strategic buildings, and defensive anti-aircraft gun emplacements. Active defense systems create dynamic threats as AA guns automatically detect and engage enemy aircraft within range. Coordinated bombing runs allow for strategic air-to-ground attack missions with proper altitude management and target approach procedures. The fully destructible environment means all ground targets can be eliminated through sustained attack, creating evolving tactical situations as defensive positions are neutralized.

### **Interactive Control**

Real-time command and control systems allow players to issue immediate orders to individual aircraft during active combat operations. Advanced mission planning capabilities enable pre-programming of complex multi-aircraft operations with synchronized timing and coordinated objectives. The tactical overhead perspective provides clear battlefield visualization with 3D altitude indicators that maintain spatial awareness of aircraft positioning. Complete simulation control includes pause and resume functionality for detailed tactical analysis and strategic planning during complex engagements.

## Installation

### Prerequisites
```bash
pip install pygame numpy
```

### Quick Start
```bash
git clone https://github.com/rrmhearts/dogfight-simulation.git
cd dogfight-simulation
python dogfight.py
```

## Controls

### Basic Controls
| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume simulation |
| `R` | Reset simulation |
| `M` | Run sample mission |
| `1-4` | Select aircraft |

### Aircraft Commands
| Key | Command | Description |
|-----|---------|-------------|
| `Q` | Dogfight | Engage nearest enemy aircraft |
| `E` | Bomb Target | Attack nearest ground target |
| `W/S` | Change weapon | switch between available weapon systems |
| `F` | Follow | Follow nearest friendly aircraft |
| `O` | Orbit | Circle around target |
| `G` | Flank | Wide flanking attack |
| `I` | Intercept | Predictive intercept course |
| `T` | Retreat | Withdraw from combat |
| `C` | Clear Orders | Cancel all maneuvers |

## Aircraft Types

### Fighter
- **Role**: Air superiority
- **Weapons**: Machine gun, Air-to-air missiles
- **Characteristics**: High speed, excellent maneuverability
- **Best For**: Dogfighting, escort missions

### Attack Aircraft
- **Role**: Ground attack
- **Weapons**: Cannon, Bombs
- **Characteristics**: Heavy armor, ground-attack capability
- **Best For**: Tank busting, building destruction

### Fighter-Bomber
- **Role**: Multi-role
- **Weapons**: Cannon, Bombs, Missiles
- **Characteristics**: Balanced performance
- **Best For**: Mixed missions, versatile operations

## Weapon Systems

### Air-to-Air
- **Machine Gun**: High rate of fire, moderate damage
- **Missiles**: Heat-seeking, high damage, limited ammo

### Air-to-Ground
- **Cannon**: High damage, can target ground units
- **Bombs**: Massive damage, area effect, gravity-affected

### Anti-Aircraft
- **AA Guns**: Ground-based automatic defense systems

## Maneuver Types

### Combat Maneuvers
- **Dogfight**: Aggressive air-to-air engagement with lead calculation
- **Attack Run**: High-speed direct assault on targets
- **Flank**: Wide circling maneuver for positional advantage
- **Intercept**: Predictive course to cut off enemy aircraft

### Support Maneuvers
- **Follow**: Formation flying with configurable distance and offset
- **Orbit**: Circular patrol around specified point or target
- **Escort**: Protective formation flying

### Tactical Maneuvers
- **Bomb Target**: Coordinated bombing run with altitude management
- **Climb/Dive**: Altitude adjustment for tactical advantage
- **Retreat**: Evasive withdrawal from dangerous situations
- **Patrol**: Waypoint-based area coverage

## Programming Guide

### Adding Custom Aircraft
```python
# Define weapons
custom_weapon = WeaponConfig(
    weapon_type=WeaponType.CANNON,
    damage=75,
    range=400,
    fire_rate=2,
    velocity=800,
    ammo_count=60,
    can_target_ground=True
)

# Define aircraft configuration
custom_config = AircraftConfig(
    max_speed=150,
    acceleration=1.2,
    turn_rate=1.8,
    climb_rate=40,
    max_altitude=600,
    health=120,
    weapons=[custom_weapon]
)

# Add to simulation
aircraft = sim.add_aircraft(x, y, z, "blue", custom_config, (0, 255, 0))
```

### Creating Custom Missions
```python
def create_strike_mission():
    sim = DogfightSimulation()
    
    # Setup aircraft
    striker = sim.aircraft[0]
    escort = sim.aircraft[1]
    
    # Plan coordinated attack
    striker.add_maneuver(Maneuver(ManeuverType.CLIMB, 5.0, 
                                 parameters={'altitude': 400}))
    striker.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 20.0, 
                                 target=ground_target))
    
    escort.add_maneuver(Maneuver(ManeuverType.FOLLOW, 15.0, 
                                target=striker))
    escort.add_maneuver(Maneuver(ManeuverType.DOGFIGHT, 30.0))
    
    return sim
```

### Adding Ground Targets
```python
# Create different target types
tank = GroundTarget(x, y, "tank", health=100, team="red")
building = GroundTarget(x, y, "building", health=200, team="red")
aa_gun = GroundTarget(x, y, "aa_gun", health=80, team="red")

sim.ground_targets.extend([tank, building, aa_gun])
```

## Architecture

### Core Classes
- **`Aircraft`**: Individual aircraft with AI, physics, and weapon systems
- **`Projectile`**: Bullets, missiles, and bombs with ballistics
- **`GroundTarget`**: Static and active ground-based targets
- **`Maneuver`**: Individual tactical commands with parameters
- **`WeaponConfig`**: Weapon specifications and capabilities
- **`AircraftConfig`**: Aircraft performance characteristics

### Key Systems
- **Physics Engine**: 3D movement with realistic flight dynamics
- **AI System**: Behavior trees for tactical decision making
- **Maneuver System**: Queue-based command execution
- **Weapons System**: Projectile simulation with hit detection
- **Rendering Engine**: 3D-to-2D projection with visual effects

## Advanced Features

### Mission Scripting
Create complex multi-phase operations:
```python
# Phase 1: SEAD (Suppression of Enemy Air Defenses)
wild_weasel.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 15.0, 
                                 target=aa_gun))

# Phase 2: Strike Package
for bomber in strike_package:
    bomber.add_maneuver(Maneuver(ManeuverType.BOMB_TARGET, 20.0, 
                                target=primary_target))

# Phase 3: CAP (Combat Air Patrol)
for fighter in escort:
    fighter.add_maneuver(Maneuver(ManeuverType.ORBIT, 30.0, 
                                 target=target_area))
```

### Formation Flying
```python
# Diamond formation
lead.add_maneuver(Maneuver(ManeuverType.PATROL, 60.0))

wing_left.add_maneuver(Maneuver(ManeuverType.FOLLOW, 60.0, 
                               target=lead,
                               parameters={'distance': 100, 
                                         'offset_angle': math.pi/4}))

wing_right.add_maneuver(Maneuver(ManeuverType.FOLLOW, 60.0, 
                                target=lead,
                                parameters={'distance': 100, 
                                          'offset_angle': -math.pi/4}))
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This simulation is for educational and entertainment purposes. It demonstrates principles of flight dynamics, artificial intelligence, and real-time simulation programming.