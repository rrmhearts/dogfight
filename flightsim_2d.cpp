#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

// ==========================================
// UTILITY STRUCTURES
// ==========================================
struct Vector2D {
    int x;
    int y;
    
    // Calculate Euclidean distance between two points
    double distanceTo(const Vector2D& other) const {
        return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
    }
};

// ==========================================
// ENTITIES
// ==========================================

// Base class for anything that exists on the map
class Entity {
public:
    Vector2D position;
    char symbol;
    bool isActive;

    Entity(int x, int y, char sym) : position{x, y}, symbol(sym), isActive(true) {}
    virtual ~Entity() = default;
};

// Target class representing enemies/waypoints
class Target : public Entity {
public:
    string name;

    Target(string n, int x, int y) : Entity(x, y, 'T'), name(n) {}
};

// ==========================================
// AVIONICS & SYSTEMS
// ==========================================

class Radar {
private:
    double scanRadius;

public:
    Radar(double radius = 10.0) : scanRadius(radius) {}

    // Returns a list of pointers to targets that are within scan radius
    vector<Target*> scan(const Vector2D& planePos, vector<Target>& allTargets) {
        vector<Target*> detected;
        for (auto& target : allTargets) {
            if (target.isActive && planePos.distanceTo(target.position) <= scanRadius) {
                detected.push_back(&target);
            }
        }
        return detected;
    }
    
    double getRadius() const { return scanRadius; }
};

class WeaponSystem {
private:
    int missileCount;

public:
    WeaponSystem(int missiles = 4) : missileCount(missiles) {}

    bool fireWeapon(Target* target) {
        if (missileCount <= 0) {
            cout << ">>> Click... Out of missiles!\n";
            return false;
        }
        if (target == nullptr) {
            cout << ">>> No target designated to fire upon!\n";
            return false;
        }

        missileCount--;
        cout << ">>> Fox 2! Missile fired at " << target->name << "!\n";
        target->isActive = false; // Destroy the target
        cout << ">>> Target destroyed. Missiles remaining: " << missileCount << "\n";
        return true;
    }
    
    int getAmmo() const { return missileCount; }
};

class LaserDesignator {
private:
    Target* lockedTarget;

public:
    LaserDesignator() : lockedTarget(nullptr) {}

    // Cycle through available radar contacts to lock on
    void cycleTarget(const vector<Target*>& radarContacts) {
        if (radarContacts.empty()) {
            cout << ">>> No targets in range to lock onto.\n";
            lockedTarget = nullptr;
            return;
        }

        // Find current target in the list to cycle to the next one
        auto it = find(radarContacts.begin(), radarContacts.end(), lockedTarget);
        if (it != radarContacts.end() && next(it) != radarContacts.end()) {
            lockedTarget = *next(it);
        } else {
            // Wrap around to the first target
            lockedTarget = radarContacts.front();
        }
        
        cout << ">>> Laser Designator locked onto: " << lockedTarget->name << " at (" 
             << lockedTarget->position.x << ", " << lockedTarget->position.y << ")\n";
    }

    Target* getLockedTarget() const { return lockedTarget; }
    
    void clearLock() { lockedTarget = nullptr; }
};

// ==========================================
// PLAYER / AIRCRAFT
// ==========================================
class Plane : public Entity {
public:
    Radar radar;
    LaserDesignator designator;
    WeaponSystem weapons;

    Plane(int x, int y) : Entity(x, y, '^') {} // '^' represents the plane facing "North"

    void move(int dx, int dy, int mapWidth, int mapHeight) {
        // Update position with bounds checking
        position.x = max(0, min(mapWidth - 1, position.x + dx));
        position.y = max(0, min(mapHeight - 1, position.y + dy));
        
        // Update symbol based on direction
        if (dx > 0) symbol = '>';
        else if (dx < 0) symbol = '<';
        else if (dy > 0) symbol = 'v';
        else if (dy < 0) symbol = '^';
        
        // If we move, we might break our lock if the target exits radar range.
        // For simplicity, we clear the lock on movement and require re-designation.
        designator.clearLock(); 
    }
};

// ==========================================
// SIMULATOR ENGINE
// ==========================================
class Simulator {
private:
    int width, height;
    Plane player;
    vector<Target> worldTargets;
    bool isRunning;

public:
    Simulator() : width(20), height(15), player(10, 7), isRunning(true) {
        // Populate the world with some dummy targets
        worldTargets.push_back(Target("SAM Site Alpha", 3, 3));
        worldTargets.push_back(Target("Enemy Convoy", 16, 12));
        worldTargets.push_back(Target("Bunker", 5, 12));
    }

    void drawMap() {
        cout << "\n========================================\n";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (player.position.x == x && player.position.y == y) {
                    cout << player.symbol << " "; // Draw Plane
                } else {
                    bool drawn = false;
                    for (const auto& target : worldTargets) {
                        if (target.isActive && target.position.x == x && target.position.y == y) {
                            // Highlight if locked
                            if (player.designator.getLockedTarget() == &target) {
                                cout << "X "; // X represents locked target
                            } else {
                                cout << target.symbol << " "; // T represents normal target
                            }
                            drawn = true;
                            break;
                        }
                    }
                    if (!drawn) {
                        cout << ". "; // Empty space
                    }
                }
            }
            cout << "\n";
        }
        cout << "========================================\n";
    }

    void drawHUD() {
        cout << "POSITION: (" << player.position.x << ", " << player.position.y << ")\t";
        cout << "AMMO: " << player.weapons.getAmmo() << "\n";
        
        Target* locked = player.designator.getLockedTarget();
        cout << "TARGET LOCK: " << (locked ? locked->name : "NONE") << "\n";
        cout << "----------------------------------------\n";
    }

    void printInstructions() {
        cout << "FLIGHT SIMULATOR CONTROLS:\n";
        cout << " [W/A/S/D] : Move Plane (Up/Left/Down/Right)\n";
        cout << " [R] : Active Radar Ping (Show targets in range)\n";
        cout << " [T] : Cycle Laser Designator Lock (Requires targets in range)\n";
        cout << " [F] : Fire Weapon at Locked Target\n";
        cout << " [Q] : Quit Simulator\n";
        cout << "Type a letter and press ENTER: ";
    }

    void handleInput() {
        char input;
        cin >> input;
        input = toupper(input);

        // Fetch current radar contacts for targeting logic
        vector<Target*> contacts = player.radar.scan(player.position, worldTargets);

        switch (input) {
            case 'W': player.move(0, -1, width, height); break;
            case 'S': player.move(0, 1, width, height); break;
            case 'A': player.move(-1, 0, width, height); break;
            case 'D': player.move(1, 0, width, height); break;
            
            case 'R': 
                cout << ">>> RADAR PING (Radius " << player.radar.getRadius() << ")\n";
                if (contacts.empty()) {
                    cout << ">>> No contacts found.\n";
                } else {
                    for (auto c : contacts) {
                        cout << ">>> Contact: " << c->name << " | Distance: " 
                             << player.position.distanceTo(c->position) << "\n";
                    }
                }
                break;
                
            case 'T':
                player.designator.cycleTarget(contacts);
                break;
                
            case 'F':
                player.weapons.fireWeapon(player.designator.getLockedTarget());
                if(player.designator.getLockedTarget() && !player.designator.getLockedTarget()->isActive) {
                    player.designator.clearLock(); // clear lock if destroyed
                }
                break;

            case 'Q':
                isRunning = false;
                break;

            default:
                cout << ">>> Unknown command.\n";
                break;
        }
    }

    void run() {
        cout << "Starting C++ Flight Simulator Architecture Demo...\n";
        while (isRunning) {
            drawMap();
            drawHUD();
            printInstructions();
            handleInput();
        }
        cout << "Simulation ended.\n";
    }
};

// ==========================================
// MAIN ENTRY POINT
// ==========================================
int main() {
    Simulator sim;
    sim.run();
    return 0;
}