# Advanced Interactive Architecture & Autonomous Agent Simulations

Production-minded Panda3D simulation of autonomous agents operating inside a procedural digital-twin environment. The project demonstrates multi-agent AI, finite state machines, field-of-view perception, steering behaviors, dynamic event response, deterministic replay, and engine-level procedural rendering.

The repository is intentionally self-contained. There are no downloaded meshes, no hidden asset packs, and no fragile build chain: the arena, grid, obstacles, beacons, debug overlays, and agent bodies are generated at runtime from Panda3D geometry primitives.

## Executive Overview

This project is built as a portfolio-grade reference for real-time AI systems. It shows how to separate engine integration from decision logic, how to make autonomous behavior inspectable, and how to expose the information a technical reviewer needs to judge quality.

Core signals:

- Multi-agent autonomous behavior.
- Explicit finite state machine design.
- Field-of-view perception using vector math.
- Reynolds-style steering with arrival, wander, separation, and obstacle avoidance.
- Procedural digital-twin scene construction.
- Deterministic replay through CLI seeding.
- Structured console telemetry for state, vector, and event review.

## Architecture

```text
SimulationApplication
|-- DigitalTwinEnvironment
|   |-- Procedural floor, grid, obstacles, and beacons
|   |-- Navigable bounds and collision metadata
|   `-- Event targets used by the Director
|-- Director
|   |-- Activates and expires beacon events
|   `-- Coordinates simulation-level telemetry
`-- Agent[]
    |-- Body: kinematic position and velocity
    |-- Sensor: range and field-of-view perception
    |-- Brain: FSM state and blackboard memory
    |-- Steering: composable movement forces
    `-- DebugDraw: live FOV and vector overlays
```

The most important design decision is that agent intelligence does not live in the Panda3D application object. The engine supplies the render loop and scene graph; the AI components consume world snapshots and return steering intent. That keeps behavior testable and makes the script useful as a reference for larger game, simulation, robotics, or digital twin systems.

## Core Mechanics

### Finite State Machine

The agent brain uses four states:

- `IDLE`: brief bounded wait before re-entering patrol; perception can preempt it.
- `PATROL`: move toward a sampled waypoint while avoiding obstacles and peers.
- `SEEK`: pursue an active perceived beacon and acknowledge it on arrival.
- `INVESTIGATE`: continue toward the last known beacon location if the event expires mid-pursuit.

Every transition is routed through one entry point and logged with a human-readable reason. This makes behavior reviewable instead of mysterious.

### Perception

The sensor evaluates active beacons with two tests:

1. Distance must be inside the configured sight range.
2. The dot product between forward direction and target direction must be inside the FOV cone.

This keeps perception cheap, deterministic, and mathematically clear.

### Steering

The movement layer blends multiple force generators:

- `seek(target)`: direct pursuit.
- `arrive(target)`: pursuit with slowdown near the target.
- `wander()`: smooth non-deterministic patrol variation.
- `avoid_obstacles(obstacles)`: forward lookahead obstacle correction.
- `separate(peers)`: inverse-distance peer repulsion.

Each state chooses its own weights, so tuning behavior does not require rewriting the motion model.

### Runtime Safety

The simulation clamps frame delta time before integration. If the window stalls, a debugger pauses, or the OS schedules slowly, agents do not teleport across the arena. Boundary reflection is a final safety layer behind steering.

## Installation

```bash
pip install -r requirements.txt
python ai_agent_simulation.py
```

## Runtime Flags

```bash
python ai_agent_simulation.py --agents 5
python ai_agent_simulation.py --arena 12
python ai_agent_simulation.py --seed 42
python ai_agent_simulation.py --no-debug
```

## Console Telemetry

The console output is designed for inspection:

- `FSM`: every state transition and reason.
- `vec`: navigation vector, direction, distance, and step data.
- `snap`: periodic agent position, velocity, and target state.
- `Director`: beacon activation and clearing events.

That makes the demo useful not only as a visual artifact, but as a behavior-debugging reference.

## Repository Layout

```text
.
|-- ai_agent_simulation.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- .gitignore
`-- docs/
    `-- TECHNICAL_REVIEW.md
```

## Review Notes

See [docs/TECHNICAL_REVIEW.md](docs/TECHNICAL_REVIEW.md) for the deeper engineering review guide: behavior contracts, quality gates, extension seams, and suggested reviewer checks.

## Roadmap

- Behavior-tree or GOAP brain implementation alongside the FSM.
- Shared blackboard coordination for team behaviors.
- Headless replay trace export for regression analysis.
- Navmesh integration for concave environments.
- Sensor fusion channels for audio, proximity, and external event feeds.
- JSONL telemetry export for offline evaluation.
