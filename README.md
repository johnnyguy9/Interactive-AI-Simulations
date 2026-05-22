# Advanced Interactive Architecture & Autonomous Agent Simulations

A self-contained Panda3D simulation of multiple autonomous agents operating inside a procedurally generated digital-twin operations floor. Each agent runs a four-state Finite State Machine layered on a Reynolds-style steering stack with field-of-view perception, dynamic obstacle avoidance, and peer separation.

The repository ships **zero external assets**: every box, beacon, grid line, and obstacle is built at runtime from `GeomVertexData` primitives. `pip install panda3d` is the entire dependency surface.

---

## Overview

This is a portfolio-grade reference for behaviour-driven autonomous agents in real-time 3D — the kind of substrate that underlies game AI, robotics test harnesses, and operations-research sandboxes. It demonstrates several things at once: clean separation of decision logic from the rendering engine, composable steering instead of hard-coded navigation, perception-as-data instead of probability rolls, and deterministic replay via a single CLI flag.

The code is written to be read. Class boundaries match conceptual boundaries; every non-obvious calculation is commented with the math, the assumption, or the design choice behind it.

---

## System Architecture

```
SimulationApplication (Panda3D ShowBase host)
    ├── DigitalTwinEnvironment   procedural arena, named obstacles, beacons
    ├── Director                 activates / expires beacons over time
    └── Agent[]                  autonomous units, one per simulated entity
            ├── Body             kinematic state (position, velocity)
            ├── Sensor           FOV-cone + range perception
            ├── Brain            FSM controller + blackboard memory
            ├── Steering         Reynolds behaviour primitives
            └── DebugDraw        live FOV cone + velocity overlays
```

Composition over inheritance throughout. `Agent` owns its collaborators; none of them reach back into the agent. `Brain` is the only component that holds FSM state, and it expresses each decision as a single steering force returned to the agent each tick — a contract small enough to mock against in a headless unit test.

The Panda3D `taskMgr` drives one application-level tick. That tick advances the `Director`, snapshots peer bodies (so steering blends are order-independent), then runs each agent's `step()`. There is no shared mutable state between agents inside a tick.

---

## Core Mechanics

### Finite State Machine

Four discrete states with explicit entry hooks: `IDLE`, `PATROL`, `SEEK`, `INVESTIGATE`.

- `IDLE` — bounded wait, then promote to `PATROL`. Perception can preempt.
- `PATROL` — `arrive` at a sampled waypoint while avoiding obstacles and peers; falls back to `IDLE` on arrival or timer expiry.
- `SEEK` — pursue an acknowledged beacon. On contact, the beacon is marked acknowledged and removed from the active set; agent returns to `IDLE`. If the beacon expires mid-pursuit (or seek timeout fires), the agent transitions to `INVESTIGATE` at the last known location.
- `INVESTIGATE` — walk to the cached last-known location; re-acquire as `SEEK` if a beacon re-enters the sight cone, otherwise time out to `IDLE`.

Every transition flows through `Brain._enter_state(...)`, which logs the previous state, the new state, and a human-readable reason. Adding a state (e.g. `FLEE`) is a four-line change.

### Perception

A beacon is perceived when distance < `sight_range` AND the angle between the agent's heading and the agent→beacon ray is less than half the configured FOV. The cone test uses a dot product against a precomputed `cos(half_fov)`, so each perception query is `O(n)` in active beacons with no trigonometric calls in the hot path.

The `Sensor` knows only what's in the world; the `Brain` knows only what the `Sensor` reports. Swapping the sensor for a mock (or for a richer sensor-fusion implementation) requires no changes elsewhere.

### Steering

Five composable behaviours, each returning a force vector clamped to `max_force`:

- `seek(target)` — head straight at the target at max speed.
- `arrive(target)` — `seek` with a linear deceleration ramp inside `slowdown_radius`, so the agent settles cleanly instead of overshooting.
- `wander()` — Reynolds wander: project a circle ahead of the agent, pick a jittered point on it, and `seek` it. Produces smooth, plausibly aimless motion.
- `avoid_obstacles(obstacles)` — cast a forward lookahead ray and steer laterally away from the closest intersecting obstacle. Cheap, deterministic, single allocation.
- `separate(peers)` — repel from any peer inside the separation radius, with an inverse-distance falloff so near-collisions dominate the blend.

States choose the blend. `PATROL`, for example, mixes `arrive(waypoint) * 1.0 + avoid_obstacles * 1.6 + separate * 0.8`. The weights are explicit, in one place, easy to tune.

### Digital-Twin Environment

The arena models a small operations floor with five named assets — `control-cabinet`, `pump-station`, `storage-rack`, `inspection-node`, `data-relay` — each generated procedurally and registered as a circular obstacle.

A `Director` periodically activates one of five named beacons (`thermal-anomaly`, `operator-ping`, `data-relay-check`, `inventory-audit`, `pump-vibration`). Active beacons glow; agents perceive them through their sight cone and acknowledge them on arrival. Beacons time out if no agent reaches them in time.

This gives the simulation a real operational rhythm: events fire, agents notice, the closest available agent responds, and the system settles back to patrol.

### Integration & Safety

Standard semi-implicit Euler: `v' = clamp(v + F·dt, max_speed)`, then `p' = p + v'·dt`. `dt` is sourced from `globalClock` and clamped to 100 ms so a paused window doesn't teleport agents on resume. Arena walls reflect velocity with a 65% damping factor as a final safety net behind the steering layer. A top-level exception handler logs unhandled errors with a full traceback rather than dropping into an opaque engine crash.

---

## Installation & Dependencies

Requires Python 3.10+ and a working Panda3D install.

```bash
pip install -r requirements.txt
python ai_agent_simulation.py
```

### Runtime Flags

```bash
python ai_agent_simulation.py --agents 5         # five-agent swarm
python ai_agent_simulation.py --arena 12         # larger arena
python ai_agent_simulation.py --seed 42          # deterministic replay
python ai_agent_simulation.py --no-debug         # hide FOV / velocity overlays
```

All tunables (max speed, FOV degrees, sight range, separation radius, beacon lifetime, etc.) live in the `SimulationConfig` dataclass at the top of the script. Config is frozen at construction so misconfigurations fail loudly rather than drifting mid-run.

### Reading the Console

Three structured log streams:

- `FSM | <previous> -> <next> | <reason>` — every state transition for every agent.
- `vec[<agent>|<state>] pos=… dir=… d=… step=…` — per-tick navigation vector, throttled.
- `snap[<agent>] state=… pos=… v=… target=…` — periodic agent snapshot.

The Director emits `beacon ACTIVE / CLEAR` events at the application log level.

---

## Future Roadmap

The current FSM is a stepping stone. The architecture is shaped to accommodate the following without rewrites:

- **Behaviour Trees / GOAP** — promote `Brain` into a pluggable interface; ship a BT implementation alongside the FSM for compound behaviours (selectors, sequences, decorators).
- **Agentic Orchestration** — coordinate agents through a shared blackboard and a `Coordinator` role, enabling team behaviours: flanking, formation movement, beacon division-of-labour.
- **Navigation Mesh** — replace lookahead-ray avoidance with a proper nav-mesh for concave environments and corridors.
- **Sensor Fusion** — extend `Sensor` to model auditory and proximity channels, each with its own decay curve, into a unified perception event stream.
- **Headless Replay** — flag-driven detached run that emits a JSONL trace of every decision; designed for offline analytics and regression testing.
- **Live Telemetry Ingestion** — replace the synthetic Director with a real event bus feed (Kafka, MQTT, Webhooks) for true digital-twin synchronisation.

Each item is bounded, reviewable, and additive. The current code is the foundation; the roadmap is what's next.

---

## Repository Layout

```
.
├── ai_agent_simulation.py      Panda3D entry point — agents, FSM, steering, world
├── README.md                   This file
├── requirements.txt            Pinned-minimum dependency surface
├── LICENSE                     MIT
└── .gitignore                  Python / IDE / build artifacts
```

Self-contained. No build step, no package layout, no external assets. `git clone && pip install -r requirements.txt && python ai_agent_simulation.py` runs the demo.
