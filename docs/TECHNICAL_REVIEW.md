# Technical Review Guide

This guide is written for a reviewer evaluating the project as an interactive AI systems artifact.

## What To Look For

- The script separates rendering, environment semantics, perception, decision logic, and steering.
- Agents do not use omniscient target selection; they perceive events through configured sensor constraints.
- Movement is force-based and frame-rate independent.
- Procedural assets are not merely decorative; they participate in navigation as obstacles and landmarks.
- Logs expose the reasoning loop instead of only showing final movement.

## Behavior Contract

Each simulation tick follows a stable order:

1. Clamp frame delta time.
2. Update the event director.
3. Snapshot peer positions for order-independent separation.
4. Query perception for each agent.
5. Advance the FSM.
6. Blend steering forces.
7. Integrate velocity and position.
8. Update Panda3D scene graph nodes.

This makes the demo deterministic under a fixed seed and predictable enough to extend.

## Quality Gates

Recommended checks:

```bash
python -m py_compile ai_agent_simulation.py
python ai_agent_simulation.py --seed 7 --agents 4
```

Visual review:

- Agents should patrol without clustering permanently.
- FOV/debug overlays should move with the agent bodies.
- Beacon events should trigger visible state changes.
- Agents should slow near targets rather than oscillating violently.
- Console logs should show transition reasons and vector telemetry.

## Extension Points

- Replace `Brain` with a behavior tree while preserving the same sensor and steering contracts.
- Replace the synthetic `Director` with external telemetry.
- Export the event stream to JSONL for analysis.
- Add authored models while keeping obstacle metadata in the environment layer.

## Portfolio Signal

The project demonstrates senior-level judgment by making the underlying system inspectable. The visible demo matters, but the more important signal is the traceability from event generation to perception, state transition, steering vector, and final rendered transform.
