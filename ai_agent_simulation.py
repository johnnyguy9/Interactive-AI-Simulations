"""
ai_agent_simulation.py
======================

Multi-Agent Autonomous Behaviour Simulation - Panda3D digital twin.

A self-contained 3D simulation of multiple autonomous agents operating
inside a procedurally generated digital-twin operations floor. Each
agent is governed by a four-state Finite State Machine (FSM) layered on
top of a Reynolds-style steering stack with field-of-view perception,
dynamic obstacle avoidance, and peer separation.

The repository deliberately ships **zero external assets**: every box,
beacon, grid line, and obstacle is generated at runtime from
``GeomVertexData`` primitives so the simulation runs from a clean
``pip install panda3d`` with nothing else on disk.

Architecture
------------
SimulationApplication (Panda3D ShowBase host)
    |-- DigitalTwinEnvironment   procedural arena, obstacles, beacons
    |-- Director                 activates beacons over time
    `-- Agent[]                  autonomous units
            |-- Body             kinematic state (position, velocity)
            |-- Sensor           FOV-cone + range perception
            |-- Brain            FSM + blackboard memory
            |-- Steering         Reynolds behaviour primitives
            `-- DebugDraw        live FOV / velocity overlays

Design notes
------------
* Decision logic is fully separated from the engine. ``Brain`` reads
  perception events and returns a steering force; the application layer
  integrates motion and renders. The same ``Brain`` runs unchanged in a
  headless unit test against a mocked sensor feed.
* Behaviour is frame-rate independent. Per-tick updates consume
  ``globalClock.getDt()`` clamped to 100 ms, so debugger pauses and
  window moves cannot teleport an agent.
* The RNG is seedable from the CLI, producing fully deterministic
  replays for regression testing.

Run
---
    python ai_agent_simulation.py
    python ai_agent_simulation.py --agents 5 --obstacles 7
    python ai_agent_simulation.py --seed 42        # deterministic replay
    python ai_agent_simulation.py --no-debug       # hide overlays

Engine:  Panda3D 1.10+
Python:  3.10+
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
    Vec3,
    Vec4,
    loadPrcFileData,
)


# ---------------------------------------------------------------------------
# Engine bootstrap
# ---------------------------------------------------------------------------
loadPrcFileData(
    "",
    "\n".join((
        "window-title Interactive AI Simulation - Autonomous Agents",
        "win-size 1280 720",
        "sync-video true",
        "show-frame-rate-meter true",
    )),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(name)-12s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sim")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable simulation tunables. ``frozen=True`` so any misconfiguration
    fails loudly at construction rather than drifting silently mid-run.
    """

    # World
    arena_half_extent: float = 8.0
    obstacle_clearance: float = 0.55

    # Agents
    agent_count: int = 3
    agent_max_speed: float = 3.6
    agent_max_force: float = 11.0
    agent_arrival_radius: float = 0.45
    agent_slowdown_radius: float = 2.5
    agent_fov_degrees: float = 110.0
    agent_sight_range: float = 6.5
    agent_avoid_lookahead: float = 2.4
    agent_separation_radius: float = 1.4
    agent_wander_circle_distance: float = 2.0
    agent_wander_circle_radius: float = 0.9
    agent_wander_jitter_radians: float = 0.55

    # FSM timers (seconds)
    idle_duration: Tuple[float, float] = (0.7, 2.0)
    patrol_duration: Tuple[float, float] = (4.0, 7.5)
    investigate_duration: float = 3.0
    seek_timeout: float = 9.0

    # Beacons (procedural digital-twin events)
    beacon_activation_interval: Tuple[float, float] = (3.5, 7.0)
    beacon_max_active: int = 2
    beacon_lifetime: float = 12.0

    # Telemetry / rendering
    vector_report_period: float = 0.6
    state_report_period: float = 1.0
    debug_draw: bool = True


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------
class AgentState(Enum):
    IDLE = "IDLE"
    PATROL = "PATROL"
    SEEK = "SEEK"
    INVESTIGATE = "INVESTIGATE"


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned navigable envelope on the XY plane."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, point: Vec3) -> bool:
        return (self.min_x <= point.x <= self.max_x
                and self.min_y <= point.y <= self.max_y)

    def clamp(self, point: Vec3) -> Vec3:
        return Vec3(
            max(self.min_x, min(self.max_x, point.x)),
            max(self.min_y, min(self.max_y, point.y)),
            point.z,
        )

    def random_point(self, rng: random.Random, z: float = 0.38) -> Vec3:
        return Vec3(
            rng.uniform(self.min_x, self.max_x),
            rng.uniform(self.min_y, self.max_y),
            z,
        )


@dataclass(frozen=True)
class Obstacle:
    """Static circular keep-out zone backed by a visible 3D asset."""
    name: str
    center: Vec3
    radius: float

    def contains_2d(self, point: Vec3, clearance: float) -> bool:
        """Z-flat clearance test - XY distance only."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        threshold = self.radius + clearance
        return dx * dx + dy * dy <= threshold * threshold


@dataclass
class SensorBeacon:
    """
    A semantic digital-twin event source. Beacons sit at fixed locations
    (control cabinets, pump stations, etc.) and the ``Director`` cycles
    them between active and inactive states to drive agent behaviour.
    """
    name: str
    location: Vec3
    priority: float
    node: NodePath
    active: bool = False
    activated_at: float = 0.0
    acknowledged_by: Optional[str] = None


@dataclass(frozen=True)
class NavigationVector:
    """Per-tick motion solution captured for telemetry and debugging."""
    origin: Vec3
    target: Vec3
    direction: Vec3
    distance: float
    step_length: float
    proposed_position: Vec3
    corrected_position: Vec3
    boundary_corrected: bool


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _truncate(v: Vec3, max_length: float) -> Vec3:
    """Clamp ``v`` to ``max_length``. Skips the sqrt when it's a no-op."""
    length_sq = v.lengthSquared()
    if length_sq <= max_length * max_length or length_sq == 0.0:
        return v
    return v * (max_length / math.sqrt(length_sq))


def _flatten(v: Vec3) -> Vec3:
    """Project ``v`` onto the XY plane - steering math is 2D."""
    return Vec3(v.x, v.y, 0.0)


# ---------------------------------------------------------------------------
# Procedural geometry
# ---------------------------------------------------------------------------
def create_box(name: str, half_extents: Vec3, color: Vec4) -> NodePath:
    """
    Build a lit, six-face box mesh from scratch.

    Each face owns its own vertices and normals so flat shading reads
    correctly under the directional key light. This keeps the repository
    portable: ``pip install panda3d`` is the entire dependency surface,
    no model files required.
    """
    vertex_format = GeomVertexFormat.getV3n3c4()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UHStatic)
    vertices = GeomVertexWriter(vertex_data, "vertex")
    normals = GeomVertexWriter(vertex_data, "normal")
    colors = GeomVertexWriter(vertex_data, "color")

    hx, hy, hz = half_extents.x, half_extents.y, half_extents.z
    faces = (
        (Vec3(+1, 0, 0), (Vec3(hx, -hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, hy, hz), Vec3(hx, -hy, hz))),
        (Vec3(-1, 0, 0), (Vec3(-hx, hy, -hz), Vec3(-hx, -hy, -hz), Vec3(-hx, -hy, hz), Vec3(-hx, hy, hz))),
        (Vec3(0, +1, 0), (Vec3(-hx, hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz))),
        (Vec3(0, -1, 0), (Vec3(hx, -hy, -hz), Vec3(-hx, -hy, -hz), Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz))),
        (Vec3(0, 0, +1), (Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz))),
        (Vec3(0, 0, -1), (Vec3(-hx, hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, -hy, -hz), Vec3(-hx, -hy, -hz))),
    )

    triangles = GeomTriangles(Geom.UHStatic)
    vertex_index = 0
    for normal, face_vertices in faces:
        for vertex in face_vertices:
            vertices.addData3(vertex)
            normals.addData3(normal)
            colors.addData4(color)
        triangles.addVertices(vertex_index, vertex_index + 1, vertex_index + 2)
        triangles.addVertices(vertex_index, vertex_index + 2, vertex_index + 3)
        vertex_index += 4

    geom = Geom(vertex_data)
    geom.addPrimitive(triangles)
    geom_node = GeomNode(name)
    geom_node.addGeom(geom)
    return NodePath(geom_node)


# ---------------------------------------------------------------------------
# Agent kinematic / memory state
# ---------------------------------------------------------------------------
@dataclass
class Body:
    """Position + velocity for one agent. 2.5D: motion in XY, render in Z."""
    position: Vec3
    velocity: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    @property
    def speed(self) -> float:
        return self.velocity.length()

    @property
    def heading_radians(self) -> float:
        """Heading from velocity; +Y fallback when at rest."""
        if self.velocity.lengthSquared() < 1e-6:
            return math.pi / 2.0
        return math.atan2(self.velocity.y, self.velocity.x)


@dataclass
class Blackboard:
    """Compact per-agent memory shared across FSM ticks."""
    state_timer: float = 0.0
    patrol_waypoint: Optional[Vec3] = None
    pursued_beacon: Optional[str] = None
    last_known_location: Optional[Vec3] = None
    wander_target_angle: float = 0.0
    vector_log_timer: float = 0.0
    snapshot_timer: float = 0.0


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------
class Sensor:
    """
    Field-of-view + range based perception.

    A beacon is perceived when distance < ``sight_range`` AND the angle
    between the agent's heading and the agent->beacon ray is less than
    half the FOV. Inactive beacons are filtered out at the source so the
    dot-product loop only ever sees real events.
    """

    def __init__(self, cfg: SimulationConfig):
        self.sight_range = cfg.agent_sight_range
        self.half_fov_radians = math.radians(cfg.agent_fov_degrees) / 2.0
        self._cos_half_fov = math.cos(self.half_fov_radians)

    def perceive(
        self,
        body: Body,
        beacons: Sequence[SensorBeacon],
    ) -> Optional[SensorBeacon]:
        """Return the highest-priority active beacon inside the sight cone."""
        heading_vec = Vec3(
            math.cos(body.heading_radians),
            math.sin(body.heading_radians),
            0.0,
        )

        best: Optional[SensorBeacon] = None
        best_score = -math.inf
        max_distance_sq = self.sight_range * self.sight_range

        for beacon in beacons:
            if not beacon.active or beacon.acknowledged_by is not None:
                continue
            offset = _flatten(beacon.location - body.position)
            distance_sq = offset.lengthSquared()
            if distance_sq > max_distance_sq or distance_sq < 1e-6:
                continue

            distance = math.sqrt(distance_sq)
            cos_angle = (offset.x * heading_vec.x
                         + offset.y * heading_vec.y) / distance
            if cos_angle < self._cos_half_fov:
                continue

            # Combine priority with inverse distance so a high-priority
            # beacon nearby beats a low-priority one closer to the nose.
            score = beacon.priority - 0.05 * distance
            if score > best_score:
                best = beacon
                best_score = score

        return best


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
class Steering:
    """
    Reynolds-style steering primitives.

    Each method returns a *force* - the acceleration that would push the
    agent toward the desired velocity in one tick. States combine these
    forces via weighted sums; the result is clamped by ``max_force``.
    """

    def __init__(self, cfg: SimulationConfig):
        self.max_speed = cfg.agent_max_speed
        self.max_force = cfg.agent_max_force
        self.slowdown_radius = cfg.agent_slowdown_radius
        self.avoid_lookahead = cfg.agent_avoid_lookahead
        self.separation_radius = cfg.agent_separation_radius
        self.wander_circle_distance = cfg.agent_wander_circle_distance
        self.wander_circle_radius = cfg.agent_wander_circle_radius
        self.wander_jitter = cfg.agent_wander_jitter_radians

    def seek(self, body: Body, target: Vec3) -> Vec3:
        desired = _flatten(target - body.position)
        if desired.lengthSquared() < 1e-8:
            return Vec3(0, 0, 0)
        desired.normalize()
        desired *= self.max_speed
        return _truncate(desired - body.velocity, self.max_force)

    def arrive(self, body: Body, target: Vec3) -> Vec3:
        offset = _flatten(target - body.position)
        distance = offset.length()
        if distance < 1e-4:
            return -body.velocity
        ramped = self.max_speed * min(distance / self.slowdown_radius, 1.0)
        desired = offset * (ramped / distance)
        return _truncate(desired - body.velocity, self.max_force)

    def wander(self, body: Body, blackboard: Blackboard,
               rng: random.Random) -> Vec3:
        """Reynolds wander - smooth, plausibly aimless motion."""
        blackboard.wander_target_angle += rng.uniform(
            -self.wander_jitter, self.wander_jitter,
        )
        heading = body.heading_radians
        circle_center = body.position + Vec3(
            math.cos(heading) * self.wander_circle_distance,
            math.sin(heading) * self.wander_circle_distance,
            0.0,
        )
        offset = Vec3(
            math.cos(blackboard.wander_target_angle) * self.wander_circle_radius,
            math.sin(blackboard.wander_target_angle) * self.wander_circle_radius,
            0.0,
        )
        return self.seek(body, circle_center + offset)

    def avoid_obstacles(self, body: Body,
                        obstacles: Sequence[Obstacle]) -> Vec3:
        """
        Forward lookahead-ray test against circular obstacles. The
        worst (deepest-penetrated) hit drives a lateral force away from
        the obstacle centre. Cheap, deterministic, single allocation.
        """
        if body.velocity.lengthSquared() < 1e-6 or not obstacles:
            return Vec3(0, 0, 0)

        forward = Vec3(body.velocity)
        forward.normalize()
        lookahead_tip = body.position + forward * self.avoid_lookahead

        worst: Optional[Obstacle] = None
        worst_penetration = 0.0
        for obs in obstacles:
            offset = _flatten(lookahead_tip - obs.center)
            penetration = obs.radius - offset.length()
            if penetration > worst_penetration:
                worst_penetration = penetration
                worst = obs

        if worst is None:
            return Vec3(0, 0, 0)

        away = _flatten(lookahead_tip - worst.center)
        if away.lengthSquared() < 1e-6:
            # Degenerate: lookahead tip dead-centre on the obstacle.
            # Push perpendicular so we don't stall.
            away = Vec3(-forward.y, forward.x, 0.0)
        away.normalize()
        return away * self.max_force

    def separate(self, body: Body, peers: Sequence[Body]) -> Vec3:
        """Repel from peers inside the separation radius (inverse falloff)."""
        force = Vec3(0, 0, 0)
        for peer in peers:
            if peer is body:
                continue
            offset = _flatten(body.position - peer.position)
            distance = offset.length()
            if 1e-6 < distance < self.separation_radius:
                offset *= (self.separation_radius - distance) / distance
                force += offset
        return _truncate(force, self.max_force)


# ---------------------------------------------------------------------------
# Brain - FSM controller
# ---------------------------------------------------------------------------
class Brain:
    """
    Four-state FSM with explicit entry hooks and structured logging.

    States
    ------
    IDLE         bounded dwell, then promote to PATROL
    PATROL       arrive at a sampled waypoint while avoiding obstacles
    SEEK         pursue an acknowledged beacon
    INVESTIGATE  walk to last known location if the seek target expires
    """

    def __init__(self, cfg: SimulationConfig, agent_name: str,
                 rng: random.Random):
        self.cfg = cfg
        self.name = agent_name
        self.rng = rng
        self.state: AgentState = AgentState.IDLE
        self.blackboard = Blackboard()
        self._logger = logging.getLogger(f"brain.{agent_name}")
        self._enter_state(AgentState.IDLE, reason="spawn")

    # ------------------------------------------------------------------
    def _enter_state(self, new_state: AgentState, reason: str) -> None:
        previous = self.state
        self.state = new_state
        bb = self.blackboard

        if new_state is AgentState.IDLE:
            bb.state_timer = self.rng.uniform(*self.cfg.idle_duration)
            bb.patrol_waypoint = None

        elif new_state is AgentState.PATROL:
            bb.state_timer = self.rng.uniform(*self.cfg.patrol_duration)
            bb.pursued_beacon = None

        elif new_state is AgentState.SEEK:
            bb.state_timer = self.cfg.seek_timeout

        elif new_state is AgentState.INVESTIGATE:
            bb.state_timer = self.cfg.investigate_duration

        self._logger.info("FSM | %-11s -> %-11s | %s",
                          previous.value, new_state.value, reason)

    # ------------------------------------------------------------------
    def think(
        self,
        dt: float,
        body: Body,
        sensor: Sensor,
        steering: Steering,
        environment: "DigitalTwinEnvironment",
        peers: Sequence[Body],
    ) -> Vec3:
        bb = self.blackboard
        bb.state_timer -= dt
        bb.vector_log_timer -= dt
        bb.snapshot_timer -= dt

        # --- Perception preempts IDLE / PATROL ----------------------
        if self.state in (AgentState.IDLE, AgentState.PATROL,
                          AgentState.INVESTIGATE):
            perceived = sensor.perceive(body, environment.beacons)
            if perceived is not None and perceived.name != bb.pursued_beacon:
                bb.pursued_beacon = perceived.name
                bb.last_known_location = Vec3(perceived.location)
                self._enter_state(AgentState.SEEK,
                                  reason=f"perceived '{perceived.name}'")

        # --- State behaviour ---------------------------------------
        if self.state is AgentState.IDLE:
            if bb.state_timer <= 0.0:
                self._enter_state(AgentState.PATROL,
                                  reason="idle dwell complete")
            return steering.separate(body, peers)

        if self.state is AgentState.PATROL:
            if bb.patrol_waypoint is None:
                bb.patrol_waypoint = environment.sample_waypoint(
                    self.cfg.obstacle_clearance, self.rng,
                )
            if bb.state_timer <= 0.0:
                self._enter_state(AgentState.IDLE,
                                  reason="patrol timer expired")
                return Vec3(0, 0, 0)
            if _flatten(bb.patrol_waypoint - body.position).length() \
                    < self.cfg.agent_arrival_radius:
                self._enter_state(AgentState.IDLE,
                                  reason="waypoint reached")
                return Vec3(0, 0, 0)
            return (
                steering.arrive(body, bb.patrol_waypoint) * 1.0
                + steering.avoid_obstacles(body, environment.obstacles) * 1.6
                + steering.separate(body, peers) * 0.8
            )

        if self.state is AgentState.SEEK:
            beacon = environment.find_beacon(bb.pursued_beacon)
            if beacon is None or not beacon.active:
                self._enter_state(AgentState.INVESTIGATE,
                                  reason="target deactivated")
                return Vec3(0, 0, 0)
            bb.last_known_location = Vec3(beacon.location)

            if _flatten(beacon.location - body.position).length() \
                    < self.cfg.agent_arrival_radius:
                environment.acknowledge_beacon(beacon, by_agent=self.name)
                self._enter_state(AgentState.IDLE,
                                  reason=f"acknowledged '{beacon.name}'")
                return Vec3(0, 0, 0)

            if bb.state_timer <= 0.0:
                self._enter_state(AgentState.INVESTIGATE,
                                  reason="seek timeout")
                return Vec3(0, 0, 0)

            return (
                steering.arrive(body, beacon.location) * 1.0
                + steering.avoid_obstacles(body, environment.obstacles) * 1.5
                + steering.separate(body, peers) * 0.7
            )

        if self.state is AgentState.INVESTIGATE:
            lkp = bb.last_known_location
            if lkp is None or bb.state_timer <= 0.0:
                self._enter_state(AgentState.IDLE,
                                  reason="investigation timeout")
                return Vec3(0, 0, 0)
            if _flatten(lkp - body.position).length() \
                    < self.cfg.agent_arrival_radius:
                self._enter_state(AgentState.IDLE,
                                  reason="reached last-known location")
                return Vec3(0, 0, 0)
            return (
                steering.arrive(body, lkp) * 1.0
                + steering.avoid_obstacles(body, environment.obstacles) * 1.4
                + steering.separate(body, peers) * 0.8
            )

        # Exhaustive over AgentState; this is a defensive fallthrough.
        self._logger.error("unhandled state %s - resetting", self.state)
        self._enter_state(AgentState.IDLE, reason="defensive fallback")
        return Vec3(0, 0, 0)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    """A single autonomous unit: body + sensor + brain + steering + visuals."""

    def __init__(
        self,
        index: int,
        cfg: SimulationConfig,
        render_root: NodePath,
        rng: random.Random,
        start_position: Vec3,
        body_color: Vec4,
    ):
        self.name = f"agent-{index}"
        self.cfg = cfg
        self.rng = rng
        self.body = Body(position=Vec3(start_position))
        self.sensor = Sensor(cfg)
        self.steering = Steering(cfg)
        self.brain = Brain(cfg, agent_name=self.name, rng=rng)

        # Visual body - a small coloured prism.
        self.node = create_box(
            name=f"{self.name}-body",
            half_extents=Vec3(0.22, 0.32, 0.30),
            color=body_color,
        )
        self.node.setPos(start_position)
        self.node.reparentTo(render_root)

        self._debug_root: NodePath = render_root.attachNewNode(
            f"{self.name}-debug",
        )
        self._velocity_arrow: Optional[NodePath] = None
        self._fov_cone: Optional[NodePath] = None

    # ------------------------------------------------------------------
    def step(
        self,
        dt: float,
        environment: "DigitalTwinEnvironment",
        peers: Sequence[Body],
    ) -> NavigationVector:
        force = self.brain.think(
            dt, self.body, self.sensor, self.steering, environment, peers,
        )

        # Semi-implicit Euler integration.
        self.body.velocity = _truncate(
            self.body.velocity + force * dt,
            self.cfg.agent_max_speed,
        )
        # Bleed velocity to zero when idle so we settle cleanly.
        if self.brain.state is AgentState.IDLE:
            self.body.velocity *= max(0.0, 1.0 - 4.0 * dt)

        origin = Vec3(self.body.position)
        proposed = origin + self.body.velocity * dt
        clamped = environment.bounds.clamp(proposed)
        boundary_corrected = (clamped - proposed).length() > 1e-3
        if boundary_corrected:
            # Drop the component of velocity that pushed us out - this
            # preserves steering continuity at the arena edge: the agent
            # corrects its motion vector instead of teleporting back.
            if abs(clamped.x - proposed.x) > 1e-4:
                self.body.velocity.x *= -0.35
            if abs(clamped.y - proposed.y) > 1e-4:
                self.body.velocity.y *= -0.35

        self.body.position = clamped
        self.node.setPos(self.body.position)
        if self.body.velocity.lengthSquared() > 1e-4:
            heading_deg = math.degrees(self.body.heading_radians) - 90.0
            self.node.setH(heading_deg)

        # NavigationVector for telemetry - composed once per tick.
        direction = Vec3(self.body.velocity)
        speed = direction.length()
        if speed > 1e-6:
            direction *= 1.0 / speed
        return NavigationVector(
            origin=origin,
            target=clamped,
            direction=direction,
            distance=(clamped - origin).length(),
            step_length=speed * dt,
            proposed_position=proposed,
            corrected_position=clamped,
            boundary_corrected=boundary_corrected,
        )

    # ------------------------------------------------------------------
    def maybe_log_vector(self, vector: NavigationVector) -> None:
        bb = self.brain.blackboard
        if bb.vector_log_timer > 0.0:
            return
        bb.vector_log_timer = self.cfg.vector_report_period
        log.info(
            "vec[%s|%s] pos=(%+.2f,%+.2f) dir=(%+.2f,%+.2f) "
            "d=%.2f step=%.3f bnd=%s",
            self.name, self.brain.state.value,
            vector.origin.x, vector.origin.y,
            vector.direction.x, vector.direction.y,
            vector.distance, vector.step_length,
            vector.boundary_corrected,
        )

    def maybe_log_snapshot(self) -> None:
        bb = self.brain.blackboard
        if bb.snapshot_timer > 0.0:
            return
        bb.snapshot_timer = self.cfg.state_report_period
        log.info(
            "snap[%s] state=%s pos=(%+.2f,%+.2f) v=%.2f target=%s",
            self.name, self.brain.state.value,
            self.body.position.x, self.body.position.y,
            self.body.speed,
            bb.pursued_beacon or "-",
        )

    # ------------------------------------------------------------------
    def redraw_debug(self) -> None:
        """Refresh velocity arrow + FOV cone overlays."""
        if self._velocity_arrow is not None:
            self._velocity_arrow.removeNode()
        if self._fov_cone is not None:
            self._fov_cone.removeNode()

        if self.body.velocity.lengthSquared() > 1e-4:
            segs = LineSegs()
            segs.setColor(1.0, 1.0, 0.25, 1.0)
            segs.setThickness(2.0)
            base = self.body.position + Vec3(0, 0, 0.45)
            tip = base + self.body.velocity * 0.4
            segs.moveTo(base)
            segs.drawTo(tip)
            self._velocity_arrow = self._debug_root.attachNewNode(segs.create())

        # FOV cone - two edge rays + arc baseline.
        segs = LineSegs()
        segs.setColor(0.25, 0.85, 1.0, 1.0)
        segs.setThickness(1.1)
        heading = self.body.heading_radians
        half_fov = math.radians(self.cfg.agent_fov_degrees) / 2.0
        sight = self.cfg.agent_sight_range
        origin = self.body.position + Vec3(0, 0, 0.05)
        for sign in (-1.0, +1.0):
            ang = heading + sign * half_fov
            tip = origin + Vec3(math.cos(ang) * sight,
                                math.sin(ang) * sight, 0.0)
            segs.moveTo(origin)
            segs.drawTo(tip)
        prev = None
        for i in range(17):
            t = i / 16.0
            ang = heading - half_fov + t * (2.0 * half_fov)
            point = origin + Vec3(math.cos(ang) * sight,
                                  math.sin(ang) * sight, 0.0)
            if prev is None:
                segs.moveTo(point)
            else:
                segs.drawTo(point)
            prev = point
        self._fov_cone = self._debug_root.attachNewNode(segs.create())


# ---------------------------------------------------------------------------
# Digital-twin environment
# ---------------------------------------------------------------------------
class DigitalTwinEnvironment:
    """
    Procedural arena. Owns world bounds, named obstacles, and the
    sensor beacon set. All geometry is generated at runtime - the
    repository runs from a clean ``pip install panda3d`` with no model
    files on disk.
    """

    # Named operational assets - each contributes a visual obstacle.
    ASSET_SPECS: Tuple[
        Tuple[str, Vec3, Vec3, Vec4, float], ...,
    ] = (
        ("control-cabinet", Vec3(-4.9, 3.2, 0.58),
         Vec3(0.60, 1.20, 0.58), Vec4(0.83, 0.57, 0.25, 1), 1.30),
        ("pump-station", Vec3(4.1, 2.5, 0.48),
         Vec3(1.05, 0.78, 0.48), Vec4(0.42, 0.68, 0.88, 1), 1.30),
        ("storage-rack", Vec3(-2.6, -4.3, 0.68),
         Vec3(1.28, 0.52, 0.68), Vec4(0.63, 0.72, 0.55, 1), 1.30),
        ("inspection-node", Vec3(3.7, -3.9, 0.76),
         Vec3(0.46, 0.46, 0.76), Vec4(0.86, 0.34, 0.40, 1), 1.00),
        ("data-relay", Vec3(0.2, 4.8, 0.62),
         Vec3(0.42, 0.42, 0.62), Vec4(0.66, 0.52, 0.92, 1), 0.95),
    )

    # Beacon set - semantic digital-twin events the agents can perceive.
    BEACON_SPECS: Tuple[Tuple[str, Vec3, float], ...] = (
        ("thermal-anomaly", Vec3(-5.8, -1.5, 0.38), 0.95),
        ("operator-ping", Vec3(5.6, -0.4, 0.38), 0.75),
        ("data-relay-check", Vec3(0.2, 6.3, 0.38), 0.85),
        ("inventory-audit", Vec3(-5.0, -5.4, 0.38), 0.58),
        ("pump-vibration", Vec3(5.8, 4.2, 0.38), 0.92),
    )

    def __init__(self, cfg: SimulationConfig, render_root: NodePath):
        self.cfg = cfg
        self.render_root = render_root
        ext = cfg.arena_half_extent
        self.bounds = WorldBounds(
            min_x=-ext + 1.0, max_x=ext - 1.0,
            min_y=-ext + 1.0, max_y=ext - 1.0,
        )
        self.obstacles: Tuple[Obstacle, ...] = self._build_assets()
        self.beacons: List[SensorBeacon] = self._build_beacons()
        self._build_floor()
        self._build_grid()

    # ------------------------------------------------------------------
    def _build_floor(self) -> None:
        floor = create_box(
            name="operations-floor",
            half_extents=Vec3(self.cfg.arena_half_extent,
                              self.cfg.arena_half_extent, 0.04),
            color=Vec4(0.09, 0.11, 0.13, 1.0),
        )
        floor.setZ(-0.04)
        floor.reparentTo(self.render_root)

    def _build_grid(self) -> None:
        grid = LineSegs("digital-twin-grid")
        grid.setColor(0.28, 0.33, 0.36, 1.0)
        grid.setThickness(1.0)
        step = 1.0
        ext = self.cfg.arena_half_extent
        for i in range(int(ext * 2.0 / step) + 1):
            offset = -ext + i * step
            grid.moveTo(-ext, offset, 0.02)
            grid.drawTo(ext, offset, 0.02)
            grid.moveTo(offset, -ext, 0.02)
            grid.drawTo(offset, ext, 0.02)
        self.render_root.attachNewNode(grid.create())

    def _build_assets(self) -> Tuple[Obstacle, ...]:
        obstacles: List[Obstacle] = []
        for name, position, half_ext, color, radius in self.ASSET_SPECS:
            node = create_box(name=name, half_extents=half_ext, color=color)
            node.setPos(position)
            node.reparentTo(self.render_root)
            obstacles.append(Obstacle(
                name=name,
                center=Vec3(position.x, position.y, 0.38),
                radius=radius,
            ))
            log.info("asset %-16s pos=(%+.2f,%+.2f) r=%.2f",
                     name, position.x, position.y, radius)
        return tuple(obstacles)

    def _build_beacons(self) -> List[SensorBeacon]:
        beacons: List[SensorBeacon] = []
        for name, location, priority in self.BEACON_SPECS:
            node = create_box(
                name=f"beacon-{name}",
                half_extents=Vec3(0.18, 0.18, 0.22),
                color=Vec4(0.25, 0.25, 0.28, 1.0),
            )
            node.setPos(location)
            node.reparentTo(self.render_root)
            beacons.append(SensorBeacon(
                name=name, location=Vec3(location),
                priority=priority, node=node,
            ))
        return beacons

    # ------------------------------------------------------------------
    def sample_waypoint(self, clearance: float,
                        rng: random.Random) -> Vec3:
        """Random navigable point inside bounds and outside obstacles."""
        for _ in range(64):
            point = self.bounds.random_point(rng)
            if all(not o.contains_2d(point, clearance) for o in self.obstacles):
                return point
        log.warning("waypoint sampler exhausted retries - falling back to origin")
        return Vec3(0.0, 0.0, 0.38)

    def find_beacon(self, name: Optional[str]) -> Optional[SensorBeacon]:
        if name is None:
            return None
        for beacon in self.beacons:
            if beacon.name == name:
                return beacon
        return None

    def activate_beacon(self, beacon: SensorBeacon, now: float) -> None:
        beacon.active = True
        beacon.activated_at = now
        beacon.acknowledged_by = None
        beacon.node.setColor(Vec4(1.0, 0.55, 0.20, 1.0))
        log.info("beacon ACTIVE   '%s' priority=%.2f",
                 beacon.name, beacon.priority)

    def deactivate_beacon(self, beacon: SensorBeacon, reason: str) -> None:
        beacon.active = False
        beacon.node.setColor(Vec4(0.25, 0.25, 0.28, 1.0))
        log.info("beacon CLEAR    '%s' (%s)", beacon.name, reason)

    def acknowledge_beacon(self, beacon: SensorBeacon, by_agent: str) -> None:
        beacon.acknowledged_by = by_agent
        self.deactivate_beacon(beacon, reason=f"acknowledged by {by_agent}")


# ---------------------------------------------------------------------------
# Director - drives beacon activation over time
# ---------------------------------------------------------------------------
class Director:
    """Activates and times out beacons to drive agent behaviour."""

    def __init__(self, cfg: SimulationConfig,
                 environment: DigitalTwinEnvironment,
                 rng: random.Random):
        self.cfg = cfg
        self.env = environment
        self.rng = rng
        self._elapsed = 0.0
        self._next_activation = rng.uniform(*cfg.beacon_activation_interval)

    def tick(self, dt: float) -> None:
        self._elapsed += dt

        # Expire any active beacons past their lifetime.
        for beacon in self.env.beacons:
            if beacon.active and (self._elapsed - beacon.activated_at) \
                    > self.cfg.beacon_lifetime:
                self.env.deactivate_beacon(beacon, reason="lifetime exceeded")

        # Activate a new beacon when due, respecting the active cap.
        if self._elapsed < self._next_activation:
            return
        active_count = sum(1 for b in self.env.beacons if b.active)
        if active_count >= self.cfg.beacon_max_active:
            return

        candidates = [b for b in self.env.beacons if not b.active]
        if not candidates:
            return
        # Priority-weighted selection so operationally important signals
        # fire more often than low-priority ones.
        weights = [b.priority for b in candidates]
        chosen = self.rng.choices(candidates, weights=weights, k=1)[0]
        self.env.activate_beacon(chosen, now=self._elapsed)
        self._next_activation = self._elapsed + self.rng.uniform(
            *self.cfg.beacon_activation_interval,
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class SimulationApplication(ShowBase):
    """
    Panda3D application host. Owns the scene graph, agents, environment,
    and the master simulation tick. All decision logic lives in each
    agent's ``Brain`` - this class only orchestrates.
    """

    AGENT_PALETTE: Tuple[Vec4, ...] = (
        Vec4(0.06, 0.80, 0.88, 1.0),  # cyan
        Vec4(0.95, 0.55, 0.30, 1.0),  # amber
        Vec4(0.55, 0.85, 0.45, 1.0),  # lime
        Vec4(0.85, 0.45, 0.85, 1.0),  # magenta
        Vec4(0.95, 0.88, 0.40, 1.0),  # gold
    )

    MAX_FRAME_DT = 0.1   # seconds - safety clamp against scheduler stalls

    def __init__(self, cfg: SimulationConfig, rng: random.Random):
        super().__init__()
        self.cfg = cfg
        self.rng = rng
        self.disableMouse()
        self.setBackgroundColor(0.035, 0.044, 0.052, 1.0)

        self._configure_lighting()
        self._configure_camera()

        self.environment = DigitalTwinEnvironment(cfg, self.render)
        self.director = Director(cfg, self.environment, rng)
        self.agents: List[Agent] = self._spawn_agents()

        self.taskMgr.add(self._simulation_tick, "advance-simulation")
        log.info(
            "simulation ready | agents=%d obstacles=%d arena=%.0f debug=%s",
            cfg.agent_count, len(self.environment.obstacles),
            cfg.arena_half_extent * 2.0, cfg.debug_draw,
        )

    # ------------------------------------------------------------------
    def _configure_lighting(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.34, 0.36, 0.42, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))
        key = DirectionalLight("key")
        key.setColor(Vec4(0.90, 0.92, 0.96, 1.0))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-35.0, -55.0, 0.0)
        self.render.setLight(key_np)

    def _configure_camera(self) -> None:
        ext = self.cfg.arena_half_extent
        self.camera.setPos(0.0, -ext * 2.4, ext * 1.5)
        self.camera.lookAt(0.0, 0.0, 0.0)

    def _spawn_agents(self) -> List[Agent]:
        agents: List[Agent] = []
        for i in range(self.cfg.agent_count):
            angle = (i / max(1, self.cfg.agent_count)) * 2.0 * math.pi
            start = Vec3(math.cos(angle) * 2.0,
                         math.sin(angle) * 2.0, 0.38)
            color = self.AGENT_PALETTE[i % len(self.AGENT_PALETTE)]
            agents.append(Agent(
                index=i, cfg=self.cfg, render_root=self.render,
                rng=self.rng, start_position=start, body_color=color,
            ))
        return agents

    # ------------------------------------------------------------------
    def _simulation_tick(self, task: Task.Task) -> int:
        dt = min(ClockObject.getGlobalClock().getDt(), self.MAX_FRAME_DT)

        self.director.tick(dt)

        # Snapshot peer bodies once so steering reads are order-independent.
        peer_bodies = [a.body for a in self.agents]

        for agent in self.agents:
            vector = agent.step(dt, self.environment, peer_bodies)
            agent.maybe_log_vector(vector)
            agent.maybe_log_snapshot()
            if self.cfg.debug_draw:
                agent.redraw_debug()

        return Task.cont


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str]) -> Tuple[SimulationConfig, random.Random]:
    parser = argparse.ArgumentParser(
        description="Multi-agent autonomous behaviour simulation (Panda3D).",
    )
    parser.add_argument("--agents", type=int, default=3,
                        help="Number of agents to spawn (default 3).")
    parser.add_argument("--arena", type=float, default=8.0,
                        help="Arena half-extent in world units (default 8).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for deterministic replay.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable FOV / velocity overlays.")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    if args.seed is not None:
        log.info("rng seeded with %d", args.seed)

    cfg = SimulationConfig(
        agent_count=max(1, args.agents),
        arena_half_extent=max(5.0, args.arena),
        debug_draw=not args.no_debug,
    )
    return cfg, rng


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg, rng = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        app = SimulationApplication(cfg, rng)
        app.run()
    except KeyboardInterrupt:
        log.info("interrupted by user")
        return 0
    except Exception:   # top-level guard so the engine fails loudly, not silently
        log.exception("simulation crashed:")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
