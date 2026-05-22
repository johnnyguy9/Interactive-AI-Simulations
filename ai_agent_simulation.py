"""
Advanced Panda3D Autonomous Agent Simulation

This module demonstrates a compact interactive systems architecture built around
Panda3D's rendering loop and a deterministic finite state machine. The simulated
agent navigates a procedural 3D digital twin, evaluates boundaries and obstacles,
and emits clear console telemetry for state transitions and vector calculations.

Run:
    python ai_agent_simulation.py

Dependency:
    pip install panda3d
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees
from random import Random

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


loadPrcFileData(
    "",
    "\n".join(
        (
            "window-title Advanced Interactive AI Simulation",
            "win-size 1280 720",
            "sync-video true",
            "show-frame-rate-meter true",
        )
    ),
)


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
LOGGER = logging.getLogger("ai-agent-simulation")


class AgentState(Enum):
    """Finite-state machine states for the autonomous interactive agent."""

    PATROL = "Patrol"
    SEEK = "Seek"
    IDLE = "Idle"


@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned navigable limits for the digital twin floor plan."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, point: Vec3) -> bool:
        """Return True when a point is inside the operational envelope."""

        return self.min_x <= point.x <= self.max_x and self.min_y <= point.y <= self.max_y

    def clamp(self, point: Vec3) -> Vec3:
        """Project a point back into the navigable envelope."""

        return Vec3(
            max(self.min_x, min(self.max_x, point.x)),
            max(self.min_y, min(self.max_y, point.y)),
            point.z,
        )

    def random_point(self, rng: Random, z: float = 0.38) -> Vec3:
        """Sample a random waypoint from the walkable area."""

        return Vec3(
            rng.uniform(self.min_x, self.max_x),
            rng.uniform(self.min_y, self.max_y),
            z,
        )


@dataclass(frozen=True)
class Obstacle:
    """Circular navigation exclusion zone tied to a visible 3D asset."""

    name: str
    center: Vec3
    radius: float

    def contains(self, point: Vec3, clearance: float) -> bool:
        """Return True when a point violates this obstacle clearance."""

        return (point - self.center).length() <= self.radius + clearance


@dataclass(frozen=True)
class SensorBeacon:
    """
    Simulated event source used by the Seek state.

    A production system might replace this with live telemetry, a message bus,
    CV detections, operator commands, or an agentic orchestration layer. Here it
    remains intentionally procedural so the repository stays self-contained.
    """

    name: str
    location: Vec3
    priority: float


@dataclass(frozen=True)
class MovementConfig:
    """Simulation tuning values for motion, sensing, and console telemetry."""

    patrol_speed: float = 2.15
    seek_speed: float = 3.45
    arrival_distance: float = 0.16
    obstacle_clearance: float = 0.7
    idle_min_seconds: float = 0.75
    idle_max_seconds: float = 2.0
    seek_timeout_seconds: float = 6.0
    sensor_scan_period_seconds: float = 1.25
    state_report_period_seconds: float = 1.0
    vector_report_period_seconds: float = 0.55
    beacon_activation_probability: float = 0.42


@dataclass(frozen=True)
class NavigationVector:
    """Computed vector solution for one motion step."""

    origin: Vec3
    target: Vec3
    direction: Vec3
    distance: float
    step_length: float
    proposed_position: Vec3
    corrected_position: Vec3
    boundary_corrected: bool


def create_box(name: str, half_extents: Vec3, color: Vec4) -> NodePath:
    """
    Create a lit box mesh without external model assets.

    Each face receives its own vertices and normals. This is slightly more
    verbose than loading a model, but it keeps the repository portable and makes
    the rendering pipeline explicit for portfolio review.
    """

    vertex_format = GeomVertexFormat.getV3n3c4()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UHStatic)
    vertices = GeomVertexWriter(vertex_data, "vertex")
    normals = GeomVertexWriter(vertex_data, "normal")
    colors = GeomVertexWriter(vertex_data, "color")

    hx, hy, hz = half_extents.x, half_extents.y, half_extents.z
    faces = (
        (Vec3(1, 0, 0), (Vec3(hx, -hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, hy, hz), Vec3(hx, -hy, hz))),
        (Vec3(-1, 0, 0), (Vec3(-hx, hy, -hz), Vec3(-hx, -hy, -hz), Vec3(-hx, -hy, hz), Vec3(-hx, hy, hz))),
        (Vec3(0, 1, 0), (Vec3(-hx, hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz))),
        (Vec3(0, -1, 0), (Vec3(hx, -hy, -hz), Vec3(-hx, -hy, -hz), Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz))),
        (Vec3(0, 0, 1), (Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz))),
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

    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


class DigitalTwinEnvironment:
    """
    Owns spatial constraints, visual assets, and procedural event generation.

    The environment is intentionally separate from the agent. This keeps the
    agent controller focused on decisions while the environment owns boundaries,
    collision semantics, and digital twin assets.
    """

    WORLD_SIZE = 8.0

    def __init__(self, render_root: NodePath, rng: Random) -> None:
        self.render_root = render_root
        self.rng = rng
        self.bounds = WorldBounds(
            min_x=-self.WORLD_SIZE + 1.0,
            max_x=self.WORLD_SIZE - 1.0,
            min_y=-self.WORLD_SIZE + 1.0,
            max_y=self.WORLD_SIZE - 1.0,
        )
        self.obstacles: tuple[Obstacle, ...] = self._create_scene_assets()
        self.beacons = self._create_sensor_beacons()
        self._create_floor_grid()

    def sample_waypoint(self, clearance: float) -> Vec3:
        """
        Return a valid navigation waypoint inside bounds and outside obstacles.

        The bounded retry loop is a defensive guard: it prevents a malformed
        scene from trapping the controller in an infinite random search.
        """

        for _ in range(96):
            candidate = self.bounds.random_point(self.rng)
            if self.is_navigable(candidate, clearance):
                return candidate

        LOGGER.warning("Waypoint sampler exhausted retries; falling back to origin")
        return Vec3(0.0, 0.0, 0.38)

    def is_navigable(self, point: Vec3, clearance: float) -> bool:
        """Evaluate both world boundaries and obstacle clearance constraints."""

        return self.bounds.contains(point) and not any(
            obstacle.contains(point, clearance) for obstacle in self.obstacles
        )

    def detect_beacon(self, probability: float) -> SensorBeacon | None:
        """
        Simulate a periodic sensor scan.

        Higher-priority beacons receive a larger share of selection weight. The
        agent decides whether and how to respond; the environment only reports
        that a meaningful signal exists.
        """

        if self.rng.random() > probability:
            return None

        weighted_beacons = sorted(self.beacons, key=lambda beacon: beacon.priority, reverse=True)
        return self.rng.choice(weighted_beacons[:3])

    def _create_scene_assets(self) -> tuple[Obstacle, ...]:
        """Create digital twin assets that double as navigation exclusions."""

        floor = create_box(
            name="operations-floor",
            half_extents=Vec3(self.WORLD_SIZE, self.WORLD_SIZE, 0.04),
            color=Vec4(0.10, 0.12, 0.14, 1.0),
        )
        floor.reparentTo(self.render_root)
        floor.setZ(-0.04)

        asset_specs = (
            ("control-cabinet", Vec3(-4.9, 3.2, 0.58), Vec3(0.60, 1.20, 0.58), Vec4(0.83, 0.57, 0.25, 1), 1.38),
            ("pump-station", Vec3(4.1, 2.5, 0.48), Vec3(1.05, 0.78, 0.48), Vec4(0.42, 0.68, 0.88, 1), 1.30),
            ("storage-rack", Vec3(-2.6, -4.3, 0.68), Vec3(1.28, 0.52, 0.68), Vec4(0.63, 0.72, 0.55, 1), 1.35),
            ("inspection-node", Vec3(3.7, -3.9, 0.76), Vec3(0.46, 0.46, 0.76), Vec4(0.86, 0.34, 0.40, 1), 1.05),
            ("data-relay", Vec3(0.2, 4.8, 0.62), Vec3(0.42, 0.42, 0.62), Vec4(0.66, 0.52, 0.92, 1), 0.92),
        )

        obstacles: list[Obstacle] = []
        for name, position, half_extents, color, radius in asset_specs:
            asset = create_box(name, half_extents, color)
            asset.reparentTo(self.render_root)
            asset.setPos(position)
            obstacles.append(Obstacle(name=name, center=Vec3(position.x, position.y, 0.38), radius=radius))

        return tuple(obstacles)

    def _create_sensor_beacons(self) -> tuple[SensorBeacon, ...]:
        """Place semantic targets around the floor for simulated Seek events."""

        return (
            SensorBeacon("thermal-anomaly", Vec3(-5.8, -1.5, 0.38), 0.95),
            SensorBeacon("operator-ping", Vec3(5.6, -0.4, 0.38), 0.75),
            SensorBeacon("data-relay-check", Vec3(0.2, 6.3, 0.38), 0.85),
            SensorBeacon("inventory-audit", Vec3(-5.0, -5.4, 0.38), 0.58),
            SensorBeacon("pump-vibration", Vec3(5.8, 4.2, 0.38), 0.92),
        )

    def _create_floor_grid(self) -> None:
        """Render a lightweight grid that reads as a digital twin plan view."""

        grid = LineSegs("digital-twin-grid")
        grid.setColor(0.28, 0.33, 0.36, 1.0)
        grid.setThickness(1.0)

        step = 1.0
        line_count = int((self.WORLD_SIZE * 2.0) / step) + 1
        origin = -self.WORLD_SIZE

        for index in range(line_count):
            offset = origin + index * step
            grid.moveTo(-self.WORLD_SIZE, offset, 0.02)
            grid.drawTo(self.WORLD_SIZE, offset, 0.02)
            grid.moveTo(offset, -self.WORLD_SIZE, 0.02)
            grid.drawTo(offset, self.WORLD_SIZE, 0.02)

        self.render_root.attachNewNode(grid.create())


class AutonomousAgent:
    """
    Finite-state autonomous controller connected to a Panda3D NodePath.

    The NodePath is the bridge between AI logic and rendering. This controller
    computes movement vectors, then writes transforms to the NodePath. Panda3D
    reads those transforms during its render pass and draws the updated world.
    """

    def __init__(
        self,
        body: NodePath,
        environment: DigitalTwinEnvironment,
        rng: Random,
        config: MovementConfig | None = None,
    ) -> None:
        self.body = body
        self.environment = environment
        self.rng = rng
        self.config = config or MovementConfig()

        self.state = AgentState.IDLE
        self.state_elapsed = 0.0
        self.idle_duration = self._next_idle_duration()
        self.patrol_target = self.environment.sample_waypoint(self.config.obstacle_clearance)
        self.seek_target: Vec3 | None = None
        self.active_beacon: SensorBeacon | None = None

        self.sensor_elapsed = 0.0
        self.state_report_elapsed = self.config.state_report_period_seconds
        self.vector_report_elapsed = self.config.vector_report_period_seconds

        self.body.setPos(0.0, 0.0, 0.38)
        self.transition_to(AgentState.PATROL, "initial patrol route assigned")

    def update(self, dt: float) -> None:
        """
        Advance the FSM and movement solution for one frame.

        Panda3D invokes this method indirectly through the task manager. Keeping
        all game-time input as an explicit delta makes the controller testable
        and resilient to frame-rate differences.
        """

        self.state_elapsed += dt
        self.sensor_elapsed += dt
        self.state_report_elapsed += dt
        self.vector_report_elapsed += dt

        if self.state is AgentState.PATROL:
            self._update_patrol(dt)
        elif self.state is AgentState.SEEK:
            self._update_seek(dt)
        elif self.state is AgentState.IDLE:
            self._update_idle()

        if self.state_report_elapsed >= self.config.state_report_period_seconds:
            self.state_report_elapsed = 0.0
            self._log_state_snapshot()

    def transition_to(self, next_state: AgentState, reason: str) -> None:
        """
        Execute a controlled finite-state transition.

        Centralizing this hook keeps state entry side effects auditable: timers
        reset, state-specific parameters update, and transition logs stay
        consistent across the entire AI lifecycle.
        """

        previous = self.state
        self.state = next_state
        self.state_elapsed = 0.0

        if next_state is AgentState.IDLE:
            self.idle_duration = self._next_idle_duration()

        LOGGER.info("FSM transition | %s -> %s | reason=%s", previous.value, next_state.value, reason)

    def _update_patrol(self, dt: float) -> None:
        """Default roaming behavior with periodic sensor evaluation."""

        vector = self._move_toward(self.patrol_target, self.config.patrol_speed, dt)
        self._log_vector_solution("patrol", vector)

        if vector.boundary_corrected:
            self.patrol_target = self.environment.sample_waypoint(self.config.obstacle_clearance)
            self.transition_to(AgentState.IDLE, "boundary correction requested route replanning")
            return

        if self._arrived_at(self.patrol_target):
            self.transition_to(AgentState.IDLE, "patrol waypoint reached")
            return

        if self.sensor_elapsed >= self.config.sensor_scan_period_seconds:
            self.sensor_elapsed = 0.0
            beacon = self.environment.detect_beacon(self.config.beacon_activation_probability)
            if beacon is not None and self.environment.is_navigable(beacon.location, self.config.obstacle_clearance):
                self.active_beacon = beacon
                self.seek_target = beacon.location
                self.transition_to(AgentState.SEEK, f"beacon detected: {beacon.name}")

    def _update_seek(self, dt: float) -> None:
        """Goal-directed response to an environmental signal."""

        if self.seek_target is None:
            self.active_beacon = None
            self.transition_to(AgentState.IDLE, "seek entered without target")
            return

        vector = self._move_toward(self.seek_target, self.config.seek_speed, dt)
        self._log_vector_solution("seek", vector)

        if vector.boundary_corrected:
            self.seek_target = None
            self.active_beacon = None
            self.transition_to(AgentState.IDLE, "seek target required boundary correction")
            return

        if self._arrived_at(self.seek_target):
            beacon_name = self.active_beacon.name if self.active_beacon else "unknown"
            self.seek_target = None
            self.active_beacon = None
            self.transition_to(AgentState.IDLE, f"seek objective completed: {beacon_name}")
            return

        if self.state_elapsed >= self.config.seek_timeout_seconds:
            self.seek_target = None
            self.active_beacon = None
            self.transition_to(AgentState.IDLE, "seek timeout exceeded")

    def _update_idle(self) -> None:
        """
        Short deliberate dwell state.

        Real interactive systems often need dwell windows for animation blending,
        decision batching, operator readability, or external message handling.
        """

        if self.state_elapsed >= self.idle_duration:
            self.patrol_target = self.environment.sample_waypoint(self.config.obstacle_clearance)
            self.transition_to(AgentState.PATROL, "idle dwell complete")

    def _move_toward(self, target: Vec3, speed: float, dt: float) -> NavigationVector:
        """Calculate and apply one navigation vector toward the active target."""

        origin = self.body.getPos()
        target_vector = target - origin
        distance = target_vector.length()

        if distance <= self.config.arrival_distance:
            direction = Vec3(0.0, 0.0, 0.0)
            proposed_position = origin
            corrected_position = origin
            step_length = 0.0
            boundary_corrected = False
        else:
            direction = target_vector.normalized()
            step_length = min(speed * dt, distance)
            proposed_position = origin + direction * step_length
            corrected_position = self.environment.bounds.clamp(proposed_position)
            boundary_corrected = (corrected_position - proposed_position).length() > 0.001

            self.body.setPos(corrected_position)
            self.body.setH(degrees(atan2(-direction.x, direction.y)))

        return NavigationVector(
            origin=origin,
            target=target,
            direction=direction,
            distance=distance,
            step_length=step_length,
            proposed_position=proposed_position,
            corrected_position=corrected_position,
            boundary_corrected=boundary_corrected,
        )

    def _arrived_at(self, target: Vec3) -> bool:
        """Determine whether the current NodePath position satisfies arrival."""

        return (self.body.getPos() - target).length() <= self.config.arrival_distance

    def _next_idle_duration(self) -> float:
        """Sample the next dwell duration for the Idle state."""

        return self.rng.uniform(self.config.idle_min_seconds, self.config.idle_max_seconds)

    def _log_vector_solution(self, channel: str, vector: NavigationVector) -> None:
        """Throttle and emit vector telemetry for console inspection."""

        if self.vector_report_elapsed < self.config.vector_report_period_seconds:
            return

        self.vector_report_elapsed = 0.0
        LOGGER.info(
            "vector[%s] | origin=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) "
            "direction=(%.3f, %.3f, %.3f) distance=%.3f step=%.3f corrected=%s",
            channel,
            vector.origin.x,
            vector.origin.y,
            vector.origin.z,
            vector.target.x,
            vector.target.y,
            vector.target.z,
            vector.direction.x,
            vector.direction.y,
            vector.direction.z,
            vector.distance,
            vector.step_length,
            vector.boundary_corrected,
        )

    def _log_state_snapshot(self) -> None:
        """Emit a compact operational status line once per reporting period."""

        position = self.body.getPos()
        target = self.seek_target if self.state is AgentState.SEEK else self.patrol_target
        LOGGER.info(
            "agent snapshot | state=%s position=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f)",
            self.state.value,
            position.x,
            position.y,
            position.z,
            target.x,
            target.y,
            target.z,
        )


class SimulationApplication(ShowBase):
    """
    Panda3D application boundary.

    ShowBase owns the window, scene graph, render pipeline, and task manager.
    This class wires those engine services to the simulation domain model.
    """

    MAX_FRAME_DT = 0.1

    def __init__(self) -> None:
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.035, 0.044, 0.052, 1.0)

        self.rng = Random()
        self._configure_lighting()
        self._configure_camera()

        self.environment = DigitalTwinEnvironment(self.render, self.rng)
        agent_body = create_box(
            name="autonomous-agent-body",
            half_extents=Vec3(0.30, 0.45, 0.38),
            color=Vec4(0.06, 0.80, 0.88, 1.0),
        )
        agent_body.reparentTo(self.render)

        self.agent = AutonomousAgent(agent_body, self.environment, self.rng)

        # Panda3D's task manager is the main simulation heartbeat. Returning
        # Task.cont from the callback keeps this update active every frame.
        self.taskMgr.add(self._update_simulation, "advance-autonomous-agent-fsm")

    def _update_simulation(self, task: Task) -> int:
        """
        Frame callback registered with Panda3D's task manager.

        The engine supplies real elapsed frame time through ClockObject. The
        clamp prevents a window move, debugger pause, or OS scheduling spike from
        generating an unrealistic navigation jump.
        """

        dt = min(ClockObject.getGlobalClock().getDt(), self.MAX_FRAME_DT)
        self.agent.update(dt)
        return Task.cont

    def _configure_lighting(self) -> None:
        """Install a simple, readable lighting rig for procedural geometry."""

        ambient = AmbientLight("ambient-light")
        ambient.setColor(Vec4(0.35, 0.38, 0.42, 1.0))
        ambient_path = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_path)

        key_light = DirectionalLight("key-light")
        key_light.setColor(Vec4(0.88, 0.91, 0.96, 1.0))
        key_light_path = self.render.attachNewNode(key_light)
        key_light_path.setHpr(-35.0, -55.0, 0.0)
        self.render.setLight(key_light_path)

    def _configure_camera(self) -> None:
        """Position a top-oblique camera over the full digital twin floor."""

        self.camera.setPos(0.0, -18.0, 12.0)
        self.camera.lookAt(0.0, 0.0, 0.0)


def main() -> None:
    """Application entry point."""

    SimulationApplication().run()


if __name__ == "__main__":
    main()
