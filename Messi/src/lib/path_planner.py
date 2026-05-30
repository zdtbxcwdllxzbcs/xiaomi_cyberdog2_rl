"""A* path planner for the striker.

The planner works in the red field frame used by the rest of the project:
``x`` spans the field width and ``y`` spans the field length. The ROS node
listens for the striker target selected by ``striker.py`` and republishes a
waypoint path for ``path_follower.py``. The pure ``AStarPathPlanner`` class is
kept independent from ROS messages so it can be unit-tested without a running
ROS graph.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.node import Node

try:
    from nav_msgs.msg import Path as NavPath
except ImportError:
    # Some lightweight simulation environments only install geometry_msgs.
    # In that case the path topic uses PoseArray with identical waypoint data.
    NavPath = None

from .geometry import Point2D, clamp, quaternion_from_yaw, yaw_to_point
from .topics import soccer_topic


PATH_MESSAGE_TYPE = NavPath if NavPath is not None else PoseArray
GridCell = Tuple[int, int]


@dataclass(frozen=True)
class CircularObstacle:
    """Inflated circular obstacle in field coordinates."""

    center: Point2D
    radius: float


class AStarPathPlanner:
    """Grid-based A* planner with circular obstacle inflation."""

    def __init__(
        self,
        field_length: float = 10.0,
        field_width: float = 5.5,
        resolution: float = 0.1,
        boundary_margin: float = 0.15,
        max_waypoints: int = 160,
        max_expansions: int = 0,
    ):
        self.field_length = float(field_length)
        self.field_width = float(field_width)
        self.resolution = max(float(resolution), 0.02)
        self.boundary_margin = max(float(boundary_margin), 0.0)
        self.max_waypoints = max(int(max_waypoints), 2)
        self.max_expansions = max(int(max_expansions), 0)

        # In the red frame, x is lateral across the pitch and y is goal-to-goal.
        self.x_min = -self.field_width * 0.5 + self.boundary_margin
        self.x_max = self.field_width * 0.5 - self.boundary_margin
        self.y_min = -self.field_length * 0.5 + self.boundary_margin
        self.y_max = self.field_length * 0.5 - self.boundary_margin
        self.x_count = max(1, int(round((self.x_max - self.x_min) / self.resolution)))
        self.y_count = max(1, int(round((self.y_max - self.y_min) / self.resolution)))

    def plan(
        self,
        start: Point2D,
        goal: Point2D,
        obstacles: Sequence[CircularObstacle],
    ) -> List[Point2D]:
        """Plan a path from ``start`` to ``goal`` while avoiding obstacles."""

        start = self.clamp_to_field(start)
        goal = self.clamp_to_field(goal)
        if self._segment_clear(start, goal, obstacles):
            return [start, goal]

        start_cell = self.point_to_cell(start)
        goal_cell = self.point_to_cell(goal)
        blocked = self.inflate_obstacles(obstacles)

        # The robot may already overlap an inflated obstacle; allow the first
        # and final cell so A* can still escape or approach a tight target.
        blocked.discard(start_cell)
        blocked.discard(goal_cell)

        frontier: List[Tuple[float, GridCell]] = [(0.0, start_cell)]
        came_from: Dict[GridCell, Optional[GridCell]] = {start_cell: None}
        cost_so_far: Dict[GridCell, float] = {start_cell: 0.0}

        expansions = 0
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            expansions += 1
            if self.max_expansions and expansions > self.max_expansions:
                break

            for neighbor, step_cost in self.neighbors(current):
                if neighbor in blocked:
                    continue
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_cell)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal_cell not in came_from:
            return []

        cells = self._reconstruct_cells(came_from, goal_cell)
        cells = self._compress_collinear_cells(cells)
        points = [self.cell_to_point(cell) for cell in cells]
        points[0] = start
        points[-1] = goal
        return self._limit_waypoints(points)

    def clamp_to_field(self, point: Point2D) -> Point2D:
        """Keep requested targets inside the playable field rectangle."""

        return Point2D(
            clamp(point.x, self.x_min, self.x_max),
            clamp(point.y, self.y_min, self.y_max),
        )

    def point_to_cell(self, point: Point2D) -> GridCell:
        """Convert a field point to the nearest grid cell."""

        clamped = self.clamp_to_field(point)
        ix = int(round((clamped.x - self.x_min) / self.resolution))
        iy = int(round((clamped.y - self.y_min) / self.resolution))
        return (
            int(clamp(ix, 0, self.x_count)),
            int(clamp(iy, 0, self.y_count)),
        )

    def cell_to_point(self, cell: GridCell) -> Point2D:
        """Convert a grid cell back to the center point in field coordinates."""

        return Point2D(
            self.x_min + cell[0] * self.resolution,
            self.y_min + cell[1] * self.resolution,
        )

    def inflate_obstacles(self, obstacles: Sequence[CircularObstacle]) -> set:
        """Return the set of grid cells covered by inflated obstacles."""

        blocked = set()
        for obstacle in obstacles:
            radius_cells = int(math.ceil(obstacle.radius / self.resolution))
            center_cell = self.point_to_cell(obstacle.center)
            for ix in range(center_cell[0] - radius_cells, center_cell[0] + radius_cells + 1):
                for iy in range(center_cell[1] - radius_cells, center_cell[1] + radius_cells + 1):
                    cell = (ix, iy)
                    if not self.in_bounds(cell):
                        continue
                    point = self.cell_to_point(cell)
                    if math.hypot(point.x - obstacle.center.x, point.y - obstacle.center.y) <= obstacle.radius:
                        blocked.add(cell)
        return blocked

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell[0] <= self.x_count and 0 <= cell[1] <= self.y_count

    def neighbors(self, cell: GridCell) -> Iterable[Tuple[GridCell, float]]:
        """Yield 8-connected neighbors and their metric step cost."""

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cell[0] + dx, cell[1] + dy)
                if self.in_bounds(neighbor):
                    yield neighbor, math.hypot(dx, dy) * self.resolution

    def heuristic(self, a: GridCell, b: GridCell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution

    def _reconstruct_cells(self, came_from: Dict[GridCell, Optional[GridCell]], goal: GridCell):
        cells = [goal]
        current = goal
        while came_from[current] is not None:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        return cells

    def _compress_collinear_cells(self, cells: Sequence[GridCell]) -> List[GridCell]:
        """Drop intermediate cells that do not change path direction."""

        if len(cells) <= 2:
            return list(cells)

        compressed = [cells[0]]
        last_direction = (
            cells[1][0] - cells[0][0],
            cells[1][1] - cells[0][1],
        )
        for index in range(2, len(cells)):
            direction = (
                cells[index][0] - cells[index - 1][0],
                cells[index][1] - cells[index - 1][1],
            )
            if direction != last_direction:
                compressed.append(cells[index - 1])
                last_direction = direction
        compressed.append(cells[-1])
        return compressed

    def _limit_waypoints(self, points: Sequence[Point2D]) -> List[Point2D]:
        """Keep published paths bounded so the follower handles stale paths well."""

        if len(points) <= self.max_waypoints:
            return list(points)

        stride = int(math.ceil((len(points) - 2) / max(self.max_waypoints - 2, 1)))
        limited = [points[0]]
        limited.extend(points[index] for index in range(1, len(points) - 1, stride))
        limited.append(points[-1])
        return limited[: self.max_waypoints - 1] + [points[-1]]

    @staticmethod
    def _segment_clear(
        start: Point2D,
        goal: Point2D,
        obstacles: Sequence[CircularObstacle],
    ) -> bool:
        """Return true when the straight path to the goal avoids all obstacles."""

        dx = goal.x - start.x
        dy = goal.y - start.y
        length_sq = dx * dx + dy * dy
        for obstacle in obstacles:
            if length_sq <= 1e-12:
                closest_x = start.x
                closest_y = start.y
            else:
                t = (
                    (obstacle.center.x - start.x) * dx
                    + (obstacle.center.y - start.y) * dy
                ) / length_sq
                t = clamp(t, 0.0, 1.0)
                closest_x = start.x + t * dx
                closest_y = start.y + t * dy
            if math.hypot(closest_x - obstacle.center.x, closest_y - obstacle.center.y) <= obstacle.radius:
                return False
        return True


def make_obstacles(points: Iterable[Point2D], radius: float) -> List[CircularObstacle]:
    """Build uniform circular obstacles from obstacle center points."""

    return [CircularObstacle(point, float(radius)) for point in points]


def plan_path(start: Point2D, goal: Point2D, obstacles: Sequence[CircularObstacle], config=None):
    """Convenience function used by tests and small scripts."""

    config = config or {}
    field_config = config.get("field", {})
    path_config = config.get("path", {})
    planner = AStarPathPlanner(
        field_length=field_config.get("length", 10.0),
        field_width=field_config.get("width", 5.5),
        resolution=path_config.get("resolution", 0.1),
        boundary_margin=field_config.get("boundary_margin", 0.15),
        max_waypoints=path_config.get("max_waypoints", 160),
        max_expansions=path_config.get("max_expansions", 0),
    )
    return planner.plan(start, goal, obstacles)


def build_path_message(points: Sequence[Point2D], frame_id: str, stamp):
    """Serialize planner waypoints as nav_msgs/Path or PoseArray."""

    if NavPath is not None:
        msg = NavPath()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        poses = []
        for index, point in enumerate(points):
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = _pose_for_point(points, index)
            poses.append(pose_stamped)
        msg.poses = poses
        return msg

    msg = PoseArray()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.poses = [_pose_for_point(points, index) for index in range(len(points))]
    return msg


def path_points_from_message(msg) -> List[Point2D]:
    """Deserialize either nav_msgs/Path or PoseArray into ``Point2D`` values."""

    points = []
    for item in msg.poses:
        pose = item.pose if hasattr(item, "pose") else item
        points.append(Point2D(pose.position.x, pose.position.y))
    return points


def _pose_for_point(points: Sequence[Point2D], index: int) -> Pose:
    pose = Pose()
    point = points[index]
    pose.position.x = point.x
    pose.position.y = point.y
    if len(points) > 1:
        next_index = min(index + 1, len(points) - 1)
        previous_index = max(index - 1, 0)
        target = points[next_index] if next_index != index else points[previous_index]
        yaw = yaw_to_point(point, target)
    else:
        yaw = 0.0
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


class PathPlannerNode(Node):
    """ROS adapter that plans striker paths from world-state topics."""

    def __init__(self, config, name="path_planner"):
        super().__init__(name)
        self.config = config
        self.frame_id = config.get("frames", {}).get("field", "field_red")
        field_config = config.get("field", {})
        path_config = config.get("path", {})
        self.obstacle_radius = float(path_config.get("obstacle_radius", 0.35))
        self.planner = AStarPathPlanner(
            field_length=field_config.get("length", 10.0),
            field_width=field_config.get("width", 5.5),
            resolution=path_config.get("resolution", 0.1),
            boundary_margin=field_config.get("boundary_margin", 0.15),
            max_waypoints=path_config.get("max_waypoints", 160),
            max_expansions=path_config.get("max_expansions", 0),
        )
        default_delta = max(self.planner.resolution, 0.05)
        self.replan_min_start_delta = float(path_config.get("replan_min_start_delta", default_delta))
        self.replan_min_goal_delta = float(path_config.get("replan_min_goal_delta", default_delta))
        self.replan_min_obstacle_delta = float(
            path_config.get("replan_min_obstacle_delta", default_delta)
        )
        self.replan_max_interval = float(path_config.get("replan_max_interval", 0.35))
        self.fallback_direct_on_failure = bool(
            path_config.get("fallback_direct_on_failure", False)
        )
        self.self_pose: Optional[PoseStamped] = None
        self.target_pose: Optional[PoseStamped] = None
        self.obstacle_poses: Dict[str, PoseStamped] = {}
        self._last_warning_time = 0.0
        self._last_plan_inputs = None
        self._last_plan_time = 0.0

        self.path_pub = self.create_publisher(
            PATH_MESSAGE_TYPE,
            soccer_topic(config, "striker", "path"),
            10,
        )
        self.target_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "striker", "target"),
            self._target_callback,
            10,
        )
        self.self_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "self"),
            self._self_callback,
            10,
        )
        self.obstacle_subs = [
            self.create_subscription(
                PoseStamped,
                soccer_topic(config, "world", object_name),
                self._make_obstacle_callback(object_name),
                10,
            )
            for object_name in ("teammate", "opponent_1", "opponent_2")
        ]

        rate = float(path_config.get("replan_rate_hz", 5.0))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_callback)

    def _self_callback(self, msg):
        self.self_pose = msg

    def _target_callback(self, msg):
        self.target_pose = msg

    def _make_obstacle_callback(self, object_name):
        def callback(msg):
            self.obstacle_poses[object_name] = msg

        return callback

    def _timer_callback(self):
        if self.self_pose is None or self.target_pose is None:
            return

        start = Point2D(
            self.self_pose.pose.position.x,
            self.self_pose.pose.position.y,
        )
        goal = Point2D(
            self.target_pose.pose.position.x,
            self.target_pose.pose.position.y,
        )
        obstacle_points = tuple(
            (
                name,
                Point2D(pose.pose.position.x, pose.pose.position.y),
            )
            for name, pose in sorted(self.obstacle_poses.items())
        )
        now = self.get_clock().now().nanoseconds * 1e-9
        inputs = (start, goal, obstacle_points)
        if not self._should_replan(inputs, now):
            return
        self._last_plan_inputs = inputs
        self._last_plan_time = now

        obstacles = make_obstacles(
            (point for _, point in obstacle_points),
            self.obstacle_radius,
        )
        points = self.planner.plan(start, goal, obstacles)
        if not points:
            self._warn_throttled("A* could not find a striker path")
            self._publish_path([start, goal] if self.fallback_direct_on_failure else [])
            return
        self._publish_path(points)

    def _should_replan(self, inputs, now: float) -> bool:
        if self._last_plan_inputs is None:
            return True
        if now - self._last_plan_time >= self.replan_max_interval:
            return True

        last_start, last_goal, last_obstacles = self._last_plan_inputs
        start, goal, obstacles = inputs
        if self._distance(last_start, start) >= self.replan_min_start_delta:
            return True
        if self._distance(last_goal, goal) >= self.replan_min_goal_delta:
            return True
        if len(last_obstacles) != len(obstacles):
            return True
        for (last_name, last_point), (name, point) in zip(last_obstacles, obstacles):
            if last_name != name:
                return True
            if self._distance(last_point, point) >= self.replan_min_obstacle_delta:
                return True
        return False

    @staticmethod
    def _distance(a: Point2D, b: Point2D) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _publish_path(self, points: Sequence[Point2D]):
        stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(build_path_message(points, self.frame_id, stamp))

    def _warn_throttled(self, message: str):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_warning_time > 2.0:
            self.get_logger().warning(message)
            self._last_warning_time = now
