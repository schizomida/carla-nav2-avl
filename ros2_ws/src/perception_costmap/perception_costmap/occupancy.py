"""
occupancy.py
------------
Turn grid-space masks (road / obstacle / observed) into a ROS
``nav_msgs/OccupancyGrid``.

This is the heart of the costmap and is deliberately ROS-free at its core so
it can be unit-tested without a running ROS graph. ``build_cost_array`` is pure
numpy; ``to_occupancy_grid_msg`` imports ROS message types lazily.

Conventions
-----------
- Robot-centric grid, REP-103 axes: +x forward, +y left.
- The grid array is shaped (height, width) = (rows along y, cols along x),
  matching ``OccupancyGrid`` semantics where index = row * width + col,
  column indexes world x and row indexes world y.
- Cost values follow the ROS convention:
      -1  unknown (not observed yet)
       0  free / drivable road
     100  lethal (obstacle, or off-road if off-road is treated as lethal)
"""

from dataclasses import dataclass
import numpy as np

UNKNOWN = -1
FREE = 0
LETHAL = 100


@dataclass(frozen=True)
class GridSpec:
    """Metric extent + resolution of the costmap, in the robot frame."""
    x_min: float = -4.0      # metres behind the robot
    x_max: float = 16.0      # metres ahead of the robot
    y_min: float = -10.0     # metres to the right (+y is left)
    y_max: float = 10.0      # metres to the left
    resolution: float = 0.1  # metres per cell
    frame_id: str = "base_link"

    @property
    def width(self) -> int:          # cells along x (forward)
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:         # cells along y (left)
        return int(round((self.y_max - self.y_min) / self.resolution))

    def world_to_cell(self, x: float, y: float):
        """World (x,y) in metres -> (col, row), or None if outside the grid."""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        if 0 <= col < self.width and 0 <= row < self.height:
            return col, row
        return None

    def cell_to_world(self, col: int, row: int):
        """(col,row) -> world (x,y) at the cell centre, in metres."""
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y


def build_cost_array(grid: GridSpec,
                     road_mask: np.ndarray,
                     obstacle_mask: np.ndarray,
                     known_mask: np.ndarray = None,
                     offroad_cost: int = LETHAL) -> np.ndarray:
    """
    Fuse grid-space boolean masks into an int8 cost array (height, width).

    Priority (low -> high): unknown < off-road < road < obstacle.
      - cells not in ``known_mask``      -> UNKNOWN (-1)
      - known cells                      -> ``offroad_cost`` (default lethal)
      - road cells (within known)        -> FREE (0)
      - obstacle cells                   -> LETHAL (100)

    Road is clipped to ``known_mask``: a "road" pixel outside the observed
    footprint (e.g. a mirror-projected sky pixel from a side camera) must
    never mark unobserved ground drivable. Obstacles are NOT clipped --
    the lidar legitimately sees beyond the camera FOVs, and a spurious
    obstacle is the safe direction.

    All masks must be shape (grid.height, grid.width).
    """
    shape = (grid.height, grid.width)
    for name, m in (("road_mask", road_mask), ("obstacle_mask", obstacle_mask)):
        if m.shape != shape:
            raise ValueError(f"{name} shape {m.shape} != grid {shape}")

    if known_mask is None:
        known_mask = np.ones(shape, dtype=bool)

    cost = np.full(shape, UNKNOWN, dtype=np.int8)
    cost[known_mask] = np.int8(offroad_cost)   # observed but not road -> off-road
    cost[road_mask.astype(bool) & known_mask] = FREE   # road overrides off-road
    cost[obstacle_mask.astype(bool)] = LETHAL  # obstacle overrides everything
    return cost


def to_occupancy_grid_msg(cost: np.ndarray, grid: GridSpec, stamp=None,
                          frame_id: str = None):
    """
    Wrap a cost array in a ``nav_msgs/OccupancyGrid``. ROS imports are lazy so
    the rest of this module stays usable (and testable) without ROS installed.
    """
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import Pose

    msg = OccupancyGrid()
    msg.header.frame_id = frame_id or grid.frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution)
    msg.info.width = grid.width
    msg.info.height = grid.height
    origin = Pose()
    origin.position.x = float(grid.x_min)
    origin.position.y = float(grid.y_min)
    origin.orientation.w = 1.0
    msg.info.origin = origin
    # OccupancyGrid is row-major (index = row*width + col); numpy C-order
    # flatten of a (height, width) array gives exactly that.
    msg.data = cost.astype(np.int8).flatten().tolist()
    return msg
