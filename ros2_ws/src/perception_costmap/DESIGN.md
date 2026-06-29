# Perception → Costmap → Nav2

Design for the perception stack that turns camera + lidar into a Nav2-usable
costmap: where the road is, and where the obstacles are.

## Goal

Produce a 2D costmap, in the robot frame, where:

- **drivable road** = free (cost 0)
- **off-road** (grass, sidewalk, anything not road) = high cost / lethal
- **obstacles** (vehicles, pedestrians, cones) = lethal
- **unknown** (not yet observed) = unknown (-1)

Nav2's planner + controller then plan paths that stay on the road and avoid
obstacles. We do **not** reinvent the planner or the costmap engine — we feed
Nav2's existing `nav2_costmap_2d` through standard interfaces.

## How perception feeds Nav2 (the key decision)

Nav2 builds its costmap from stacked *layers*. We use three:

| Layer | Source | What it does |
|-------|--------|--------------|
| `StaticLayer` | our `/perception/costmap` (`nav_msgs/OccupancyGrid`) | road vs off-road + camera-seen obstacles |
| `obstacle_layer` | our `/perception/obstacle_points` (`sensor_msgs/PointCloud2`) | lidar obstacles, with raytrace clearing |
| `inflation_layer` | (built-in) | inflate lethal cells by the car's radius |

Two reasons for splitting it this way:

1. **Road/drivable area** is naturally a dense grid (every cell is road or
   not), so an `OccupancyGrid` is the right message and `StaticLayer` ingests
   it directly — no custom C++ plugin needed.
2. **Obstacles** are sparse and benefit from Nav2's native raytrace *clearing*
   (a cell stops being an obstacle once the sensor sees through it). That is
   exactly what `obstacle_layer` does when fed a `PointCloud2`. Lidar points
   are already metric, so this path is the most reliable.

This keeps everything on standard ROS2 messages, which is also what makes it
portable to the Jetson later (swap the sim sensor topics for real driver
topics; the perception node and Nav2 config are unchanged).

## Pipeline

```
camera Image ─┐
              ├─► segmentation.py ─► road mask ─┐
CameraInfo  ──┘                                 │
                                  bev.py (IPM)   ├─► occupancy.py ─► /perception/costmap
camera Image ─► obstacles.py ─► obstacle mask ──┘                    (OccupancyGrid)

lidar PointCloud2 ─► obstacles.py (ground filter + cluster) ─────► /perception/obstacle_points
                                                                    (PointCloud2)
```

- **segmentation.py** — drivable-road mask. Start classical (HSV threshold,
  largest blob — reused from Adam Castillo's `perception/costmap.py`), with a
  learned model (TwinLiteNet+) pluggable behind the same interface.
- **obstacles.py** — obstacle mask from the camera (classical contrast / YOLO)
  and obstacle points from lidar (remove ground plane, keep what's above it).
- **bev.py** — inverse-perspective mapping: warp the camera/segmentation from
  image space to a top-down metric grid using a homography. The homography
  comes from camera intrinsics + mounting (height/pitch); calibratable from a
  single frame via 4 point correspondences. In CARLA we know these exactly; on
  the car we calibrate once.
- **occupancy.py** — fuse masks into a `nav_msgs/OccupancyGrid` with correct
  metadata (resolution, origin, frame). This is the offline-testable heart.
- **costmap_node.py** — the ROS2 node wiring subscriptions → modules →
  publishers, all parameterized.

## Grid geometry

- Robot-centric, REP-103 axes: **+x forward, +y left**.
- Default: 20 m forward × 20 m wide, **0.1 m/cell** → 200 × 200 cells.
- Robot at the bottom-center of the forward range.
- `OccupancyGrid.info.origin` is set so cell→world matches the chosen extent;
  `header.frame_id = base_link` (or `ego`), republished every frame.
- Cost encoding follows ROS convention: `0` free, `100` lethal, `-1` unknown,
  intermediate values for "off-road but traversable if forced."

## Topics & frames

| Topic | Type | Dir |
|-------|------|-----|
| `/camera/image` | `sensor_msgs/Image` | in |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | in |
| `/lidar/points` | `sensor_msgs/PointCloud2` | in (optional) |
| `/perception/costmap` | `nav_msgs/OccupancyGrid` | out |
| `/perception/obstacle_points` | `sensor_msgs/PointCloud2` | out |

Frames: `map → odom → base_link → camera/lidar`. Perception publishes in
`base_link`; Nav2 handles the rest via TF.

## Sim vs car (sim-to-real)

| | Dev / CARLA (x86) | Car / Jetson (arm64) |
|---|---|---|
| camera/lidar source | CARLA sensors | real camera + lidar drivers |
| perception node | same code | same code |
| Nav2 config | same | same |
| control output | `cmd_to_carla` | real actuator/CAN node |

Only the sensor source and the actuator sink change. The perception → costmap
→ Nav2 core is identical, which is the whole point of doing it on standard
messages.

## Build order

1. ✅ this design
2. scaffold `perception_costmap` package
3. geometry core (`bev.py`, `occupancy.py`) + offline test
4. `segmentation.py`, `obstacles.py`
5. `costmap_node.py`
6. Nav2 costmap params + launch
7. validation (offline test + colcon build, then CARLA smoke test on the 5090)

## Calibration / known TODOs

- IPM homography needs real numbers from CameraInfo + camera mounting; ships
  with a flagged placeholder until calibrated.
- HSV road thresholds are lighting-dependent; the learned seg path exists to
  remove that fragility.
- Lidar ground-plane removal assumes roughly flat ground near the car.
