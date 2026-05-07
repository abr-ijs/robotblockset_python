# Calibration

This module supports both:
- **Intrinsic calibration** (camera matrix + distortion)
- **Extrinsic (hand-eye) calibration** (camera pose relative to robot)

It is based on:
- `robotblockset/cameras/collect_calibration_data.py`
- `robotblockset/cameras/calibration_boards.py`
- `robotblockset/cameras/camera_calibration.py`

For board-specific intrinsic examples, use:
- `robotblockset/tutorials/tutorial_calibrate_camera_charuco.ipynb`
- `robotblockset/tutorials/tutorial_calibrate_camera_checker.ipynb`

## 1. Calibration modes (extrinsics)

Extrinsic calibration supports two setups:
- `eye_in_hand`: camera is mounted on robot TCP/wrist, estimate camera pose in TCP frame.
- `eye_to_hand`: camera is fixed in workspace, estimate camera pose in robot base frame.

## 2. Data collection (manual)

A calibration dataset is stored in a `calibration_dir` and contains:
- board images (`image_XXXX.png`)
- synchronized robot TCP poses (`tcp_pose_XXXX.json`)
- camera intrinsics snapshot (`intrinsics.json`)

### 2.1 Interactive manual collection

Use:
- `manually_collect_calibration_data(robot, camera, calibration_dir)`

Behavior:
- puts robot in teach/freedrive mode
- shows live camera image with ChArUco detection overlay
- `s` saves one synchronized `(image, tcp pose)` sample
- `q` exits collection

```python
from robotblockset.cameras.collect_calibration_data import manually_collect_calibration_data

calibration_dir = r"D:/calib/session_01"
manually_collect_calibration_data(robot, camera, calibration_dir)
```

### 2.2 Manual/Scripted single-sample capture helpers

If you want custom collection logic, use:
- `create_data_dir(calibration_dir)`
- `prepare_collect_calibration_data(camera, calibration_dir)`
- `save_calibration_sample(sample_index, robot, camera, data_dir)`

```python
from robotblockset.cameras.collect_calibration_data import (
    create_data_dir,
    prepare_collect_calibration_data,
    save_calibration_sample,
)

data_dir = create_data_dir(r"D:/calib/session_02")
prepare_collect_calibration_data(camera, r"D:/calib/session_02")

for i in range(20):
    # move robot to a new, diverse pose where board is clearly visible
    save_calibration_sample(i, robot, camera, data_dir)
```

## 3. Intrinsic calibration

Use the tutorials for full step-by-step procedure and good pose sampling strategy:
- `robotblockset/tutorials/tutorial_calibrate_camera_charuco.ipynb`
- `robotblockset/tutorials/tutorial_calibrate_camera_checker.ipynb`

### 3.1 ChArUco intrinsic calibration

```python
from robotblockset.cameras.calibration_boards import CharucoBoard

board = CharucoBoard(
    squares_x=7,
    squares_y=5,
    square_length_m=0.040,
    marker_length_m=0.031,
)

# images: list of BGR images containing the board
res = board.intrinsic_calibration(images)

# Convert/save intrinsics
intrinsics_model = res.to_camera_intrinsics(include_distortion=True)
out_path = res.write_intrinsics_json(camera="cam0", stream="rgb", out_dir=".")
```

### 3.2 Checkerboard intrinsic calibration

```python
from robotblockset.cameras.calibration_boards import CheckerBoard

board = CheckerBoard.from_mm(cols=6, rows=4, square_length_mm=40.0)

# images: list of BGR images containing the board
res = board.intrinsic_calibration(images)
intrinsics_model = res.to_camera_intrinsics(include_distortion=True)
out_path = res.write_intrinsics_json(camera="cam0", stream="rgb", out_dir=".")
```

## 4. Extrinsic (hand-eye) calibration

### 4.1 Load collected dataset

```python
from robotblockset.cameras.camera_calibration import load_calibration_data

images, tcp_poses_in_base, intrinsics, resolution = load_calibration_data(calibration_dir)
```

### 4.2 Run calibration for all OpenCV methods

You can use either:
- board-object API from `calibration_boards.py` (recommended when using `CharucoBoard` / `CheckerBoard` objects), or
- functional API from `camera_calibration.py` (ChArUco-based)

#### A) Board-object API (`calibration_boards.py`)

```python
import os
from robotblockset.cameras.calibration_boards import CharucoBoard

board = CharucoBoard()
results_dir = os.path.join(calibration_dir, "results")
os.makedirs(results_dir, exist_ok=True)

poses, errors = board.extrinsic_calibration_all_methods(
    results_dir=results_dir,
    images=images,
    tcp_poses_in_base=tcp_poses_in_base,
    intrinsics=intrinsics,
    mode="eye_to_hand",  # or "eye_in_hand"
)
```

#### B) Functional API (`camera_calibration.py`)

```python
import os
from robotblockset.cameras.camera_calibration import extrinsic_calibration_all_methods

results_dir = os.path.join(calibration_dir, "results")
os.makedirs(results_dir, exist_ok=True)

poses, errors = extrinsic_calibration_all_methods(
    results_dir=results_dir,
    images=images,
    tcp_poses_in_base=tcp_poses_in_base,
    intrinsics=intrinsics,
    mode="eye_to_hand",  # or "eye_in_hand"
)
```

Outputs include:
- `camera_pose_<method>.json`
- `base_pose_in_camera_<method>.jpg`
- `residual_errors.json`
- `board_detections/board_detection_XXXX.jpg`

## 5. Calibration directory structure

```text
calibration_dir/
  data/
    intrinsics.json
    image_0000.png
    tcp_pose_0000.json
    image_0001.png
    tcp_pose_0001.json
    ...
  results/
    residual_errors.json
    camera_pose_*.json
    base_pose_in_camera_*.jpg
    board_detections/
      board_detection_0000.jpg
      ...
```

## 6. Practical recommendations

- Use at least 15-20 diverse samples (position + orientation changes).
- Keep the board fully visible and sharply focused.
- Avoid collecting nearly identical poses.
- For `eye_in_hand`: keep board fixed, move robot/camera.
- For `eye_to_hand`: keep camera fixed, move board with robot.
- Compare multiple OpenCV methods and inspect saved visualization images, not only residual error.

## 7. Notes

- If you are looking for `camera_calibrations.py`, in this repository the module name is `camera_calibration.py`.
- ChArUco is generally more robust than plain checkerboard in difficult viewpoints/lighting.
