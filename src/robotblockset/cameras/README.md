# RBS Cameras and Utilities

This package provides RobotBlockSet camera interfaces and utilities for RGB and RGB-D cameras, image conversion, point-cloud handling, calibration boards, and hand-eye calibration workflows.

The implementation is based in part on [airo-camera-toolkit](https://github.com/airo-ugent/airo-mono/tree/main/airo-camera-toolkit), but adapted for RobotBlockSet robot integration and SE(3) pose handling.

## Installation

The base package already includes the hardware-independent utilities in this subpackage, such as image conversion, projection, and calibration helpers.

For camera calibration and hand-eye workflows, install the camera extras:

```bash
pip install .[cameras]
```

Backend-specific dependencies depend on the camera hardware you use. The main entry points for those integrations are:

- [`zed.py`](./zed.py)
- [`realsense.py`](./realsense.py)
- [`basler.py`](./basler.py)

The original `airo-camera-toolkit` installation notes are still useful as background references:

- [ZED Installation](airo_camera_toolkit/cameras/zed/installation.md)
- [RealSense Installation](airo_camera_toolkit/cameras/realsense/realsense_installation.md)

## Getting Started

Access a camera by instantiating the corresponding class. For example, for a ZED camera:

```python
from robotblockset.cameras.zed import Zed
from robotblockset.cameras.image_converter import ImageConverter
import cv2

camera = Zed(Zed.RESOLUTION_720, fps=30)

while True:
    image_rgb_float = camera.get_rgb_image()
    image_bgr = ImageConverter.from_numpy_format(image_rgb_float).image_in_opencv_format
    cv2.imshow("Image", image_bgr)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
```

## Hand-Eye Calibration

Use the utilities in this package to estimate the pose of a camera relative to a robot. The main workflow lives in [`hand_eye_calibration.py`](./hand_eye_calibration.py)
and uses calibration boards together with RobotBlockSet robot interfaces.

## Utilities

### Image Format Conversion

By default, cameras return NumPy 32-bit float RGB images with values in the range `[0, 1]` through `get_rgb_image()`. This is convenient for downstream processing, for example with neural networks. For higher performance, 8-bit unsigned integer RGB images are also available through `get_rgb_image_as_int()`.

When using OpenCV, conversion to BGR format is usually required. For this you can use the `ImageConverter` class:

```python
from robotblockset.cameras.image_converter import ImageConverter

image_rgb_int = camera.get_rgb_image_as_int()
image_bgr = ImageConverter.from_numpy_int_format(image_rgb_int).image_in_opencv_format
```

### Annotation and Interactive Tools

For annotation and interactive workflows, inspect the utilities in this package directly and the example notebooks under `robotblockset/tutorials` and `robotblockset/examples`. 

## Image Transforms

See `image_transform.py`and the `tutorial_image_transform.ipynb` notebook for more details.

## Real-Time Visualization

For real-time robotics visualization, using [rerun.io](https://www.rerun.io/) is recommended instead of building custom OpenCV or Qt viewers. No special RBS wrapper is required.

## References

For more background on cameras, especially intrinsics, extrinsics, distortion coefficients, and camera models, see:

- Szeliski, *Computer Vision: Algorithms and Applications*:
  https://szeliski.org/Book/
- https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/schedule.html
- https://learnopencv.com/geometry-of-image-formation/
- http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
