"""Image transform abstractions and helpers."""

from abc import ABC
from typing import Tuple, Union, List
import cv2
import numpy as np

from robotblockset.rbs_typing import NumpyFloatImageType, NumpyIntImageType, OpenCVIntImageType

HWCImageType = Union[OpenCVIntImageType, NumpyFloatImageType, NumpyIntImageType]
"""an image with shape (H,W,C)"""

ImageShapeType = Union[Tuple[int, int, int], Tuple[int, int]]
ImagePointType = Union[Tuple[int, int], Tuple[float, float]]


class ImageTransform(ABC):
    def __init__(self, input_shape: ImageShapeType):
        self._input_shape = input_shape

    @property
    def _input_h(self) -> int:
        return self._input_shape[0]

    @property
    def _input_w(self) -> int:
        return self._input_shape[1]

    @property
    def shape(self) -> ImageShapeType:
        """The shape of the transformed image.

        Returns
        -------
            ImageShapeType: The shape of the transformed image.
        """
        raise NotImplementedError

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        """Apply the image transform to an image to get a new image.

        Parameters
        ----------
            image : HWCImageType
                The original image, it will be unaffected by the transform.

        Raises
        ------
            NotImplementedError: Subclasses must implement this method.

        Returns
        -------
            HWCImageType: The new, transformed image.
        """
        raise NotImplementedError

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        """Map a point into transformed-image coordinates.

        Transform the coordinates of a point from original image to transformed image."""
        raise NotImplementedError

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        """Map a transformed-image point back to the source.

        Transform the coordinates of a point in the transformed image back to the original image."""
        raise NotImplementedError

    def __call__(self, image: HWCImageType) -> HWCImageType:
        """Shorthand to transform an image."""
        return self.transform_image(image)


class ComposedTransform(ImageTransform):
    def __init__(self, transforms: List[ImageTransform]):
        if len(transforms) == 0:
            raise ValueError("transforms must be a non-empty list.")

        super().__init__(transforms[0]._input_shape)
        self.transforms = transforms

    @property
    def shape(self) -> ImageShapeType:
        return self.transforms[-1].shape

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        for transform in self.transforms:
            image = transform.transform_image(image)
        return image

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        for transform in self.transforms:
            point = transform.transform_point(point)
            print(point)
        return point

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        for transform in reversed(self.transforms):
            point = transform.reverse_transform_point(point)
        return point


def crop(image: HWCImageType, x: int, y: int, w: int, h: int) -> HWCImageType:
    """
    Crop a rectangular region from an image.

    Parameters
    ----------
    image : HWCImageType
        Image to crop.
    x : int
        X-coordinate of the top-left crop corner.
    y : int
        Y-coordinate of the top-left crop corner.
    w : int
        Crop width in pixels.
    h : int
        Crop height in pixels.
    """
    # Note that the first index of the array is the y-coordinate, because this indexes the rows of the image and the y-axis runs from top to bottom.
    if len(image.shape) == 2:
        return image[y : y + h, x : x + w].copy()

    return image[y : y + h, x : x + w, :].copy()


class Crop(ImageTransform):
    """"""

    def __init__(self, input_shape: ImageShapeType, x: int, y: int, w: int, h: int):
        super().__init__(input_shape)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def shape(self) -> ImageShapeType:
        if len(self._input_shape) == 2:
            return self.h, self.w

        c = self._input_shape[2]
        return self.h, self.w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        return crop(image, self.x, self.y, self.w, self.h)

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if not (x >= self.x and x < self.x + self.w):
            raise ValueError(f"x-coordinate {x} is outside of the crop range [{self.x}, {self.x + self.w})")
        if not (y >= self.y and y < self.y + self.h):
            raise ValueError(f"y-coordinate {y} is outside of the crop range [{self.y}, {self.y + self.h})")
        return x - self.x, y - self.y

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if not (x >= 0 and x < self.w):
            raise ValueError(f"x-coordinate {x} is outside of the crop range [0, {self.w})")
        if not (y >= 0 and y < self.h):
            raise ValueError(f"y-coordinate {y} is outside of the crop range [0, {self.h})")
        return x + self.x, y + self.y


class Resize(ImageTransform):
    def __init__(self, input_shape: ImageShapeType, h: int, w: int, round_transformed_points: bool = True):
        """Create a new Resize transform.

        Note: Transforming a point to or from a resized image can lead to non-integer coordinates. Pixel coordinates
            are however often expected to be integers, e.g. by the OpenCV draw functions. So by default, this class
            will round transformed points to the nearest integer. If you want to avoid the errors introduced by
            rounding, you can set `round_transformed_points` to False to get the exact transformed points as floats.

        Parameters
        ----------
            input_shape : ImageShapeType
                Shape of the images that will be resized.
            h : int
                Height of the resized image.
            w : int
                Width of the resized image.
            round_transformed_points : bool, optional
                Whether to round transformed points to the nearest integer.
        """
        super().__init__(input_shape)
        self.h = h
        self.w = w
        self.round_transformed_points = round_transformed_points

    @property
    def shape(self) -> ImageShapeType:
        if len(self._input_shape) == 2:
            return self.h, self.w

        c = self._input_shape[2]
        return self.h, self.w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        return cv2.resize(image, (self.w, self.h))

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if not (x >= 0 and x < self._input_w):
            raise ValueError(f"x-coordinate {x} is outside of the input image range [0, {self._input_w})")
        if not (y >= 0 and y < self._input_h):
            raise ValueError(f"y-coordinate {y} is outside of the input image range [0, {self._input_h})")

        w_scale = self.w / self._input_w
        h_scale = self.h / self._input_h

        x_float = w_scale * x
        y_float = h_scale * y

        if self.round_transformed_points:
            return round(x_float), round(y_float)

        return x_float, y_float

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if not (x >= 0 and x < self.w):
            raise ValueError(f"x-coordinate {x} is outside of the resized image range [0, {self.w})")
        if not (y >= 0 and y < self.h):
            raise ValueError(f"y-coordinate {y} is outside of the resized image range [0, {self.h})")
        w_scale_inverse = self._input_w / self.w
        h_scale_inverse = self._input_h / self.h

        x_float = w_scale_inverse * x
        y_float = h_scale_inverse * y

        if self.round_transformed_points:
            return round(x_float), round(y_float)

        return x_float, y_float


class Rotate90(ImageTransform):
    """Rotate an image by multiples of 90 degrees."""

    def __init__(
        self,
        input_shape: ImageShapeType,
        num_rotations: int = 1,
    ):
        """Create a new Rotate transform.

        Parameters
        ----------
            num_rotations : int, optional
                the number of 90-degree rotations to apply. Positive values rotate counter-clockwise.
        """
        super().__init__(input_shape)

        if not isinstance(num_rotations, int):
            raise TypeError("num_rotations must be an int")

        self._num_rotations = num_rotations % 4

    @property
    def shape(self) -> ImageShapeType:
        if self._num_rotations % 2 == 0:
            h, w = self._input_shape[:2]
        else:
            w, h = self._input_shape[:2]

        if len(self._input_shape) == 2:
            return h, w

        c = self._input_shape[2]
        return h, w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        # The copy here ensure the result is not a view into the original image.
        return np.rot90(image, self._num_rotations).copy()

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if not (x >= 0 and x < self._input_w):
            raise ValueError(f"x-coordinate {x} is outside of the input image range [0, {self._input_w})")
        if not (y >= 0 and y < self._input_h):
            raise ValueError(f"y-coordinate {y} is outside of the input image range [0, {self._input_h})")

        if self._num_rotations == 1:
            return y, self._input_w - x - 1
        elif self._num_rotations == 2:
            return self._input_w - x - 1, self._input_h - y - 1
        elif self._num_rotations == 3:
            return self._input_h - y - 1, x
        return x, y

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if self._num_rotations == 1:
            return self._input_w - y - 1, x
        elif self._num_rotations == 2:
            return self._input_w - x - 1, self._input_h - y - 1
        elif self._num_rotations == 3:
            return y, self._input_h - x - 1
        return x, y
