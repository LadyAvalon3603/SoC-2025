"""Fixed-square Pillow Letterbox preprocessing for deployment inference."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import PadTransform, ResizeTransform, TransformList


class DocLetterboxDatasetMapper:
    """Contain-resize every source image in one fixed square Letterbox canvas."""

    def __init__(
        self,
        is_train: bool,
        image_size: int,
        image_format: str = "RGB",
        pad_value: int = 128,
    ) -> None:
        if is_train:
            raise ValueError("Deployment mapper supports inference only")
        self.image_size = int(image_size)
        if self.image_size <= 0 or self.image_size % 32:
            raise ValueError("Fixed Letterbox image_size must be a positive multiple of 32")
        self.image_format = str(image_format)
        self.pad_value = int(pad_value)

    @classmethod
    def from_config(cls, cfg, is_train: bool = False) -> Dict[str, Any]:
        return {
            "is_train": is_train,
            "image_size": cfg.INPUT.IMAGE_SIZE,
            "image_format": cfg.INPUT.FORMAT,
            "pad_value": 128,
        }

    def _geometry(self, height: int, width: int) -> Dict[str, int | float]:
        if min(height, width) <= 0:
            raise ValueError(f"Image dimensions must be positive: {(width, height)}")
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = min(self.image_size, max(1, int(round(width * scale))))
        resized_height = min(self.image_size, max(1, int(round(height * scale))))
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        return {
            "scale": float(scale),
            "left": left,
            "top": top,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "canvas_width": self.image_size,
            "canvas_height": self.image_size,
        }

    def _transforms(
        self, height: int, width: int, geometry: Dict[str, int | float]
    ) -> TransformList:
        resized_width = int(geometry["resized_width"])
        resized_height = int(geometry["resized_height"])
        left = int(geometry["left"])
        top = int(geometry["top"])
        transforms: List[Any] = [
            ResizeTransform(
                height,
                width,
                resized_height,
                resized_width,
                interp=Image.Resampling.LANCZOS,
            ),
            PadTransform(
                left,
                top,
                self.image_size - resized_width - left,
                self.image_size - resized_height - top,
                orig_w=resized_width,
                orig_h=resized_height,
                pad_value=self.pad_value,
                seg_pad_value=0,
            ),
        ]
        return TransformList(transforms)

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        image = dataset_dict.get("_preloaded_image")
        if image is None:
            record = copy.deepcopy(dataset_dict)
            image = utils.read_image(record["file_name"], format=self.image_format)
        else:
            record = copy.deepcopy(
                {
                    key: value
                    for key, value in dataset_dict.items()
                    if key != "_preloaded_image"
                }
            )
            image = np.asarray(image)
        utils.check_image_size(record, image)
        original_height, original_width = image.shape[:2]
        geometry = self._geometry(original_height, original_width)
        transforms = self._transforms(original_height, original_width, geometry)
        transformed = transforms.apply_image(image)
        expected_shape = (self.image_size, self.image_size)
        if transformed.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Letterbox produced {transformed.shape[:2]}, expected {expected_shape}"
            )

        valid_pixels = np.ones((original_height, original_width), dtype=np.uint8)
        padding_mask = ~transforms.apply_segmentation(valid_pixels).astype(bool)
        record["image"] = torch.as_tensor(
            np.ascontiguousarray(transformed.transpose(2, 0, 1))
        )
        record["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        record["letterbox_meta"] = {
            **geometry,
            "original_width": original_width,
            "original_height": original_height,
            "resize_mode": "fixed_square_letterbox",
        }
        record.pop("annotations", None)
        return record
