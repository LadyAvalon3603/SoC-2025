"""Deployment contracts exercised without loading GPU model weights."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
from detectron2.structures import Boxes, Instances
import numpy as np
import torch

from instance_seg.cli import parse_args
from instance_seg.config import load_config
from instance_seg.io import input_images
from instance_seg.preprocessing import DocLetterboxDatasetMapper
from instance_seg.runtime import (
    filter_instances_by_mask_area,
    keep_largest_mask_component,
    select_largest_visible_instance,
)
from instance_seg.settings import DEFAULT_INPUT, PACKAGE_ROOT


class DeploymentTests(unittest.TestCase):
    def test_fixed_square_letterbox_uses_configured_size(self):
        image_size = int(load_config().INPUT.IMAGE_SIZE)
        for shape in [(312, 421), (421, 312)]:
            with self.subTest(shape=shape):
                height, width = shape
                image = np.zeros((height, width, 3), dtype=np.uint8)
                scale = min(image_size / width, image_size / height)
                expected_height = int(round(height * scale))
                expected_width = int(round(width * scale))
                expected_top = (image_size - expected_height) // 2
                expected_left = (image_size - expected_width) // 2
                mapper = DocLetterboxDatasetMapper(
                    is_train=False,
                    image_size=image_size,
                    image_format="RGB",
                )
                record = mapper(
                    {
                        "file_name": "in-memory",
                        "height": height,
                        "width": width,
                        "image_id": 7,
                        "_preloaded_image": image,
                    }
                )
                self.assertEqual(
                    tuple(record["image"].shape),
                    (3, image_size, image_size),
                )
                self.assertEqual(
                    tuple(record["padding_mask"].shape),
                    (image_size, image_size),
                )
                meta = record["letterbox_meta"]
                self.assertEqual(
                    (meta["resized_height"], meta["resized_width"]),
                    (expected_height, expected_width),
                )
                self.assertEqual(
                    (meta["top"], meta["left"]),
                    (expected_top, expected_left),
                )
                self.assertEqual(
                    (meta["canvas_height"], meta["canvas_width"]),
                    (image_size, image_size),
                )
                self.assertEqual(meta["resize_mode"], "fixed_square_letterbox")

    def test_cli_is_independent_of_cwd_and_model_imports(self):
        self.assertEqual(parse_args([]).input, DEFAULT_INPUT)
        self.assertEqual(
            parse_args([], output_mode="production").output_mode, "production"
        )
        self.assertEqual(
            parse_args(["--output-mode", "debug"], output_mode="production").output_mode,
            "debug",
        )
        self.assertEqual(DEFAULT_INPUT, PACKAGE_ROOT.parent / "TEST_dewarp")
        with tempfile.TemporaryDirectory() as folder:
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from instance_seg.cli import parse_args; parse_args([]); "
                'assert "torch" not in sys.modules; '
                'assert "detectron2" not in sys.modules'
            )
            subprocess.run(
                [sys.executable, "-c", code, str(PACKAGE_ROOT)],
                check=True,
                cwd=folder,
            )

    def test_removed_roi_arguments_are_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--roi-refine"])

    def test_invalid_arguments_fail_before_model_loading(self):
        for args in [
            ["--score-threshold", "nan"],
            ["--score-threshold", "1.1"],
            ["--max-images", "-1"],
            ["--max-detections", "0"],
            ["--min-mask-area", "-1"],
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                parse_args(args)

    def test_recursive_discovery_ignores_directories_with_image_extensions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "folder.jpg").mkdir()
            image = root / "folder.jpg" / "scan.PNG"
            image.touch()
            (root / "notes.txt").touch()
            self.assertEqual(list(input_images(root)), [image])

    def test_small_masks_are_filtered_before_largest_selection(self):
        instances = Instances((30, 40))
        masks = torch.zeros((3, 30, 40), dtype=torch.bool)
        masks[0, 0:6, 0:10] = True
        masks[0, 20:26, 30:40] = True
        masks[1, 2:18, 3:27] = True
        masks[1, 25:30, 35:40] = True
        masks[2, 5:15, 5:15] = True
        instances.pred_masks = masks
        instances.pred_boxes = Boxes(
            torch.tensor(
                [
                    [0.0, 0.0, 40.0, 26.0],
                    [3.0, 2.0, 40.0, 30.0],
                    [5.0, 5.0, 15.0, 15.0],
                ]
            )
        )
        instances.scores = torch.tensor([0.99, 0.70, 0.80])

        candidates = keep_largest_mask_component(instances)
        candidates = filter_instances_by_mask_area(candidates, 100)
        selected = select_largest_visible_instance(candidates)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected.pred_masks.sum()), 16 * 24)
        self.assertTrue(torch.isclose(selected.scores[0], torch.tensor(0.70)))

    def test_batch_outputs_keep_nested_names(self):
        from instance_seg import pipeline

        class FakeModel:
            def __call__(self, records):
                record = records[0]
                height, width = record["height"], record["width"]
                instances = Instances((height, width))
                masks = torch.zeros((3, height, width), dtype=torch.bool)
                masks[0, 2:14, 2:14] = True
                masks[1, 8:24, 12:36] = True
                masks[2, 10:22, 5:17] = True
                instances.pred_masks = masks
                instances.pred_boxes = Boxes(
                    torch.tensor(
                        [
                            [2.0, 2.0, 14.0, 14.0],
                            [12.0, 8.0, 36.0, 24.0],
                            [5.0, 10.0, 17.0, 22.0],
                        ]
                    )
                )
                instances.scores = torch.tensor([0.99, 0.70, 0.80])
                return [{"instances": instances}]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "input"
            for nested in ["a", "b"]:
                (inputs / nested).mkdir(parents=True)
                cv2.imwrite(
                    str(inputs / nested / "same.jpg"),
                    np.full((32, 48, 3), 128, np.uint8),
                )
            checkpoint = root / "placeholder.pth"
            checkpoint.touch()

            for mode in ("production", "debug"):
                with self.subTest(mode=mode):
                    output = root / mode
                    if mode == "production":
                        stale = output / "instances" / "a" / "same_instance_002.png"
                        stale.parent.mkdir(parents=True)
                        stale.touch()
                    args = parse_args(
                        [
                            str(inputs),
                            "--checkpoint",
                            str(checkpoint),
                            "--output-dir",
                            str(output),
                            "--output-mode",
                            mode,
                            "--opts",
                            "MODEL.DEVICE",
                            "cpu",
                        ]
                    )
                    with patch.object(
                        pipeline, "load_model", lambda *args: FakeModel()
                    ), patch.object(torch.cuda, "is_available", lambda: False):
                        pipeline.run(args)

                    for nested in ["a", "b"]:
                        overlay = output / "overlays" / nested / "same.jpg"
                        mask = output / "instances" / nested / "same_instance_001.png"
                        self.assertEqual(overlay.exists(), mode == "debug")
                        self.assertEqual(mask.exists(), mode == "production")
                        if mask.exists():
                            pixels = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                            self.assertEqual(pixels.shape, (32, 48))
                            self.assertEqual(int((pixels == 255).sum()), 16 * 24)
                        if overlay.exists():
                            pixels = cv2.imread(str(overlay))
                            self.assertEqual(pixels.shape, (32, 48, 3))
                            self.assertFalse(np.all(pixels == 128))
                    if mode == "production":
                        self.assertFalse(
                            (output / "instances" / "a" / "same_instance_002.png").exists()
                        )


if __name__ == "__main__":
    unittest.main()
