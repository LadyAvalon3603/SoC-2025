"""Single-pass batch inference, output writing and timing summary."""

from typing import List
import resource
import time

import cv2
from detectron2.data import detection_utils
import numpy as np
import torch

from .config import load_config
from .geometry import mask_bbox
from .io import (
    input_images,
    output_file,
    output_key,
    remove_previous_instance_artifacts,
    write_image,
)
from .preprocessing import DocLetterboxDatasetMapper
from .runtime import (
    filter_and_sort_instances,
    filter_instances_by_mask_area,
    keep_largest_mask_component,
    load_model,
    select_largest_visible_instance,
    synchronize_cuda,
)
from .visualization import COLORS, render_mask


def run(args) -> None:
    use_amp = bool(args.use_amp and torch.cuda.is_available())
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    image_paths = list(input_images(input_path))
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No supported images found under: {input_path}")

    opts = list(args.opts) + [
        "MODEL.WEIGHTS",
        str(checkpoint_path),
        "TEST.DETECTIONS_PER_IMAGE",
        str(args.max_detections),
    ]
    cfg = load_config(args.config_file, opts)
    if cfg.INPUT.IMAGE_SIZE is None:
        raise ValueError("INPUT.IMAGE_SIZE must be set in the deployment YAML")
    input_size = int(cfg.INPUT.IMAGE_SIZE)
    if input_size <= 0 or input_size % 32 != 0:
        raise ValueError(
            "INPUT.IMAGE_SIZE must be a positive multiple of 32, "
            f"got {input_size}"
        )

    print("Inference settings:")
    print(f"  input: {input_path}")
    print(f"  images: {len(image_paths)}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  preprocessing: fixed {input_size}x{input_size} Letterbox")
    print(f"  score threshold: {args.score_threshold}")
    print(f"  max detections: {args.max_detections}")
    print("  instance cleanup: keep largest connected component")
    print(f"  minimum mask area: > {args.min_mask_area} original-image pixels")
    print("  final selection: largest visible mask (one document)")
    print(f"  AMP/FP16: {use_amp}")
    print(f"  output mode: {args.output_mode}")
    print(f"  output: {output_dir}")

    process_wall_start = time.perf_counter()
    model_load_start = time.perf_counter()
    model = load_model(cfg, checkpoint_path)
    synchronize_cuda()
    model_load_seconds = time.perf_counter() - model_load_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mapper = DocLetterboxDatasetMapper(
        **DocLetterboxDatasetMapper.from_config(cfg, is_train=False)
    )
    output_names = ("overlays",) if args.output_mode == "debug" else ("instances",)
    directories = {name: output_dir / name for name in output_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    image_seconds: List[float] = []
    preprocess_seconds: List[float] = []
    inference_seconds: List[float] = []

    for image_index, image_path in enumerate(image_paths, start=1):
        image_start = time.perf_counter()
        preprocess_start = image_start
        image_for_model = detection_utils.read_image(
            str(image_path), format=cfg.INPUT.FORMAT
        )
        if cfg.INPUT.FORMAT == "RGB":
            image_bgr = cv2.cvtColor(image_for_model, cv2.COLOR_RGB2BGR)
        elif cfg.INPUT.FORMAT == "BGR":
            image_bgr = np.ascontiguousarray(image_for_model)
        else:
            raise ValueError(
                f"Standalone inference requires RGB or BGR input, got "
                f"{cfg.INPUT.FORMAT!r}"
            )
        height, width = image_for_model.shape[:2]
        record = mapper(
            {
                "file_name": str(image_path.resolve()),
                "height": height,
                "width": width,
                "image_id": 0,
                "_preloaded_image": image_for_model,
            }
        )
        preprocess_seconds.append(time.perf_counter() - preprocess_start)

        synchronize_cuda()
        inference_start = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=use_amp, cache_enabled=False
        ):
            instances = model([record])[0]["instances"].to("cpu")
        synchronize_cuda()
        inference_seconds.append(time.perf_counter() - inference_start)

        instances = filter_and_sort_instances(instances, args.score_threshold)
        score_filtered_count = len(instances)
        instances = keep_largest_mask_component(instances)
        instances = filter_instances_by_mask_area(instances, args.min_mask_area)
        candidate_count = len(instances)
        instances = select_largest_visible_instance(instances)

        overlay = image_bgr.copy() if args.output_mode == "debug" else None
        key = output_key(image_path, input_path)
        if "instances" in directories:
            remove_previous_instance_artifacts(directories["instances"], key, ".png")

        kept_count = 0
        for zero_index, (mask_tensor, box_tensor, score_tensor) in enumerate(
            zip(instances.pred_masks, instances.pred_boxes.tensor, instances.scores)
        ):
            instance_id = zero_index + 1
            mask = mask_tensor.numpy().astype(bool)
            model_bbox = [float(value) for value in box_tensor.tolist()]
            score = float(score_tensor)
            color = COLORS[zero_index % len(COLORS)]
            tight_bbox = mask_bbox(mask)
            bbox = (
                [float(value) for value in tight_bbox]
                if tight_bbox is not None
                else model_bbox
            )

            if overlay is not None:
                render_mask(overlay, mask, bbox, score, instance_id, color)
            if args.output_mode == "production":
                mask_path = output_file(
                    directories["instances"],
                    key,
                    f"_instance_{instance_id:03d}.png",
                )
                write_image(mask_path, mask.astype(np.uint8) * 255)
            kept_count += 1

        if overlay is not None:
            overlay_path = output_file(directories["overlays"], key, ".jpg")
            write_image(overlay_path, overlay)
        image_seconds.append(time.perf_counter() - image_start)
        print(
            f"[{image_index}/{len(image_paths)}] {image_path.name}: "
            f"{score_filtered_count} score-filtered candidate(s), "
            f"{candidate_count} with mask area > {args.min_mask_area}, "
            f"kept {kept_count}"
        )

    print("Done.")
    if "overlays" in directories:
        print(f"  Final overlays: {directories['overlays']}")
    if "instances" in directories:
        print(f"  Instance masks: {directories['instances']}")

    process_wall_seconds = time.perf_counter() - process_wall_start
    image_array = np.asarray(image_seconds, dtype=np.float64)
    inference_total = float(np.sum(inference_seconds))
    preprocess_total = float(np.sum(preprocess_seconds))
    image_total = float(np.sum(image_array))
    other_total = max(0.0, image_total - preprocess_total - inference_total)
    print("Timing summary:")
    print(f"  Model load:       {model_load_seconds:.3f} s")
    print(f"  Image processing: {image_total:.3f} s")
    print(f"  Total wall time:  {process_wall_seconds:.3f} s")
    print(f"  Per image mean:   {float(np.mean(image_array)):.3f} s")
    print(f"  Per image P50:    {float(np.percentile(image_array, 50)):.3f} s")
    print(f"  Per image P95:    {float(np.percentile(image_array, 95)):.3f} s")
    print(f"  Throughput:       {len(image_paths) / image_total:.3f} image/s")
    print(f"  Preprocess total: {preprocess_total:.3f} s")
    print(f"  Inference total:  {inference_total:.3f} s")
    print(f"  Other/write:      {other_total:.3f} s")
    process_peak_rss_gib = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    )
    print(f"  Peak process RSS:   {process_peak_rss_gib:.3f} GiB")
    if torch.cuda.is_available():
        allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"  Peak GPU allocated: {allocated_gib:.3f} GiB")
        print(f"  Peak GPU reserved:  {reserved_gib:.3f} GiB")
