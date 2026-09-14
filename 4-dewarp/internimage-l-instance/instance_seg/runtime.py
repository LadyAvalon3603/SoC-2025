"""Model loading, detection filtering and CUDA timing."""

from pathlib import Path
from detectron2.modeling import build_model
import torch


def filter_and_sort_instances(instances, score_threshold: float):
    """Remove weak detections and keep a deterministic confidence order."""
    keep = instances.scores >= float(score_threshold)
    instances = instances[keep]
    order = torch.argsort(instances.scores, descending=True)
    return instances[order]


def select_largest_visible_instance(instances):
    """Keep only the prediction with the largest visible mask area.

    Masks have already been restored to the source image size at this point,
    so counting foreground pixels compares candidates in one coordinate space.
    The confidence order from ``filter_and_sort_instances`` breaks exact area
    ties deterministically.
    """
    if len(instances) <= 1:
        return instances
    visible_areas = instances.pred_masks.flatten(1).sum(dim=1)
    largest_index = int(torch.argmax(visible_areas).item())
    return instances[largest_index : largest_index + 1]


def synchronize_cuda() -> None:
    """Synchronize only when CUDA is active so timing includes GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(cfg, checkpoint_path: Path):
    model = build_model(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.eval()
