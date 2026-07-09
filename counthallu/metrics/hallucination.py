"""Counting-hallucination measurement for generated images.

``CountHalluQuantifier`` classifies each generated image as
  * a *visual failure* (too broken to count, detected by a quality classifier),
  * a *counting hallucination* (countable but with a wrong instance count), or
  * *counting-correct*.

Two counting backends are supported:
  * "regression": a ``CountingRegressor`` that predicts per-class instance
    counts (ToyShape / SimObject).
  * "fingers_detection": a YOLO finger detector plus a quality classifier
    (RealHand); an image is correct iff exactly the reference number of
    fingers is detected.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from counthallu.metrics.fid import calculate_fid_given_paths


class CountHalluQuantifier:
    def __init__(self, counting_model, counting_model_type, device="cuda",
                 target_class_indices_list=None, reference_counts_list=None,
                 quality_cls_model=None, save_detection_results=False):
        """
        Args:
            counting_model: counting network (regressor or YOLO detector).
            counting_model_type: "regression" or "fingers_detection".
            reference_counts_list: acceptable ground-truth counts, e.g. [5]
                (required for detection).
            quality_cls_model: binary good/bad-quality classifier (required
                for detection; broken images are excluded from counting).
            target_class_indices_list: detector class indices to count, e.g. [0].
            save_detection_results: also save YOLO visualizations next to the
                generated images.
        """
        self.counting_model = counting_model.to(device)
        self.counting_model.eval()
        self.counting_model_type = counting_model_type
        self.device = device
        self.reference_counts_list = reference_counts_list
        self.target_class_indices_list = target_class_indices_list
        self.quality_cls_model = quality_cls_model
        self.save_detection_results = save_detection_results

        if counting_model_type == "fingers_detection":
            if reference_counts_list is None:
                raise ValueError("reference_counts_list is required for fingers_detection.")
            if target_class_indices_list is None:
                raise ValueError("target_class_indices_list is required for fingers_detection.")
            if quality_cls_model is None:
                raise ValueError("quality_cls_model is required for fingers_detection.")
            self.quality_cls_model = quality_cls_model.to(device)
            self.quality_cls_model.eval()

    @torch.no_grad()
    def __call__(self, img_path, global_indices):
        """Score the images ``<img_path>/<idx>.png`` for idx in global_indices.

        Returns (hallu_indices, low_quality_indices,
                 hallu_pred_counts, correct_pred_counts).
        """
        if self.counting_model_type == "regression":
            return self._score_regression(img_path, global_indices)
        if self.counting_model_type == "fingers_detection":
            return self._score_fingers_detection(img_path, global_indices)
        raise ValueError(f"Unknown counting_model_type: {self.counting_model_type}")

    def _score_regression(self, img_path, global_indices):
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        _, batch = load_generated_images(img_path, global_indices, transform)
        preds = torch.round(self.counting_model(batch.to(self.device)))

        # Valid training images contain at most one instance per class and at
        # least one shape overall; anything else is a counting hallucination.
        hallu_mask = torch.any(preds >= 2.0, dim=1) | torch.all(preds == 0.0, dim=1)
        hallu_idx = torch.nonzero(hallu_mask, as_tuple=True)[0]
        correct_idx = torch.nonzero(~hallu_mask, as_tuple=True)[0]

        pred_counts = preds.long().tolist()
        return (
            [global_indices[i] for i in hallu_idx],
            [],  # no separate visual-failure category for regression datasets
            [pred_counts[i] for i in hallu_idx],
            [pred_counts[i] for i in correct_idx],
        )

    def _score_fingers_detection(self, img_path, global_indices):
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        paths, batch = load_generated_images(img_path, global_indices, transform)

        # 1) Quality gate: only clean images proceed to finger counting.
        probs = F.softmax(self.quality_cls_model(batch.to(self.device)), dim=1)
        is_good = (probs[:, 1] >= 0.5)
        lq_indices = [global_indices[i] for i in torch.where(~is_good)[0]]
        hq_indices = [global_indices[i] for i in torch.where(is_good)[0]]
        hq_paths = [p for p, good in zip(paths, is_good) if good]

        if not hq_paths:
            return [], lq_indices, [], []

        # 2) Count fingers with YOLO on the clean subset.
        yolo_results = self.counting_model.predict(
            source=hq_paths, imgsz=640, conf=0.25, iou=0.1,
            save=False, save_txt=False, device=self.device,
        )
        detected_classes = [r.boxes.cls for r in yolo_results]
        correct_mask = get_count_correct_batch_mask(
            detected_classes, self.target_class_indices_list, self.reference_counts_list
        )

        pred_counts = [len(cls) for cls in detected_classes]
        hallu_indices = [idx for idx, ok in zip(hq_indices, correct_mask) if not ok]
        hallu_counts = [c for c, ok in zip(pred_counts, correct_mask) if not ok]
        correct_counts = [c for c, ok in zip(pred_counts, correct_mask) if ok]

        if self.save_detection_results:
            save_images_with_detection_labels(
                img_path.replace("gen_images", "yolo_detection_results"),
                hq_indices, hallu_indices, yolo_results,
            )
        return hallu_indices, lq_indices, hallu_counts, correct_counts


def load_generated_images(img_path, global_indices, transform):
    """Load ``<img_path>/<idx>.png`` for every index: (paths, stacked tensor)."""
    paths, tensors = [], []
    for i in global_indices:
        path = os.path.join(img_path, f"{i}.png")
        paths.append(path)
        tensors.append(transform(Image.open(path).convert("RGB")))
    return paths, torch.stack(tensors, dim=0)


def save_images_with_detection_labels(save_dir, hq_indices, hallu_indices, yolo_results):
    """Save YOLO-annotated images, tagged with verdict and detected count."""
    os.makedirs(save_dir, exist_ok=True)
    hallu_set = set(hallu_indices)
    for idx, result in zip(hq_indices, yolo_results):
        verdict = "count_hallu" if idx in hallu_set else "count_correct"
        result.save(
            os.path.join(save_dir, f"{idx}-{verdict}-[{len(result)}].png"),
            font_size=5, line_width=1, labels=False,
        )


def get_count_correct_batch_mask(batch_preds, target_class_indices_list,
                                 reference_counts_list):
    """Per-sample bool mask: detections contain exactly one target class whose
    instance count is in ``reference_counts_list``.

    Args:
        batch_preds: list of 1-D tensors with the detected class index of every
            box in one image.
    """
    if not batch_preds:
        return torch.tensor([], dtype=torch.bool)

    device = batch_preds[0].device
    target_classes = torch.tensor(target_class_indices_list, device=device)
    valid_counts = torch.tensor(reference_counts_list, device=device, dtype=torch.long)

    results = []
    for pred_cls in batch_preds:
        relevant = pred_cls[torch.isin(pred_cls.to(target_classes.dtype), target_classes)]
        ok = (
            relevant.numel() > 0
            and len(torch.unique(relevant)) == 1
            and torch.isin(torch.tensor(len(relevant), device=device), valid_counts)
        )
        results.append(bool(ok))
    return torch.tensor(results, dtype=torch.bool, device=device)


# --------------------------------------------------------------------------- #
# Trajectory-MSE aggregation
# --------------------------------------------------------------------------- #

def mse_list_to_df(mse_list, global_sample_indices):
    """Flatten per-sample {timestep: mse} dicts into a long-format DataFrame."""
    rows = [
        {"Time Step": t, "MSE": mse, "Sample Index": sample_index}
        for per_sample, sample_index in zip(mse_list, global_sample_indices)
        for t, mse in per_sample.items()
    ]
    return pd.DataFrame(rows, columns=["Time Step", "MSE", "Sample Index"])


def cal_diffusion_prior_gap_fid(diffused_noise_path, initial_noise_path):
    """FID between the diffused training noise q(x_T|x_0) and the sampling
    prior N(0, I) — the 'diffusion prior gap'."""
    return calculate_fid_given_paths(
        paths=[diffused_noise_path, initial_noise_path],
        batch_size=128,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_workers=8,
    )

# Plot styling per sample category.
_MSE_CATEGORIES = {
    "count_hallu": ("Hallucinations", "o", "orange"),
    "count_correct": ("Non-hallucinations", "s", "blue"),
    "lq": ("Visual failure samples", "^", "green"),
}


def compute_mse(file_path, mse_df, count_hallu_df, count_correct_df, lq_df):
    """Aggregate the per-timestep reconstruction MSE by sample category.

    Returns a dict with, per non-empty category and for all samples:
    ``<name>_mean_each_t`` / ``<name>_std_each_t`` (over the reverse process,
    latest timestep first) and ``<name>_mean_all`` / ``<name>_std_all``.
    Also saves a mean±std curve plot to ``<file_path>/mse.pdf``.
    """
    stats = {}
    plotted = False
    for name, df in [("count_hallu", count_hallu_df), ("count_correct", count_correct_df),
                     ("lq", lq_df), ("all", mse_df)]:
        if df.empty:
            continue
        mean = df.groupby("Time Step")["MSE"].mean().sort_index(ascending=False)
        std = df.groupby("Time Step")["MSE"].std().sort_index(ascending=False)
        stats[f"{name}_mean_each_t"] = mean.tolist()
        stats[f"{name}_std_each_t"] = std.tolist()
        stats[f"{name}_mean_all"] = df["MSE"].mean().item()
        stats[f"{name}_std_all"] = df["MSE"].std()

        if name in _MSE_CATEGORIES:
            label, marker, color = _MSE_CATEGORIES[name]
            mark_step = max(len(mean) // 100, 1)
            plt.plot(mean.index, mean.values, label=label, marker=marker,
                     markevery=mark_step, markersize=4, color=color)
            plt.fill_between(mean.index, mean.values - std, mean.values + std,
                             alpha=0.1, color=color)
            plotted = True

    if plotted:
        plt.legend()
        plt.ylabel(r"$\|x_t - \tilde{x}_t\|^2$")
        plt.xlabel("Reverse Process (Time Step)")
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(file_path, "mse.pdf"))
        plt.close()
    return stats
