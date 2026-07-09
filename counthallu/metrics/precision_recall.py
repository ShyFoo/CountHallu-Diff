"""Improved Precision & Recall between real and generated image folders.

Follows Kynkäänniemi et al., "Improved Precision and Recall Metric for
Assessing Generative Models" (NeurIPS 2019), using the StyleGAN2-ADA VGG16
feature extractor.
"""

import math
import os
from collections import namedtuple

import torch
import torch.nn as nn
from torch.hub import download_url_to_file, get_dir
from torch.utils.data import DataLoader
from tqdm import tqdm

from counthallu.metrics.inception import ImagePathDataset

Manifold = namedtuple("Manifold", ["features", "kth"])


class VGGFeatureExtractor(nn.Module):
    WEIGHTS_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"

    def __init__(self):
        super().__init__()
        model_path = os.path.join(get_dir(), os.path.basename(self.WEIGHTS_URL))
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_url_to_file(self.WEIGHTS_URL, model_path)
        self.model = torch.jit.load(model_path).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.model(x, return_features=True)


def compute_distance(row_features, col_features, row_batch_size, col_batch_size, device):
    """Pairwise distances, computed block-wise so features can stay on CPU."""
    dist = []
    for row_batch in row_features.split(row_batch_size, dim=0):
        row_batch = row_batch.to(device).to(torch.float32)
        dist_batch = [
            torch.cdist(row_batch.unsqueeze(0),
                        col_batch.to(device).to(torch.float32).unsqueeze(0)).squeeze(0).cpu()
            for col_batch in col_features.split(col_batch_size, dim=0)
        ]
        dist.append(torch.cat(dist_batch, dim=1))
    return torch.cat(dist, dim=0)


class ManifoldBuilder:
    """Extracts VGG features for a dataset and the k-th neighbor radius of each
    feature, which together define the manifold used by precision/recall."""

    def __init__(self, data, extr_batch_size=128, max_sample_size=50000, nhood_size=3,
                 row_batch_size=10000, col_batch_size=10000, num_workers=0,
                 device=torch.device("cpu")):
        self.nhood_size = nhood_size
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.device = device

        if len(data) > max_sample_size:
            generator = torch.Generator().manual_seed(1234)
            indices = torch.randperm(len(data), generator=generator)[:max_sample_size]
            data = torch.utils.data.Subset(data, indices=indices)

        extractor = VGGFeatureExtractor().to(device)
        dataloader = DataLoader(data, batch_size=extr_batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=False)
        features = []
        with torch.inference_mode():
            num_batches = math.ceil(len(data) / extr_batch_size)
            for x in tqdm(dataloader, desc="Extracting VGG features", total=num_batches):
                if isinstance(x, (list, tuple)):
                    x = x[0]
                features.append(extractor(x.to(device)).cpu())
        # half precision keeps the pairwise distance matrices manageable
        self.features = torch.cat(features, dim=0).to(torch.float16)
        self.kth = self._compute_kth(self.features)

    def _compute_kth(self, features):
        kth = []
        for row_batch in tqdm(features.split(self.row_batch_size, dim=0),
                              desc="Computing k-th radii"):
            dist = compute_distance(row_batch, features, self.row_batch_size,
                                    self.col_batch_size, self.device)
            # nhood_size + 1 excludes each point's zero distance to itself
            kth.append(dist.to(torch.float32)
                       .kthvalue(self.nhood_size + 1, dim=1).values.to(torch.float16))
        return torch.cat(kth)

    @property
    def manifold(self):
        return Manifold(features=self.features, kth=self.kth)


def _fraction_within(probe: Manifold, reference: Manifold, batch_size, device):
    """Fraction of probe features that fall inside the reference manifold."""
    pred = []
    for probe_batch in tqdm(probe.features.split(batch_size), desc="Comparing manifolds"):
        dist = compute_distance(probe_batch, reference.features, batch_size, batch_size, device)
        pred.append((dist <= reference.kth.unsqueeze(0)).any(dim=1))
    return torch.cat(pred).to(torch.float32).mean()


def calculate_pr_given_paths(paths, batch_size, device="cpu"):
    """(precision, recall) of generated images `paths[1]` w.r.t. real images `paths[0]`."""
    device = torch.device(device)
    real = ManifoldBuilder(ImagePathDataset(paths[0]), device=device).manifold
    gen = ManifoldBuilder(ImagePathDataset(paths[1]), device=device).manifold
    precision = _fraction_within(gen, real, batch_size, device)
    recall = _fraction_within(real, gen, batch_size, device)
    return precision, recall


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="python -m counthallu.metrics.precision_recall REAL_DIR GEN_DIR")
    parser.add_argument("path", type=str, nargs=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    precision, recall = calculate_pr_given_paths(args.path, args.batch_size, args.device)
    print(f"Precision: {precision}, Recall: {recall}")
