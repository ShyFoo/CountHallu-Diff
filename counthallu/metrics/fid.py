"""Frechet Inception Distance between two image folders.

Adapted from https://github.com/mseitzer/pytorch-fid (Apache-2.0).
"""

import os

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from tqdm import tqdm

from counthallu.metrics.inception import ImagePathDataset, InceptionV3


def get_activations(path, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    """Pool-3 activations of every image under ``path``: array (num_images, dims)."""
    model.eval()
    dataset = ImagePathDataset(path)
    if batch_size > len(dataset):
        print("Warning: batch size is bigger than the data size; reducing it.")
        batch_size = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers)
    pred_arr = np.empty((len(dataset), dims))
    start_idx = 0
    for batch in tqdm(dataloader, desc="Extracting Inception features"):
        with torch.no_grad():
            pred = model(batch.to(device))[0]
        # Non-default dims produce spatial outputs; pool them to 1x1.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable implementation by Dougal J. Sutherland.
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"FID produced a singular product; adding {eps} to the covariance diagonals.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    """(mu, sigma) of Inception activations; ``path`` may be an image folder or
    a precomputed .npz with 'mu' and 'sigma'."""
    if path.endswith(".npz"):
        with np.load(path) as f:
            return f["mu"][:], f["sigma"][:]
    act = get_activations(path, model, batch_size, dims, device, num_workers)
    return np.mean(act, axis=0), np.cov(act, rowvar=False)


def calculate_fid_given_paths(paths, batch_size, device="cpu", dims=2048, num_workers=1):
    """FID between two image folders (or .npz statistic files)."""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f"Invalid path: {p}")

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)
    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size, dims, device, num_workers)
    return calculate_frechet_distance(m1, s1, m2, s2)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="python -m counthallu.metrics.fid REAL_DIR GEN_DIR")
    parser.add_argument("path", type=str, nargs=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fid = calculate_fid_given_paths(args.path, args.batch_size, args.device,
                                    num_workers=args.num_workers)
    print("FID:", fid)
