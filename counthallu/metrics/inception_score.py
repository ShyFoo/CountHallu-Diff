"""Inception Score of a folder of generated images."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from tqdm import tqdm

from counthallu.metrics.inception import ImagePathDataset


def calculate_is_given_path(path, batch_size, device="cpu", splits=10):
    """Returns (mean, std) of the Inception Score over ``splits`` subsets."""
    dataset = ImagePathDataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing IS predictions"):
            pred = F.softmax(model(batch.to(device)), dim=1)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    split_scores = []
    split_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        scores = [entropy(pyx, py) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="python -m counthallu.metrics.inception_score --img_path DIR")
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    mean_is, std_is = calculate_is_given_path(args.img_path, args.batch_size, args.device)
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
