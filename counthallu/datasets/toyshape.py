"""ToyShape: synthetic images of non-overlapping white shapes on black background.

Each image contains 1-3 shapes chosen from {triangle, square, pentagon}, at most
one instance per type. The label is the per-type instance count. Use
``python -m counthallu.datasets.toyshape --data_root ... --num_samples ...``
to (re)generate the dataset on disk; the ``ToyShape`` class only loads it.
"""

import csv
import math
import os
import random

from PIL import Image, ImageDraw
from tqdm import tqdm

from counthallu.datasets.common import LabeledImageFolder

SHAPES = ("triangle", "square", "pentagon")


class ToyShape(LabeledImageFolder):
    SUBDIR = "ToyShape"

    def __init__(self, img_size=128, mode="train", hub_dataset=None, data_root=None):
        # No flip augmentation: shapes are orientation-sensitive labels.
        super().__init__(img_size, mode, hub_dataset, data_root, random_flip=False)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #

def _boxes_overlap(b1, b2):
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def _sample_shape(shape_type, placed_boxes, canvas, area, max_attempts=50):
    """Sample placement params for one shape that fits the canvas and avoids
    every box in ``placed_boxes``. Returns None if all attempts fail."""
    for _ in range(max_attempts):
        if shape_type == "square":
            side = int(math.sqrt(area))
            if canvas - side < 0:
                return None
            x1, y1 = random.randint(0, canvas - side), random.randint(0, canvas - side)
            bbox = (x1, y1, x1 + side, y1 + side)
        elif shape_type == "triangle":
            side = int(math.sqrt(4 * area / math.sqrt(3)))
            height = int(side * math.sqrt(3) / 2)
            if canvas - side < 0 or canvas - height < 0:
                return None
            x1, y1 = random.randint(0, canvas - side), random.randint(0, canvas - height)
            bbox = (x1, y1, x1 + side, y1 + height)
        else:  # pentagon: sample a center-ish anchor, then check the vertex bbox
            side = int(math.sqrt(4 * area / math.sqrt(5 * (5 + 2 * math.sqrt(5)))))
            x1, y1 = random.randint(0, canvas), random.randint(0, canvas)
            xs, ys = zip(*_pentagon_vertices(x1, y1, side))
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > canvas or bbox[3] > canvas:
                continue

        if any(_boxes_overlap(bbox, b) for b in placed_boxes):
            continue
        return {"type": shape_type, "x1": x1, "y1": y1, "side": side, "bbox": bbox}
    return None


def _pentagon_vertices(x1, y1, side):
    return [
        (x1 + side * math.cos(math.radians(72 + i * 72)),
         y1 - side * math.sin(math.radians(72 + i * 72)))
        for i in range(5)
    ]


def _render(shapes, img_size):
    image = Image.new("RGB", (img_size, img_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    for sp in shapes:
        x1, y1, side = sp["x1"], sp["y1"], sp["side"]
        if sp["type"] == "square":
            draw.rectangle([x1, y1, x1 + side, y1 + side], fill="white")
        elif sp["type"] == "triangle":
            height = int(side * math.sqrt(3) / 2)
            draw.polygon([(x1, y1), (x1 + side, y1), (x1 + side // 2, y1 + height)], fill="white")
        else:
            draw.polygon(_pentagon_vertices(x1, y1, side), fill="white")
    return image


def generate_toyshape(data_root, num_samples=30000, img_size=128, area=120, seed=1111):
    """Generate the ToyShape dataset under ``<data_root>/ToyShape``."""
    random.seed(seed)
    data_path = os.path.join(data_root, ToyShape.SUBDIR)
    img_path = os.path.join(data_path, "images")
    os.makedirs(img_path, exist_ok=True)

    with open(os.path.join(data_path, "labels.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", *SHAPES])

        for idx in tqdm(range(num_samples), desc="Generating ToyShape"):
            # Try to place k distinct shape types; retry with a new k on collision failure.
            shapes = []
            for _ in range(50):
                k = random.choice([1, 2, 3])
                candidate, boxes = [], []
                for shape_type in random.sample(SHAPES, k):
                    params = _sample_shape(shape_type, boxes, img_size, area)
                    if params is None:
                        break
                    candidate.append(params)
                    boxes.append(params["bbox"])
                if len(candidate) == k:
                    shapes = candidate
                    break
            if not shapes:  # fallback: a single shape always fits
                params = _sample_shape(random.choice(SHAPES), [], img_size, area)
                shapes = [params] if params else []

            filename = f"{idx:05d}.png"
            _render(shapes, img_size).save(os.path.join(img_path, filename))
            counts = {s: 0 for s in SHAPES}
            for sp in shapes:
                counts[sp["type"]] += 1
            writer.writerow([filename, *(counts[s] for s in SHAPES)])

    print(f"Saved {num_samples} samples to {data_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate the ToyShape dataset.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--area", type=int, default=120, help="Target area (px^2) of each shape.")
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()
    generate_toyshape(args.data_root, args.num_samples, args.img_size, args.area, args.seed)
