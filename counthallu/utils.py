"""Shared utilities: seeding, model loading, image I/O, and UNet surgery."""

import os
import random
import re
import shutil

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """Gather ``arr[timesteps]`` and right-pad dims to ``broadcast_shape``."""
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while res.dim() < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# --------------------------------------------------------------------------- #
# Evaluator-model loading
# --------------------------------------------------------------------------- #

def _resolve_hub_checkpoint(repo_id, filenames=("model.pth", "model.pt")):
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id)
    for name in filenames:
        if name in files:
            return hf_hub_download(repo_id=repo_id, filename=name)
    raise FileNotFoundError(f"No checkpoint ({'/'.join(filenames)}) found in {repo_id}.")


def load_counting_model(dataset_name, device="cpu", model_path=None,
                        use_hub_model=False, repo_id=None):
    """Load the counting model for a dataset.

    Returns (model, model_type, reference_counts_list, target_class_indices_list);
    the last two are only used by detection-based counters.
    """
    from counthallu.models.counting import CountingRegressor

    if use_hub_model and repo_id is not None:
        model_path = _resolve_hub_checkpoint(repo_id)

    if dataset_name == "realhand":
        from ultralytics import YOLO
        model = YOLO(model_path, task="detect").to(device)
        return model, "fingers_detection", [5], [0]  # 5 fingers, class 0 = finger

    if dataset_name in ("toyshape", "simobject"):
        model = CountingRegressor(num_classes=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, "regression", None, None

    raise NotImplementedError(f"No counting model registered for dataset: {dataset_name}")


def load_quality_cls_model(dataset_name, device="cpu", model_path=None,
                           use_hub_model=False, repo_id=None):
    """Load the binary image-quality classifier (currently RealHand only)."""
    import timm

    if use_hub_model and repo_id is not None:
        model_path = _resolve_hub_checkpoint(repo_id, filenames=("model.pth",))

    if dataset_name == "realhand":
        # Checkpoints were saved from a bare timm model; keep the same format.
        model = timm.create_model("maxvit_small_tf_224", pretrained=True, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    raise NotImplementedError(f"No quality classifier registered for dataset: {dataset_name}")


# --------------------------------------------------------------------------- #
# Checkpoint-path helpers
# --------------------------------------------------------------------------- #

def get_pipeline_path(model_path=None, hf_pipeline_path=None):
    """Locate a saved pipeline: prefer the hub snapshot, else find the
    ``final-step-*`` folder under a local training output directory."""
    if hf_pipeline_path:
        return hf_pipeline_path
    if model_path:
        for root, dirs, _ in os.walk(model_path):
            for d in dirs:
                if d.startswith("final-step"):
                    return os.path.join(root, d)
    raise FileNotFoundError(
        f"No pipeline found. Checked model_path={model_path}, "
        f"hf_pipeline_path={hf_pipeline_path}."
    )


# Info encoded in the training run-id directory name (see train/common.py).
_PATH_PATTERNS = {
    "step": r"final-step-(\d+)",
    "num_samples": r"num_samples_(\d+)",
    "dataset": r"dataset_([^-/]+)",
    "denoising_model": r"denoising_model_([^-/]+)",
    "diffusion_method": r"diffusion_method_([^-/]+)",
}


def extract_info_from_path(name, path):
    """Parse a run parameter back out of a training output path."""
    if name not in _PATH_PATTERNS:
        raise ValueError(f"Unknown path field: {name}")
    match = re.search(_PATH_PATTERNS[name], path)
    if match is None:
        raise ValueError(f"Cannot find '{name}' in path: {path}")
    value = match.group(1)
    return int(value) if value.isdigit() else value


# --------------------------------------------------------------------------- #
# Image saving during evaluation
# --------------------------------------------------------------------------- #

def save_gen_images_to_png(test_data, img_save_path, global_indices):
    for img, idx in zip(test_data, global_indices):
        img.save(os.path.join(img_save_path, f"{idx}.png"))


def single_tensor_to_pil(img_tensor):
    """(C, H, W) tensor in [-1, 1] -> PIL image."""
    if img_tensor.dim() != 3:
        raise ValueError(f"Unsupported image shape: {img_tensor.shape}")
    img = (img_tensor / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).round().astype("uint8"))


def save_initial_noise_to_png(diffused_noise, initial_noise,
                              diffused_noise_save_path, initial_noise_save_path,
                              global_indices):
    """Save x_T (diffused) and N(0,I) (initial) noise images, used later to
    measure the diffusion prior gap. Extra latent channels beyond RGB are dropped."""
    assert initial_noise.shape[0] == diffused_noise.shape[0]
    initial_noise = initial_noise[:, :3]
    diffused_noise = diffused_noise[:, :3]

    for i in tqdm(range(initial_noise.shape[0]), desc="Saving noise images"):
        idx = global_indices[i]
        single_tensor_to_pil(diffused_noise[i]).save(
            os.path.join(diffused_noise_save_path, f"{idx:05d}.png"))
        single_tensor_to_pil(initial_noise[i]).save(
            os.path.join(initial_noise_save_path, f"{idx:05d}.png"))


def categorize_and_copy_images_by_indices(all_img_path, lq_img_path,
                                          count_correct_img_path, count_hallu_img_path,
                                          lq_global_indices,
                                          count_correct_global_indices,
                                          count_correct_pred_counting_labels,
                                          count_hallu_global_indices,
                                          count_hallu_pred_counting_labels):
    """Copy generated images into per-verdict folders; counted images get
    their predicted count appended to the filename."""
    assert len(count_correct_global_indices) == len(count_correct_pred_counting_labels)
    assert len(count_hallu_global_indices) == len(count_hallu_pred_counting_labels)

    for idx in lq_global_indices:
        shutil.copy(os.path.join(all_img_path, f"{idx}.png"),
                    os.path.join(lq_img_path, f"{idx}.png"))
    for idx, count in zip(count_correct_global_indices, count_correct_pred_counting_labels):
        shutil.copy(os.path.join(all_img_path, f"{idx}.png"),
                    os.path.join(count_correct_img_path, f"{idx}-{count}.png"))
    for idx, count in zip(count_hallu_global_indices, count_hallu_pred_counting_labels):
        shutil.copy(os.path.join(all_img_path, f"{idx}.png"),
                    os.path.join(count_hallu_img_path, f"{idx}-{count}.png"))


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

# VAEs usable with the latent-space unconditional trainers.
VAE_CONFIGS = {
    "CompVis/ldm-celebahq-256": {"model_class": "VQModel", "subfolder": "vqvae"},
    "stabilityai/stable-diffusion-3.5-medium": {"model_class": "AutoencoderKL", "subfolder": "vae"},
}


def load_vae_for_training(diffusion_method, vae_hub_id, accelerator, weight_dtype):
    """Load the frozen VAE for latent methods (ldm/jdm); None for pixel-space ddpm."""
    from diffusers.models import AutoencoderKL, VQModel

    if diffusion_method.lower() == "ddpm":
        return None

    config = VAE_CONFIGS.get(vae_hub_id)
    if config is None:
        raise ValueError(
            f"VAE '{vae_hub_id}' is not registered. Supported: {list(VAE_CONFIGS)}."
        )
    model_class = {"VQModel": VQModel, "AutoencoderKL": AutoencoderKL}[config["model_class"]]
    vae = model_class.from_pretrained(vae_hub_id, subfolder=config["subfolder"])
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    return vae


def expand_unet_channels(unet, new_in_channels=8, new_out_channels=8):
    """Replace conv_in/conv_out with freshly initialized layers of the new
    channel counts, so a pretrained UNet can denoise joint image+mask latents."""
    old_in, old_out = unet.conv_in, unet.conv_out

    unet.conv_in = nn.Conv2d(
        new_in_channels, old_in.out_channels, kernel_size=old_in.kernel_size,
        stride=old_in.stride, padding=old_in.padding, bias=old_in.bias is not None,
    )
    unet.config.in_channels = new_in_channels

    unet.conv_out = nn.Conv2d(
        old_out.in_channels, new_out_channels, kernel_size=old_out.kernel_size,
        stride=old_out.stride, padding=old_out.padding, bias=old_out.bias is not None,
    )
    unet.config.out_channels = new_out_channels
    return unet
