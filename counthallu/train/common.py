"""Infrastructure shared by both trainers: config parsing, accelerator setup,
dataset/hub plumbing, EMA handling, and logging."""

import contextlib
import os
import sys

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from transformers import HfArgumentParser

from counthallu.datasets import get_dataloader
from counthallu.train.arguments import DataArguments, ModelArguments, TrainingArguments

WEIGHT_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def parse_config():
    """Parse a YAML config path (single CLI argument) or plain CLI flags into
    the three argument dataclasses."""
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
        merged = {
            **config.get("training_arguments", {}),
            **config.get("model_arguments", {}),
            **config.get("data_arguments", {}),
        }
        return parser.parse_dict(merged)  # raises on unknown keys (catches typos)
    return parser.parse_args_into_dataclasses()


def build_run_id(training_args, model_args, data_args):
    """Directory/run name encoding the key hyperparameters; eval scripts parse
    these fields back out (see utils.extract_info_from_path)."""
    return (
        f"dataset_{data_args.dataset_name}-"
        f"diffusion_method_{training_args.diffusion_method}-"
        f"denoising_model_{model_args.denoising_model}-"
        f"noise_schedule_{training_args.beta_schedule}-"
        f"max_steps_{training_args.max_train_steps}"
    )


def setup_accelerator(training_args, project_dir):
    """Accelerator plus the weight dtype implied by its mixed-precision mode."""
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
        log_with=training_args.logger,
        project_config=ProjectConfiguration(project_dir=project_dir, logging_dir=project_dir),
    )
    weight_dtype = WEIGHT_DTYPES.get(accelerator.mixed_precision, torch.float32)
    return accelerator, weight_dtype


def load_train_dataloader(accelerator, training_args, data_args):
    """Build the training dataloader, downloading the hub dataset on the main
    process first so parallel ranks hit a warm cache."""
    hub_dataset = None
    if data_args.use_hub_dataset:
        if accelerator.is_main_process:
            load_dataset(data_args.hub_dataset_id)
        accelerator.wait_for_everyone()
        hub_dataset = load_dataset(data_args.hub_dataset_id)["train"]

    dataset, dataloader, _ = get_dataloader(
        dataset_name=data_args.dataset_name,
        data_root=data_args.dataset_root,
        hub_dataset=hub_dataset,
        img_size=data_args.img_size,
        batch_size=training_args.train_batch_size,
        num_workers=training_args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        mode="train",
    )
    return dataset, dataloader


def scaled_learning_rate(training_args, accelerator):
    """Linear LR scaling: lr * effective_batch_size / lr_ref_batch_size."""
    effective = (training_args.train_batch_size * accelerator.num_processes
                 * training_args.gradient_accumulation_steps)
    return training_args.learning_rate * effective / training_args.lr_ref_batch_size


def init_trackers(accelerator, training_args, model_args, data_args, run_id, model,
                  num_samples):
    if not accelerator.is_main_process:
        return
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.init_trackers(
        project_name="Counting-Hallucination",
        config={
            "task_type": "training",
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.train_batch_size,
            "max_train_steps": training_args.max_train_steps,
            "diffusion_method": training_args.diffusion_method,
            "denoising_model": model_args.denoising_model,
            "dataset": data_args.dataset_name,
            "num_samples": num_samples,
            "total_params": total,
            "trainable_params": trainable,
        },
        init_kwargs={
            "wandb": {
                "name": run_id,
                "tags": [training_args.diffusion_method, model_args.denoising_model,
                         data_args.dataset_name],
                "group": "training",
            }
        },
    )


def create_ema(training_args, model, device):
    """EMA weight tracker, or None when disabled."""
    if not training_args.use_ema:
        return None
    from diffusers.training_utils import EMAModel

    ema = EMAModel(
        model.parameters(),
        decay=training_args.ema_max_decay,
        use_ema_warmup=True,
        inv_gamma=training_args.ema_inv_gamma,
        power=training_args.ema_power,
    )
    ema.to(device)
    return ema


@contextlib.contextmanager
def ema_weights(ema, model):
    """Temporarily swap the model to its EMA weights (no-op when ema is None)."""
    if ema is None:
        yield
        return
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    try:
        yield
    finally:
        ema.restore(model.parameters())


def log_images(accelerator, training_args, images, step, masks=None):
    """Send preview images (numpy or PIL) to wandb from the main process."""
    if not accelerator.is_main_process or training_args.logger != "wandb":
        return
    import numpy as np
    import wandb

    panels = list(images) + (list(masks) if masks is not None else [])
    panels = [np.asarray(img) for img in panels]
    accelerator.get_tracker("wandb").log(
        {"Examples": [wandb.Image(img) for img in panels], "step": step}, step=step
    )


def push_to_hub_if_requested(training_args, folder_path, global_step):
    if not training_args.push_to_hub:
        return
    from huggingface_hub import create_repo, upload_folder

    repo_id = create_repo(
        repo_id=training_args.hub_model_id,
        exist_ok=True,
        private=training_args.hub_private_repo,
    ).repo_id
    upload_folder(repo_id=repo_id, folder_path=folder_path,
                  commit_message=f"Final step {global_step}")
    print(f"Pushed to the Hugging Face Hub: {repo_id} @ step {global_step}")
