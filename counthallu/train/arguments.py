"""Argument dataclasses shared by both trainers.

Configs are YAML files with three sections (training_arguments,
model_arguments, data_arguments) mapping onto these dataclasses; see
config/train/*.yaml. Authentication for the Hugging Face Hub comes from
`huggingface-cli login` or the HF_TOKEN environment variable — never put
tokens in configs.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    seed: int = field(default=1111, metadata={"help": "Random seed."})
    train_batch_size: int = field(default=64, metadata={"help": "Per-device batch size."})
    eval_batch_size: int = field(default=4, metadata={"help": "Batch size for sample previews."})
    max_train_steps: int = field(default=50000, metadata={"help": "Total optimizer steps."})
    num_workers: int = field(default=0, metadata={"help": "Dataloader workers."})
    save_image_steps: int = field(default=10000, metadata={"help": "Log sample images every N steps."})
    save_model_steps: int = field(default=10000, metadata={"help": "Save a checkpoint every N steps."})

    # diffusion
    diffusion_method: str = field(
        default="ddpm",
        metadata={"help": "'ddpm' (pixel), 'ldm' (latent), or 'jdm' (joint image+mask latent)."})
    num_training_timesteps: int = field(default=1000, metadata={"help": "Forward-process steps."})
    num_inference_timesteps: int = field(default=1000, metadata={"help": "Sampling steps for previews."})
    beta_schedule: str = field(
        default="linear",
        metadata={"help": "'linear', 'scaled_linear', or 'squaredcos_cap_v2'."})
    prediction_type: str = field(
        default="epsilon",
        metadata={"help": "'epsilon', 'sample' (uncond only), or 'v_prediction' (t2i only)."})

    # optimization
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Base LR; scaled linearly by effective_batch_size / lr_ref_batch_size."})
    lr_ref_batch_size: int = field(
        default=256, metadata={"help": "Reference effective batch size for LR scaling."})
    lr_scheduler: str = field(default="cosine", metadata={"help": "diffusers LR schedule name."})
    lr_warmup_steps: int = field(default=500, metadata={"help": "Warmup steps."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Grad accumulation."})
    adam_beta1: float = field(default=0.95, metadata={"help": "AdamW beta1."})
    adam_beta2: float = field(default=0.999, metadata={"help": "AdamW beta2."})
    adam_weight_decay: float = field(default=1e-6, metadata={"help": "AdamW weight decay."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "AdamW epsilon."})
    mixed_precision: str = field(default="fp16", metadata={"help": "'no', 'fp16', or 'bf16'."})

    # EMA
    use_ema: bool = field(default=True, metadata={"help": "Keep an EMA copy of the weights."})
    ema_inv_gamma: float = field(default=1.0, metadata={"help": "EMA warmup inverse gamma."})
    ema_power: float = field(default=0.75, metadata={"help": "EMA warmup power."})
    ema_max_decay: float = field(default=0.9999, metadata={"help": "Maximum EMA decay."})

    # finetuning (t2i only)
    finetune_strategy: str = field(
        default="full",
        metadata={"help": "t2i trainer only: 'full' (whole UNet), 'lora' (attention "
                          "adapters), or 'conv_only' (just conv_in/conv_out)."})
    lora_rank: int = field(default=16, metadata={"help": "LoRA rank (= alpha)."})
    prompt: str = field(
        default="A single human hand.",
        metadata={"help": "Fixed training prompt (t2i trainer only)."})

    # logging & output
    logger: str = field(default="wandb", metadata={"help": "'wandb' or 'tensorboard'."})
    save_parent_dir: str = field(
        default="./results", metadata={"help": "Parent directory for run outputs."})
    push_to_hub: bool = field(default=False, metadata={"help": "Upload final model to the Hub."})
    hub_model_id: str = field(
        default=None, metadata={"help": "Hub repo id, e.g. '<user>/<model>'."})
    hub_private_repo: bool = field(default=True, metadata={"help": "Create a private Hub repo."})


@dataclass
class ModelArguments:
    denoising_model: str = field(
        default="unet_pixel_128",
        metadata={"help": "Uncond: UNet preset (see models/unet.py). "
                          "T2I: a label recorded in the run id, e.g. 'sd-1-5'."})
    pretrained_model: str = field(
        default=None,
        metadata={"help": "Base pipeline for t2i finetuning: a Hub id or local path."})
    pretrained_lora: str = field(
        default=None,
        metadata={"help": "Optional LoRA adapter merged into the UNet right after "
                          "loading pretrained_model (use when the previous stage "
                          "was LoRA-finetuned and only saved an adapter)."})
    cache_dir: str = field(default=None, metadata={"help": "Hugging Face cache directory."})
    vae_hub_id: str = field(
        default="CompVis/ldm-celebahq-256",
        metadata={"help": "VAE for latent-space unconditional training."})


@dataclass
class DataArguments:
    img_size: int = field(default=128, metadata={"help": "Training resolution."})
    dataset_name: str = field(default="toyshape", metadata={"help": "Registered dataset name."})
    dataset_root: str = field(default="./data", metadata={"help": "Local dataset root."})
    use_hub_dataset: bool = field(default=False, metadata={"help": "Load dataset from the Hub."})
    hub_dataset_id: str = field(default=None, metadata={"help": "Hub dataset repo id."})
