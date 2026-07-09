"""Finetune Stable Diffusion 1.5 on a counting dataset.

One trainer covers the regimes used in the paper, selected via config:
  * training_args.finetune_strategy:   "full", "lora", or "conv_only"
  * training_args.diffusion_method:    "ldm" (images only) or "jdm"
    (joint image+mask: conv_in/conv_out are expanded to 8 latent channels
    and trained from scratch)

Full / conv_only finetuning saves complete pipelines; LoRA saves only the
adapter. To continue from a LoRA-finetuned stage, set
``model_arguments.pretrained_lora`` — the adapter is merged into the base
UNet before training. Launch: ``bash scripts/train.sh t2i config/train/<name>.yaml``
"""

import os

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL, VQModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from counthallu.models.t2i_pipelines import SDPipeline
from counthallu.train import common
from counthallu.utils import expand_unet_channels, seed_all

logger = get_logger(__name__, log_level="INFO")


def configure_trainable_params(pipe, unet, training_args):
    """Freeze everything, then re-enable exactly what the chosen strategy trains.

    Returns the (possibly LoRA-wrapped) UNet.
    """
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if training_args.finetune_strategy == "lora":
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        base_unet = unet.base_model.model
    elif training_args.finetune_strategy == "full":
        unet.requires_grad_(True)
        base_unet = unet
    elif training_args.finetune_strategy == "conv_only":
        base_unet = unet  # only conv_in/conv_out below
    else:
        raise ValueError(f"Unknown finetune_strategy: {training_args.finetune_strategy}")

    # JDM's expanded conv layers are freshly initialized and must always train;
    # the conv_only strategy trains nothing else.
    if training_args.diffusion_method == "jdm" or training_args.finetune_strategy == "conv_only":
        for p in base_unet.conv_in.parameters():
            p.requires_grad = True
        for p in base_unet.conv_out.parameters():
            p.requires_grad = True
    return unet


def encode_to_latents(vae, batch, diffusion_method, weight_dtype, device):
    """VAE-encode the batch; JDM concatenates mask latents channel-wise."""
    def encode(tensor):
        if isinstance(vae, VQModel):
            return vae.encode(tensor).latents
        if isinstance(vae, AutoencoderKL):
            return vae.encode(tensor).latent_dist.sample()
        raise TypeError(f"Unsupported VAE type: {type(vae)}")

    latents = encode(batch["image"].to(weight_dtype).to(device))
    if diffusion_method == "jdm":
        masks = encode(batch["mask"].to(weight_dtype).to(device))
        latents = torch.cat([latents, masks], dim=1)
    return latents * vae.config.scaling_factor


def main():
    training_args, model_args, data_args = common.parse_config()
    seed_all(training_args.seed)

    run_id = common.build_run_id(training_args, model_args, data_args)
    project_dir = os.path.join(training_args.save_parent_dir, run_id)
    accelerator, weight_dtype = common.setup_accelerator(training_args, project_dir)
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(project_dir, exist_ok=True)

    train_dataset, train_dataloader = common.load_train_dataloader(
        accelerator, training_args, data_args
    )

    # Base pipeline; JDM expands the UNet to 8-channel (image+mask) latents.
    pipe = SDPipeline.from_pretrained(
        model_args.pretrained_model,
        torch_dtype=weight_dtype,
        cache_dir=model_args.cache_dir,
        use_safetensors=True,
        variant="fp16" if weight_dtype == torch.float16 else None,
    ).to(device)
    vae = pipe.vae
    unet = pipe.unet

    # If the previous stage was LoRA-finetuned, fold its adapter into the
    # UNet weights before any channel surgery or freezing.
    if model_args.pretrained_lora:
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, model_args.pretrained_lora).merge_and_unload()
        pipe.unet = unet

    if training_args.diffusion_method == "jdm":
        unet = expand_unet_channels(
            unet,
            new_in_channels=vae.config.latent_channels * 2,
            new_out_channels=vae.config.latent_channels * 2,
        )
    unet = configure_trainable_params(pipe, unet, training_args)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=common.scaled_learning_rate(training_args, accelerator),
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        weight_decay=training_args.adam_weight_decay,
        eps=training_args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=training_args.lr_warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    ema = common.create_ema(training_args, unet, device)
    common.init_trackers(accelerator, training_args, model_args, data_args, run_id,
                         unet, len(train_dataset))

    # The training prompt is fixed, so encode it once per batch size.
    prompt_cache = {}

    def cached_prompt_embeds(bsz):
        if bsz not in prompt_cache:
            prompt_cache[bsz], _ = pipe.encode_prompt(
                prompt=[training_args.prompt] * bsz, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        return prompt_cache[bsz]

    def build_inference_pipeline(current_unet):
        """Wrap current weights for preview sampling / full-model saving."""
        return SDPipeline(
            vae=vae, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
            unet=current_unet, scheduler=pipe.scheduler,
            safety_checker=pipe.safety_checker, feature_extractor=pipe.feature_extractor,
        )

    def save_checkpoint(current_unet, save_path):
        os.makedirs(save_path, exist_ok=True)
        if training_args.finetune_strategy == "lora":
            current_unet.save_pretrained(save_path)  # adapter only
        else:
            build_inference_pipeline(current_unet).save_pretrained(save_path)

    max_train_steps = training_args.max_train_steps
    logger.info(f"***** Training (SD-1.5 / {training_args.finetune_strategy}) *****")
    logger.info(f"  Examples = {len(train_dataset)}, max steps = {max_train_steps}")

    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="Training steps",
                        disable=not accelerator.is_local_main_process)

    while global_step < max_train_steps:
        unet.train()
        for batch in train_dataloader:
            latents = encode_to_latents(vae, batch, training_args.diffusion_method,
                                        weight_dtype, device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noise = torch.randn(latents.shape, device=device, dtype=weight_dtype)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(unet):
                pred = unet(noisy_latents, timesteps,
                            encoder_hidden_states=cached_prompt_embeds(bsz)).sample

                if training_args.prediction_type == "epsilon":
                    target = noise
                elif training_args.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {training_args.prediction_type}")
                loss = F.mse_loss(pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            # One optimizer step completed.
            if ema is not None:
                ema.step(unet.parameters())
            global_step += 1
            progress_bar.update(1)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step}
            if ema is not None:
                logs["ema_decay"] = ema.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Periodic sample previews.
            if (global_step % training_args.save_image_steps == 0
                    and accelerator.is_main_process):
                unwrapped = accelerator.unwrap_model(unet)
                with common.ema_weights(ema, unwrapped):
                    inference_pipe = build_inference_pipeline(unwrapped)
                    inference_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config)
                    generator = torch.Generator(device=device).manual_seed(
                        training_args.seed)
                    out = inference_pipe(
                        prompt=[training_args.prompt] * training_args.eval_batch_size,
                        num_inference_steps=training_args.num_inference_timesteps,
                        generator=generator,
                        height=data_args.img_size,
                        width=data_args.img_size,
                    )
                    common.log_images(accelerator, training_args, out.images,
                                      global_step, masks=out.masks)

            # Periodic checkpoints.
            if (global_step % training_args.save_model_steps == 0
                    and accelerator.is_main_process):
                unwrapped = accelerator.unwrap_model(unet)
                with common.ema_weights(ema, unwrapped):
                    save_checkpoint(unwrapped, os.path.join(project_dir, f"step-{global_step}"))

            if global_step >= max_train_steps:
                break
        accelerator.wait_for_everyone()

    progress_bar.close()

    # Final save (EMA weights when enabled).
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        with common.ema_weights(ema, unwrapped):
            save_path = os.path.join(project_dir, f"final-step-{global_step}")
            save_checkpoint(unwrapped, save_path)
            common.push_to_hub_if_requested(training_args, save_path, global_step)

    accelerator.end_training()


if __name__ == "__main__":
    main()
