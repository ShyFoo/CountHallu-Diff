"""Train an unconditional diffusion model on a counting dataset.

Supports three regimes via ``diffusion_method``:
  * ddpm — pixel-space diffusion;
  * ldm  — latent diffusion behind a frozen pretrained VAE;
  * jdm  — joint latent diffusion over (image, mask) pairs, concatenated on
           the channel dim (requires a dataset that yields masks).

Launch: ``bash scripts/train.sh uncond config/train/<name>.yaml``
"""

import os

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL, VQModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from counthallu.models.pipelines import DDPMPipeline, LDMPipeline
from counthallu.models.unet import create_unet
from counthallu.train import common
from counthallu.utils import extract_into_tensor, load_vae_for_training, seed_all

logger = get_logger(__name__, log_level="INFO")


def encode_to_latents(vae, batch, diffusion_method, weight_dtype, device):
    """VAE-encode a batch; JDM additionally encodes the mask and concatenates
    it channel-wise. Returns scaled latents ready for noising."""
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


def build_pipeline(model, vae, scheduler, model_args, training_args, data_args,
                   weight_dtype, num_samples, max_train_steps):
    """Wrap the current weights in an inference pipeline whose config records
    the run parameters (parsed back at evaluation time)."""
    if "latent" in model_args.denoising_model:
        pipeline = LDMPipeline(vqvae=vae, unet=model, scheduler=scheduler)
    else:
        pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    pipeline.torch_dtype = weight_dtype
    pipeline.register_to_config(
        step=max_train_steps,
        denoising_model=model_args.denoising_model,
        num_samples=num_samples,
        diffusion_method=training_args.diffusion_method,
        dataset_name=data_args.dataset_name,
    )
    return pipeline


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

    # Model, frozen VAE (latent methods only), and the forward-process scheduler.
    model = create_unet(model_args.denoising_model)
    vae = load_vae_for_training(
        training_args.diffusion_method, model_args.vae_hub_id, accelerator, weight_dtype
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=training_args.num_training_timesteps,
        beta_schedule=training_args.beta_schedule,
        prediction_type=training_args.prediction_type,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    ema = common.create_ema(training_args, model, device)
    common.init_trackers(accelerator, training_args, model_args, data_args, run_id,
                         model, len(train_dataset))

    max_train_steps = training_args.max_train_steps
    logger.info("***** Training (unconditional) *****")
    logger.info(f"  Examples = {len(train_dataset)}, max steps = {max_train_steps}")

    def make_pipeline(current_model):
        return build_pipeline(current_model, vae, noise_scheduler, model_args,
                              training_args, data_args, weight_dtype,
                              len(train_dataset), max_train_steps)

    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="Training steps",
                        disable=not accelerator.is_local_main_process)

    while global_step < max_train_steps:
        model.train()
        for batch in train_dataloader:
            if vae is not None:
                clean = encode_to_latents(vae, batch, training_args.diffusion_method,
                                          weight_dtype, device)
            else:
                clean = batch["image"].to(weight_dtype).to(device)

            bsz = clean.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noise = torch.randn(clean.shape, device=device, dtype=weight_dtype)
            if "latent" in model_args.denoising_model:
                # Slightly widened noise stabilizes latent-space training.
                noise = noise + 0.1 * torch.randn_like(noise)
            noisy = noise_scheduler.add_noise(clean, noise, timesteps)

            with accelerator.accumulate(model):
                pred = model(sample=noisy, timestep=timesteps).sample
                if training_args.prediction_type == "epsilon":
                    loss = F.mse_loss(pred.float(), noise.float())
                elif training_args.prediction_type == "sample":
                    # SNR-weighted x_0 prediction loss.
                    alpha_t = extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (bsz, 1, 1, 1))
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = (snr_weights *
                            F.mse_loss(pred.float(), clean.float(), reduction="none")).mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {training_args.prediction_type}")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            # One optimizer step completed.
            if ema is not None:
                ema.step(model.parameters())
            global_step += 1
            progress_bar.update(1)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step}
            if ema is not None:
                logs["ema_decay"] = ema.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Periodic sample previews (all ranks generate; images are gathered).
            if (global_step % training_args.save_image_steps == 0
                    or global_step >= max_train_steps):
                unwrapped = accelerator.unwrap_model(model)
                with common.ema_weights(ema, unwrapped):
                    pipeline = make_pipeline(unwrapped)
                    generator = torch.Generator(device=device).manual_seed(
                        training_args.seed + accelerator.process_index)
                    out = pipeline(
                        generator=generator,
                        batch_size=training_args.eval_batch_size,
                        num_inference_steps=training_args.num_inference_timesteps,
                        output_type="tensor",
                    )
                    images = accelerator.gather(out.final_images)
                    masks = (accelerator.gather(out.final_masks)
                             if out.final_masks is not None else None)

                if accelerator.is_main_process:
                    def to_uint8(x):
                        x = (x / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
                        return (x * 255).round().astype("uint8")
                    common.log_images(accelerator, training_args, to_uint8(images),
                                      global_step,
                                      masks=to_uint8(masks) if masks is not None else None)

            # Periodic checkpoints.
            if global_step % training_args.save_model_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                with common.ema_weights(ema, unwrapped):
                    if accelerator.is_main_process:
                        make_pipeline(unwrapped).save_pretrained(
                            os.path.join(project_dir, f"step-{global_step}"))

            if global_step >= max_train_steps:
                break
        accelerator.wait_for_everyone()

    progress_bar.close()

    # Final save (EMA weights when enabled).
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        with common.ema_weights(ema, unwrapped):
            save_path = os.path.join(project_dir, f"final-step-{global_step}")
            make_pipeline(unwrapped).save_pretrained(save_path)
            common.push_to_hub_if_requested(training_args, save_path, global_step)

    accelerator.end_training()


if __name__ == "__main__":
    main()
