"""Unconditional diffusion pipelines with hallucination-analysis hooks.

Extends the standard diffusers DDPM/LDM sampling loops with:
  * optional "diffused" start: diffuse a real sample x_0 to x_T and denoise
    from there instead of pure Gaussian noise;
  * per-step reconstruction error ||x~_t - x_t||^2 against the ground-truth
    forward trajectory of x_0 (used to localize where hallucinations emerge);
  * "perfect score" schedulers that denoise with the ground-truth x_0 / noise,
    isolating sampler error from model error.
"""

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class PipelineOutput(BaseOutput):
    final_images: Union[List[Image.Image], torch.Tensor]
    final_masks: Union[List[Image.Image], torch.Tensor, None]
    initial_noise: Optional[torch.Tensor]
    diffused_noise: Optional[torch.Tensor]
    intermediate_mse: Optional[List[Dict]]  # per sample: {timestep: mse}


@dataclass
class SchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class PerfectDDPMScheduler(DDPMScheduler):
    """DDPM step computed from the ground-truth x_0 instead of a model prediction.

    Implements Eq. (7) of https://arxiv.org/abs/2006.11239 with the true x_0.
    """

    def step(self, timestep, sample, generator=None, return_dict=True,
             x_0=None, epsilon=None) -> Union[SchedulerOutput, Tuple]:
        t = timestep
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        x0_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        xt_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = x0_coeff * x_0 + xt_coeff * sample

        if t > 0:
            noise = randn_tensor(x_0.shape, generator=generator,
                                 device=x_0.device, dtype=x_0.dtype)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t) * noise
            elif self.variance_type == "learned_range":
                variance = torch.exp(0.5 * self._get_variance(t)) * noise
            else:
                variance = (self._get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, x_0)
        return SchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=x_0)


class PerfectDDIMScheduler(DDIMScheduler):
    """DDIM step computed from the ground-truth noise epsilon.

    Implements Eq. (12) of https://arxiv.org/abs/2010.02502 with the true noise.
    """

    def step(self, timestep, sample, eta=0.0, use_clipped_model_output=False,
             generator=None, variance_noise=None, return_dict=True,
             x_0=None, epsilon=None) -> Union[SchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError("Call `set_timesteps` before using the scheduler.")

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * epsilon) / alpha_prod_t ** 0.5
        std_dev_t = eta * self._get_variance(timestep, prev_timestep) ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * epsilon
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (pred_prev_sample, pred_original_sample)
        return SchedulerOutput(prev_sample=pred_prev_sample,
                               pred_original_sample=pred_original_sample)


PERFECT_SCHEDULERS = (PerfectDDPMScheduler, PerfectDDIMScheduler)


def _prepare_start(image_shape, generator, dtype, x_0, sampling_start, ddpm_scheduler):
    """Sample the forward noise, optionally diffuse x_0 to x_T, and pick the
    denoising start point.

    Returns (noise, x_T, initial_noise). ``noise`` is the epsilon shared by the
    forward trajectory (needed by the ground-truth MSE tracking below).
    """
    noise = randn_tensor(image_shape, generator=generator,
                         device=generator.device, dtype=dtype)
    x_T = None
    if x_0 is not None and ddpm_scheduler is not None:
        T = torch.full((image_shape[0],), ddpm_scheduler.config.num_train_timesteps - 1,
                       dtype=torch.int64, device=x_0.device)
        x_T = ddpm_scheduler.add_noise(x_0, noise, T)

    if sampling_start == "diffused" and x_T is not None:
        initial_noise = x_T
    elif sampling_start == "normal":
        initial_noise = randn_tensor(image_shape, generator=generator,
                                     device=generator.device, dtype=dtype)
    else:
        raise ValueError(
            f"sampling_start='{sampling_start}' requires x_0 and a DDPM noise scheduler "
            "when set to 'diffused'; otherwise use 'normal'."
        )
    return noise, x_T, initial_noise


def _track_mse(intermediate_mse, sample, x_0, noise, ddpm_scheduler, timesteps, step_idx, t):
    """Record ||x~_{t-1} - x_{t-1}||^2 per sample, where x_{t-1} follows the
    ground-truth forward trajectory of x_0 (x_0 itself at the final step)."""
    if step_idx + 1 < len(timesteps):
        gt = ddpm_scheduler.add_noise(x_0, noise, timesteps[step_idx + 1])
    else:
        gt = x_0
    for j in range(sample.shape[0]):
        intermediate_mse[j][t.item()] = ((sample[j] - gt[j]) ** 2).mean().cpu().item()


def _to_pil(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return DiffusionPipeline.numpy_to_pil(image)


class DDPMPipeline(DiffusionPipeline):
    """Pixel-space sampling with optional diffused start and MSE tracking."""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, num_inference_steps=1000,
                 output_type="pil", return_dict=True, x_0=None,
                 sampling_start="normal", ddpm_noise_scheduler=None, **kwargs):
        image_shape = (batch_size, self.unet.config.in_channels,
                       self.unet.config.sample_size, self.unet.config.sample_size)
        dtype = getattr(self, "torch_dtype", self.unet.dtype)

        noise, x_T, initial_noise = _prepare_start(
            image_shape, generator, dtype, x_0, sampling_start, ddpm_noise_scheduler
        )

        image = initial_noise
        # set_timesteps with an explicit device matters for multi-GPU inference
        self.scheduler.set_timesteps(num_inference_steps, device=generator.device)
        intermediate_mse = [{} for _ in range(batch_size)]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if isinstance(self.scheduler, PERFECT_SCHEDULERS):
                out = self.scheduler.step(x_0=x_0, timestep=t, sample=image, epsilon=noise)
            else:
                model_output = self.unet(image, t).sample
                out = self.scheduler.step(model_output, t, image)
            image = out.prev_sample

            if x_0 is not None:
                _track_mse(intermediate_mse, image, x_0, noise, ddpm_noise_scheduler,
                           self.scheduler.timesteps, i, t)

        if output_type == "pil":
            image = _to_pil(image)
        elif output_type == "tensor":
            image = image.clamp(-1, 1)
        else:
            raise NotImplementedError(f"Unsupported output type: {output_type}")

        if not return_dict:
            return image
        return PipelineOutput(
            final_images=image,
            final_masks=None,
            initial_noise=initial_noise.cpu(),
            diffused_noise=x_T.cpu() if x_T is not None else None,
            intermediate_mse=intermediate_mse,
        )


class LDMPipeline(DiffusionPipeline):
    """Latent-space sampling; 6-channel latents decode to (image, mask) pairs (JDM)."""

    def __init__(self, vqvae, unet, scheduler):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, eta=0.0, num_inference_steps=50,
                 output_type="pil", return_dict=True, x_0=None,
                 sampling_start="normal", ddpm_noise_scheduler=None, **kwargs):
        latent_shape = (batch_size, self.unet.config.in_channels,
                        self.unet.config.sample_size, self.unet.config.sample_size)
        dtype = getattr(self, "torch_dtype", self.unet.dtype)

        noise, x_T, initial_noise = _prepare_start(
            latent_shape, generator, dtype, x_0, sampling_start, ddpm_noise_scheduler
        )

        latents = initial_noise * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(num_inference_steps, device=generator.device)
        intermediate_mse = [{} for _ in range(batch_size)]

        # Not all schedulers accept eta.
        extra_kwargs = {}
        if "eta" in inspect.signature(self.scheduler.step).parameters:
            extra_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            latent_input = self.scheduler.scale_model_input(latents, t)
            if isinstance(self.scheduler, PERFECT_SCHEDULERS):
                out = self.scheduler.step(x_0=x_0, timestep=t, sample=latents, epsilon=noise)
            else:
                noise_pred = self.unet(latent_input, t).sample
                out = self.scheduler.step(noise_pred, t, latents, **extra_kwargs)
            latents = out.prev_sample

            if x_0 is not None:
                _track_mse(intermediate_mse, latents, x_0, noise, ddpm_noise_scheduler,
                           self.scheduler.timesteps, i, t)

        latents = latents / self.vqvae.config.scaling_factor
        if latents.shape[1] == 6:  # JDM: first 3 channels image, last 3 mask
            image = self.vqvae.decode(latents[:, :3].to(dtype)).sample
            mask = self.vqvae.decode(latents[:, 3:].to(dtype)).sample
        else:
            image = self.vqvae.decode(latents.to(dtype)).sample
            mask = None

        if output_type == "pil":
            image = _to_pil(image)
            mask = _to_pil(mask) if mask is not None else None
        elif output_type == "tensor":
            image = image.clamp(-1, 1)
            mask = mask.clamp(-1, 1) if mask is not None else None
        else:
            raise NotImplementedError(f"Unsupported output type: {output_type}")

        if not return_dict:
            return image
        return PipelineOutput(
            final_images=image,
            final_masks=mask,
            initial_noise=initial_noise.cpu(),
            diffused_noise=x_T.cpu() if x_T is not None else None,
            intermediate_mse=intermediate_mse,
        )


# --------------------------------------------------------------------------- #
# Scheduler factory
# --------------------------------------------------------------------------- #

_BASE_PARAMS = {
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "beta_start": 0.0001,
    "beta_end": 0.02,
}
_DPM_PARAMS = {
    **_BASE_PARAMS,
    "thresholding": False,
    "use_karras_sigmas": False,
}

SCHEDULERS = {
    "ddpm": (DDPMScheduler, {**_BASE_PARAMS, "clip_sample_range": 1.0}),
    "dpm-1": (DPMSolverSinglestepScheduler,
              {**_DPM_PARAMS, "solver_order": 1, "final_sigmas_type": "sigma_min",
               "algorithm_type": "dpmsolver"}),
    "dpm-2": (DPMSolverSinglestepScheduler,
              {**_DPM_PARAMS, "solver_order": 2, "final_sigmas_type": "sigma_min",
               "algorithm_type": "dpmsolver"}),
    "dpm-plus": (DPMSolverMultistepScheduler,
                 {**_DPM_PARAMS, "solver_order": 2, "final_sigmas_type": "zero",
                  "algorithm_type": "dpmsolver++"}),
    "ddpm-gt": (PerfectDDPMScheduler, {**_BASE_PARAMS, "clip_sample_range": 1.0}),
    "ddim-gt": (PerfectDDIMScheduler, {**_BASE_PARAMS, "clip_sample_range": 1.0}),
}


def create_scheduler(name: str, denoising_space: str = "pixel"):
    """Instantiate a sampling scheduler by name ('ddpm', 'dpm-1', 'dpm-2',
    'dpm-plus', 'ddpm-gt', 'ddim-gt'). Latent-space models use the
    scaled_linear beta schedule, pixel-space models use linear."""
    if name not in SCHEDULERS:
        raise ValueError(f"Unsupported scheduler '{name}'. Supported: {', '.join(SCHEDULERS)}.")
    cls, params = SCHEDULERS[name]
    beta_schedule = "scaled_linear" if denoising_space == "latent" else "linear"
    return cls(**params, beta_schedule=beta_schedule)
