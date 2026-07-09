"""Stable Diffusion pipeline with joint image+mask (JDM) decoding.

``SDPipeline`` is a slimmed-down fork of the diffusers Stable Diffusion
``__call__`` loop. The only functional difference is the decoding stage: a
JDM-finetuned UNet produces 8-channel latents, which are split into image
latents and mask latents and decoded separately. All features this project
does not use (IP-Adapter, callbacks, XLA, ...) were removed; refer to
diffusers for the full-featured original.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils import BaseOutput


@dataclass
class T2IPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    masks: Optional[Union[List[PIL.Image.Image], np.ndarray]]


def _split_jdm_latents(latents):
    """8-channel JDM latents -> (image latents, mask latents); else (latents, None)."""
    if latents.shape[1] == 8:
        return latents.chunk(2, dim=1)
    return latents, None


class SDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(self, prompt=None, height=None, width=None, num_inference_steps=50,
                 guidance_scale=7.5, negative_prompt=None, num_images_per_prompt=1,
                 eta=0.0, generator=None, latents=None, output_type="pil",
                 return_dict=True):
        # Default resolution from the UNet config.
        if not height or not width:
            size = self.unet.config.sample_size
            if not isinstance(size, int):
                size = size[0]
            height = width = size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, None, negative_prompt, None, None)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = 0.0
        self._clip_skip = None
        self._cross_attention_kwargs = None
        self._interrupt = False

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, self.do_classifier_free_guidance,
            negative_prompt,
        )
        if self.do_classifier_free_guidance:
            # Single batched forward pass for unconditional + text branches.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device
        )
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, self.unet.config.in_channels,
            height, width, prompt_embeds.dtype, device, generator, latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_input = self.scheduler.scale_model_input(latent_input, t)

                noise_pred = self.unet(
                    latent_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
                )[0]
                if self.do_classifier_free_guidance:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        latents = latents.to(dtype=prompt_embeds.dtype)
        image_latents, mask_latents = _split_jdm_latents(latents)

        if output_type == "latent":
            image, mask = image_latents, mask_latents
        else:
            image = self.vae.decode(image_latents / self.vae.config.scaling_factor,
                                    return_dict=False, generator=generator)[0]
            mask = None
            if mask_latents is not None:
                mask = self.vae.decode(mask_latents / self.vae.config.scaling_factor,
                                       return_dict=False, generator=generator)[0]
            denorm = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type, do_denormalize=denorm)
            if mask is not None:
                mask = self.image_processor.postprocess(mask, output_type, do_denormalize=denorm)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return T2IPipelineOutput(images=image, masks=mask)
