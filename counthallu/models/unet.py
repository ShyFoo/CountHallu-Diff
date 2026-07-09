"""UNet factory for the unconditional diffusion experiments."""

from diffusers import UNet2DModel

# Randomly initialized pixel-space UNet for 128x128 images.
PIXEL_128_CONFIG = {
    "sample_size": 128,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (64, 128, 256, 256),
    "down_block_types": ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
}


def create_unet(name: str, **overrides) -> UNet2DModel:
    """Create a denoising UNet by preset name.

    Presets:
      - "unet_pixel_128":          random init, pixel space, 128x128 images.
      - "unet_latent_256":         pretrained CompVis/ldm-celebahq-256 UNet,
                                   64x64 latents of 256x256 images.
      - "unet_latent_256_6ch":     same, but with 6 in/out channels for joint
                                   image+mask (JDM) latents.

    ``overrides`` update the config of randomly initialized presets.
    """
    if name == "unet_pixel_128":
        return UNet2DModel(**{**PIXEL_128_CONFIG, **overrides})
    if name == "unet_latent_256":
        return UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
    if name == "unet_latent_256_6ch":
        return UNet2DModel.from_pretrained(
            "CompVis/ldm-celebahq-256",
            subfolder="unet",
            in_channels=6,
            out_channels=6,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
    raise ValueError(
        f"Unsupported UNet preset '{name}'. "
        "Supported: unet_pixel_128, unet_latent_256, unet_latent_256_6ch."
    )
