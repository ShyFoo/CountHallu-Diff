"""Evaluate counting hallucinations of a trained unconditional diffusion model.

For every training sample the pipeline generates one image (multi-GPU via
accelerate), classifies it as counting-correct / counting hallucination /
visual failure, and reports:
  * hallucination and visual-failure rates,
  * FID / Precision / Recall / Inception Score against the training images,
  * the diffusion prior gap (FID between diffused noise x_T and N(0, I)),
  * per-timestep reconstruction MSE curves split by verdict.

Run through ``bash scripts/evaluate.sh`` or directly with accelerate; most
paths come from ``config/eval/<dataset>.yaml``.
"""

import argparse
import json
import os
import shutil
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_dataset
from diffusers.models import AutoencoderKL, VQModel
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from counthallu.datasets import get_dataloader
from counthallu.metrics.fid import calculate_fid_given_paths
from counthallu.metrics.hallucination import (
    CountHalluQuantifier,
    cal_diffusion_prior_gap_fid,
    compute_mse,
    mse_list_to_df,
)
from counthallu.metrics.inception_score import calculate_is_given_path
from counthallu.metrics.precision_recall import calculate_pr_given_paths
from counthallu.models.pipelines import DDPMPipeline, LDMPipeline, create_scheduler
from counthallu.utils import (
    categorize_and_copy_images_by_indices,
    extract_info_from_path,
    get_pipeline_path,
    load_counting_model,
    load_quality_cls_model,
    save_gen_images_to_png,
    save_initial_noise_to_png,
    seed_all,
)

MSE_COLUMNS = ["Time Step", "MSE", "Sample Index"]


def load_run_metadata(args, pipeline_file_path):
    """Recover the training run parameters, either from the hub pipeline's
    model_index.json or from the local output directory name."""
    if args.use_hub_model:
        model_json = hf_hub_download(repo_id=args.hub_diffusion_pipeline_id,
                                     filename="model_index.json")
        with open(model_json, encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("denoising_model"), data.get("num_samples"),
                data.get("diffusion_method"), data.get("dataset_name"))
    return (
        extract_info_from_path("denoising_model", pipeline_file_path),
        extract_info_from_path("num_samples", pipeline_file_path),
        extract_info_from_path("diffusion_method", pipeline_file_path),
        extract_info_from_path("dataset", pipeline_file_path),
    )


def encode_batch(encoder, batch, diffusion_method, device):
    """VAE-encode a batch for latent pipelines (mask concatenated for JDM);
    pixel pipelines pass images through unchanged."""
    if encoder is None:
        return batch["image"].to(device)

    def encode(tensor):
        if isinstance(encoder, VQModel):
            return encoder.encode(tensor).latents
        if isinstance(encoder, AutoencoderKL):
            return encoder.encode(tensor).latent_dist.sample()
        raise TypeError(f"Unsupported encoder type: {type(encoder)}")

    latents = encode(batch["image"].to(device))
    if diffusion_method == "jdm" and "mask" in batch:
        latents = torch.cat([latents, encode(batch["mask"].to(device))], dim=1)
    return latents * encoder.config.scaling_factor


def evaluate(args):
    seed_all(args.seed)
    timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=300))
    accelerator = Accelerator(kwargs_handlers=[timeout], log_with="wandb")
    device = accelerator.device

    # Fill path-like defaults from the dataset's eval config.
    config_path = args.config or f"./config/eval/{args.dataset_name}.yaml"
    with open(config_path) as f:
        for key, value in yaml.safe_load(f).items():
            setattr(args, key, value)

    # Warm the hub cache on rank 0 before all ranks read it.
    if args.use_hub_model and accelerator.is_main_process:
        snapshot_download(repo_id=args.hub_counting_model_id)
        snapshot_download(repo_id=args.hub_diffusion_pipeline_id)
        if args.hub_quality_cls_model_id is not None:
            snapshot_download(repo_id=args.hub_quality_cls_model_id)
    accelerator.wait_for_everyone()

    pipeline_file_path = get_pipeline_path(
        model_path=args.diffusion_pipeline_path,
        hf_pipeline_path=(snapshot_download(repo_id=args.hub_diffusion_pipeline_id)
                          if args.use_hub_model else None),
    )
    denoising_model, num_samples, diffusion_method, dataset_name = load_run_metadata(
        args, pipeline_file_path
    )
    denoising_space = "latent" if "latent" in denoising_model else "pixel"

    # Evaluator models.
    if not args.counting_model_path and not args.use_hub_model:
        raise ValueError("No counting model provided (counting_model_path / hub id).")
    counting_model, counting_model_type, reference_counts, target_classes = load_counting_model(
        dataset_name=dataset_name,
        model_path=args.counting_model_path,
        device=device,
        use_hub_model=args.use_hub_model,
        repo_id=args.hub_counting_model_id,
    )
    quality_cls_model = None
    if args.quality_cls_model_path or args.hub_quality_cls_model_id:
        quality_cls_model = load_quality_cls_model(
            dataset_name=dataset_name,
            model_path=args.quality_cls_model_path,
            device=device,
            use_hub_model=args.use_hub_model,
            repo_id=args.hub_quality_cls_model_id,
        )
    # Visual byproducts (sorted images, YOLO overlays) are kept for one seed only.
    keep_visuals = args.seed == args.visual_seed
    quantifier = CountHalluQuantifier(
        counting_model=counting_model,
        counting_model_type=counting_model_type,
        device=device,
        reference_counts_list=reference_counts,
        quality_cls_model=quality_cls_model,
        target_class_indices_list=target_classes,
        save_detection_results=keep_visuals,
    )

    # Training data provides x_0 for the diffused start / MSE reference.
    hub_dataset = None
    if args.use_hub_dataset:
        if accelerator.is_main_process:
            load_dataset(args.hub_dataset_id)
        accelerator.wait_for_everyone()
        hub_dataset = load_dataset(args.hub_dataset_id)["train"]

    train_dataset, _, train_img_path = get_dataloader(
        dataset_name=dataset_name,
        data_root=args.dataset_root,
        hub_dataset=hub_dataset,
        img_size=args.img_size,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        mode="eval",
    )

    # Output directory layout.
    results_save_path = os.path.join(
        args.save_root,
        dataset_name,
        f"diffusion_method_{diffusion_method}-denoising_model_{denoising_model}-"
        f"num_samples_{num_samples}-sampling_solver_{args.sampling_solver}-"
        f"sampling_steps_{args.num_inference_timesteps}-initial_noise_{args.initial_noise}",
        f"seed-{args.seed}",
    )
    gen_img_path = os.path.join(results_save_path, "gen_images")
    lq_img_path = os.path.join(results_save_path, "visual_failure_samples")
    correct_img_path = os.path.join(results_save_path, "visual_success_samples",
                                    "counting_correct_samples")
    hallu_img_path = os.path.join(results_save_path, "visual_success_samples",
                                  "counting_hallucinations")
    diffused_noise_path = os.path.join(results_save_path, "diffused_noise")
    initial_noise_path = os.path.join(results_save_path, "initial_noise")

    if os.path.exists(os.path.join(results_save_path, "results.txt")):
        if accelerator.is_main_process:
            print(f"results.txt already exists, skipping: {results_save_path}")
        return

    if accelerator.is_main_process:
        if os.path.exists(results_save_path):  # clear partial previous runs
            shutil.rmtree(results_save_path)
        for path in (gen_img_path, lq_img_path, correct_img_path, hallu_img_path,
                     diffused_noise_path, initial_noise_path):
            os.makedirs(path, exist_ok=True)
        print(f"Evaluating pipeline: {pipeline_file_path}")

        run_id = (f"{diffusion_method}-{dataset_name}-{denoising_model}-{num_samples}-"
                  f"{args.sampling_solver}-{args.num_inference_timesteps}-"
                  f"{args.initial_noise}-seed-{args.seed}")
        accelerator.init_trackers(
            project_name="Counting-Hallucination-Evaluation",
            config={
                "task_type": "evaluation",
                "dataset_name": dataset_name,
                "denoising_model": denoising_model,
                "num_samples": num_samples,
                "sampling_solver": args.sampling_solver,
                "sampling_steps": args.num_inference_timesteps,
                "initial_noise": args.initial_noise,
            },
            init_kwargs={"wandb": {
                "name": run_id,
                "tags": [diffusion_method, denoising_model, str(num_samples)],
                "group": "evaluation",
            }},
        )

    # Load the pipeline with the requested sampling solver.
    pipeline_scheduler = create_scheduler(args.sampling_solver, denoising_space)
    ddpm_noise_scheduler = create_scheduler("ddpm", denoising_space)
    pipeline_cls = LDMPipeline if denoising_space == "latent" else DDPMPipeline
    pipeline = pipeline_cls.from_pretrained(pipeline_file_path, scheduler=pipeline_scheduler)
    for name in ("unet", "vae", "vqvae"):
        component = getattr(pipeline, name, None)
        if component is not None:
            component.to(device)

    encoder = getattr(pipeline, "vqvae", None) or getattr(pipeline, "vae", None)

    # Each rank generates its own contiguous slice of sample indices.
    global_rank = accelerator.process_index
    samples_per_device = num_samples // accelerator.num_processes
    generator = torch.Generator(device=device).manual_seed(args.seed + global_rank)

    sampler = DistributedSampler(train_dataset, num_replicas=accelerator.num_processes,
                                 rank=global_rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(train_dataset, sampler=sampler,
                            batch_size=args.eval_batch_size, pin_memory=True)

    hallu_indices, lq_indices, correct_indices = [], [], []
    hallu_counts, correct_counts = [], []
    mse_frames = []
    accum = 0

    for batch in dataloader:
        clean = encode_batch(encoder, batch, diffusion_method, device)
        bsz = clean.size(0)
        offset = global_rank * samples_per_device + accum
        global_indices = list(range(offset, offset + bsz))
        accum += bsz

        accelerator.wait_for_everyone()
        out = pipeline(
            generator=generator,
            batch_size=bsz,
            num_inference_steps=args.num_inference_timesteps,
            output_type="pil",
            sampling_start=args.initial_noise,
            x_0=clean,
            ddpm_noise_scheduler=ddpm_noise_scheduler,
        )

        mse_frames.append(mse_list_to_df(out.intermediate_mse, global_indices))
        save_initial_noise_to_png(
            initial_noise=out.initial_noise,
            diffused_noise=out.diffused_noise,
            diffused_noise_save_path=diffused_noise_path,
            initial_noise_save_path=initial_noise_path,
            global_indices=global_indices,
        )
        save_gen_images_to_png(out.final_images, gen_img_path, global_indices)

        b_hallu, b_lq, b_hallu_counts, b_correct_counts = quantifier(
            img_path=gen_img_path, global_indices=global_indices,
        )
        failed = set(b_hallu + b_lq)
        hallu_indices += b_hallu
        lq_indices += b_lq
        correct_indices += [i for i in global_indices if i not in failed]
        hallu_counts += b_hallu_counts
        correct_counts += b_correct_counts

    accelerator.wait_for_everyone()

    # Gather per-rank python lists (concatenated across ranks).
    def gather(values):
        return accelerator.gather_for_metrics(values, use_gather_object=True)

    hallu_indices = gather(hallu_indices)
    lq_indices = gather(lq_indices)
    correct_indices = gather(correct_indices)
    hallu_counts = gather(hallu_counts)
    correct_counts = gather(correct_counts)

    mse_df_local = pd.concat(mse_frames, ignore_index=True)[MSE_COLUMNS]
    mse_tensor = torch.from_numpy(mse_df_local.to_numpy(dtype=np.float32)).to(device)
    mse_gathered = accelerator.gather_for_metrics(mse_tensor)

    if accelerator.is_main_process:
        mse_df = pd.DataFrame(mse_gathered.cpu().numpy(), columns=MSE_COLUMNS)
        mse_df["Sample Index"] = mse_df["Sample Index"].astype(int)
        mse_df["Time Step"] = mse_df["Time Step"].astype(int)
        hallu_df = mse_df[mse_df["Sample Index"].isin(hallu_indices)]
        correct_df = mse_df[mse_df["Sample Index"].isin(correct_indices)]
        lq_df = mse_df[mse_df["Sample Index"].isin(lq_indices)]

        # Prior gap between diffused noise and the sampling prior, then clean up.
        diffusion_prior_gap_fid = cal_diffusion_prior_gap_fid(
            diffused_noise_path=diffused_noise_path,
            initial_noise_path=initial_noise_path,
        )
        shutil.rmtree(diffused_noise_path)
        shutil.rmtree(initial_noise_path)

        fid_value = calculate_fid_given_paths(
            paths=[train_img_path, gen_img_path], batch_size=16, device=device)
        precision, recall = calculate_pr_given_paths(
            paths=[train_img_path, gen_img_path], batch_size=16, device=device)
        is_value, _ = calculate_is_given_path(
            path=gen_img_path, batch_size=16, device=device)

        if keep_visuals:
            categorize_and_copy_images_by_indices(
                all_img_path=gen_img_path,
                lq_img_path=lq_img_path,
                count_correct_img_path=correct_img_path,
                count_hallu_img_path=hallu_img_path,
                lq_global_indices=lq_indices,
                count_correct_global_indices=correct_indices,
                count_hallu_global_indices=hallu_indices,
                count_correct_pred_counting_labels=correct_counts,
                count_hallu_pred_counting_labels=hallu_counts,
            )
        else:
            shutil.rmtree(gen_img_path)
            shutil.rmtree(lq_img_path)
            shutil.rmtree(os.path.join(results_save_path, "visual_success_samples"))

        mse_stats = compute_mse(results_save_path, mse_df, hallu_df, correct_df, lq_df)
        hallu_rate = len(hallu_indices) / num_samples
        lq_rate = len(lq_indices) / num_samples

        with open(os.path.join(results_save_path, "results.txt"), "w") as f:
            f.write(f"Total generated samples: {num_samples}\n")
            f.write(f"Counting hallucination rate: {hallu_rate}\n")
            f.write(f"Counting hallucination counts: {len(hallu_indices)}\n")
            f.write(f"Visual failure rate: {lq_rate}\n")
            f.write(f"Visual failure counts: {len(lq_indices)}\n")
            f.write(f"Total failure rate: {lq_rate + hallu_rate}\n")
            f.write(f"Total failure counts: {len(lq_indices) + len(hallu_indices)}\n")
            f.write(f"FID: {fid_value}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Inception score: {is_value}\n")
            f.write(f"Diffusion prior gap (fid): {diffusion_prior_gap_fid}\n")
            for name, label in (("count_hallu", "counting hallucinations"),
                                ("count_correct", "count-correct samples"),
                                ("lq", "visual failure samples"),
                                ("all", "all samples")):
                if f"{name}_mean_each_t" not in mse_stats:
                    continue
                f.write(f"Initial mse of {label}: {mse_stats[f'{name}_mean_each_t'][0]:.6f}\n")
                f.write(f"Final mse of {label}: {mse_stats[f'{name}_mean_each_t'][-1]:.6f}\n")
                f.write(f"Average mse across all time steps for {label}: "
                        f"{mse_stats[f'{name}_mean_all']:.6f}\n")

        for name, df in (("count_hallu", hallu_df), ("count_correct", correct_df),
                         ("lq", lq_df), ("all", mse_df)):
            df = df.sort_values(by="Sample Index", kind="stable").reset_index(drop=True)
            df.to_csv(os.path.join(results_save_path, f"{name}_mse_dfs.csv"), index=False)
        np.save(os.path.join(results_save_path, "mse_dict.npy"), mse_stats)

        accelerator.log({
            "count_hallucination_rate": hallu_rate,
            "low_quality_rate": lq_rate,
            "total_samples": num_samples,
        })

    accelerator.end_training()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    # Core experiment settings.
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Registered dataset name (selects the eval config).")
    parser.add_argument("--sampling_solver", type=str, required=True,
                        help="Sampler: ddpm, dpm-1, dpm-2, dpm-plus, ddpm-gt, ddim-gt.")
    parser.add_argument("--initial_noise", type=str, required=True,
                        choices=["normal", "diffused"],
                        help="Sampling start: pure noise or diffused training samples.")
    parser.add_argument("--num_inference_timesteps", type=int, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Eval config; defaults to ./config/eval/<dataset_name>.yaml.")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--visual_seed", type=int, default=111,
                        help="Seed whose run keeps sorted images and detection overlays.")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_root", type=str, default="./results")

    # Local paths (usually provided by the eval config).
    parser.add_argument("--diffusion_pipeline_path", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--counting_model_path", type=str, default=None)
    parser.add_argument("--quality_cls_model_path", type=str, default=None)

    # Hub alternatives.
    parser.add_argument("--use_hub_model", action="store_true")
    parser.add_argument("--hub_diffusion_pipeline_id", type=str, default=None)
    parser.add_argument("--hub_counting_model_id", type=str, default=None)
    parser.add_argument("--hub_quality_cls_model_id", type=str, default=None)
    parser.add_argument("--use_hub_dataset", action="store_true")
    parser.add_argument("--hub_dataset_id", type=str, default=None)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
