"""Evaluate counting hallucinations of text-to-image models.

Covers two kinds of pipelines through ``--pipeline``:
  * sd:   a finetuned SD-1.5 pipeline saved by train_t2i.py
          (JDM checkpoints also return generated masks);
  * auto: any off-the-shelf diffusers text-to-image model
          (``--model`` accepts a Hub id or an alias from MODEL_ALIASES).

Example:
    python -m counthallu.eval.eval_t2i --pipeline sd \\
        --model /path/to/final-step-30000 --save_root ./results/t2i \\
        --counting_model_path ... --quality_cls_model_path ... \\
        --solver dpm-1 --steps 100 --num_samples 5050 --img_size 512
"""

import argparse
import os
import shutil

import torch

from counthallu.metrics.hallucination import CountHalluQuantifier
from counthallu.utils import (
    categorize_and_copy_images_by_indices,
    load_counting_model,
    load_quality_cls_model,
    seed_all,
)

# Convenience aliases for off-the-shelf models (--pipeline auto).
MODEL_ALIASES = {
    "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-2.1": "stabilityai/stable-diffusion-2-1",
    "sd-3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd-3.5-medium": "stabilityai/stable-diffusion-3.5-medium",
    "sd-3.5-large": "stabilityai/stable-diffusion-3.5-large",
    "flux": "black-forest-labs/FLUX.1-dev",
}

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_pipeline(args, dtype, device):
    if args.pipeline == "sd":
        from counthallu.models.t2i_pipelines import SDPipeline
        return SDPipeline.from_pretrained(
            args.model, torch_dtype=dtype, use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)

    if args.pipeline == "auto":
        from diffusers import AutoPipelineForText2Image
        model_id = MODEL_ALIASES.get(args.model, args.model)
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
        ).to(device)
        if getattr(pipeline, "safety_checker", None) is not None:
            pipeline.safety_checker = None
        return pipeline

    raise ValueError(f"Unknown pipeline kind: {args.pipeline}")


def apply_solver(pipeline, solver):
    """Swap the sampling scheduler, keeping the pipeline's base config."""
    if solver == "default":
        return
    from diffusers import DPMSolverSinglestepScheduler, FlowMatchHeunDiscreteScheduler
    config = dict(pipeline.scheduler.config)
    if solver in ("dpm-1", "dpm-2"):
        config["solver_order"] = int(solver[-1])
        pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(config)
    elif solver == "flow-heun":  # for flow-matching models such as SD 3.5
        pipeline.scheduler = FlowMatchHeunDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown solver: {solver}")


def read_prompts(args):
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_file}")
        return prompts
    return [args.prompt]


def main(args):
    seed_all(args.seed)
    dtype = DTYPES[args.dtype]
    device = args.device

    pipeline = load_pipeline(args, dtype, device)
    apply_solver(pipeline, args.solver)
    prompts = read_prompts(args)

    counting_model, counting_model_type, reference_counts, target_classes = load_counting_model(
        dataset_name=args.dataset_name,
        model_path=args.counting_model_path,
        device=device,
        use_hub_model=args.use_hub_model,
        repo_id=args.hub_counting_model_id,
    )
    quality_cls_model = load_quality_cls_model(
        dataset_name=args.dataset_name,
        model_path=args.quality_cls_model_path,
        device=device,
        use_hub_model=args.use_hub_model,
        repo_id=args.hub_quality_cls_model_id,
    )
    quantifier = CountHalluQuantifier(
        counting_model=counting_model,
        counting_model_type=counting_model_type,
        device=device,
        reference_counts_list=reference_counts,
        quality_cls_model=quality_cls_model,
        target_class_indices_list=target_classes,
        save_detection_results=args.save_detections,
    )

    experiment_root = os.path.join(args.save_root, f"{args.solver}-{args.steps}_steps")
    shutil.rmtree(experiment_root, ignore_errors=True)
    gen_img_path = os.path.join(experiment_root, "gen_images")
    gen_mask_path = os.path.join(experiment_root, "gen_masks")
    lq_img_path = os.path.join(experiment_root, "visual_failure_samples")
    correct_img_path = os.path.join(experiment_root, "visual_success_samples",
                                    "counting_correct_samples")
    hallu_img_path = os.path.join(experiment_root, "visual_success_samples",
                                  "counting_hallucinations")
    for path in (gen_img_path, lq_img_path, correct_img_path, hallu_img_path):
        os.makedirs(path, exist_ok=True)

    total = args.num_samples * len(prompts)
    print(f"Generating {len(prompts)} prompt(s) x {args.num_samples} samples = {total} images"
          f" -> {experiment_root}")

    hallu_indices, lq_indices, correct_indices = [], [], []
    hallu_counts, correct_counts = [], []
    global_idx = 0

    for prompt in prompts:
        remaining = args.num_samples
        while remaining > 0:
            bsz = min(args.batch_size, remaining)
            remaining -= bsz

            out = pipeline(
                prompt=[prompt] * bsz,
                num_inference_steps=args.steps,
                height=args.img_size,
                width=args.img_size,
            )
            masks = getattr(out, "masks", None)
            batch_indices = list(range(global_idx, global_idx + bsz))
            global_idx += bsz

            for idx, image in zip(batch_indices, out.images):
                image.save(os.path.join(gen_img_path, f"{idx}.png"))
            if masks is not None:  # JDM checkpoints also emit masks
                os.makedirs(gen_mask_path, exist_ok=True)
                for idx, mask in zip(batch_indices, masks):
                    mask.save(os.path.join(gen_mask_path, f"{idx}.png"))

            b_hallu, b_lq, b_hallu_counts, b_correct_counts = quantifier(
                img_path=gen_img_path, global_indices=batch_indices,
            )
            failed = set(b_hallu + b_lq)
            hallu_indices += b_hallu
            lq_indices += b_lq
            correct_indices += [i for i in batch_indices if i not in failed]
            hallu_counts += b_hallu_counts
            correct_counts += b_correct_counts

            if global_idx % 50 < bsz:
                print(f"[{args.solver} | {args.steps} steps] {global_idx}/{total} images...")

    print("Categorizing images...")
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

    hallu_rate = len(hallu_indices) / total
    lq_rate = len(lq_indices) / total
    results_file = os.path.join(experiment_root, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Solver: {args.solver}\n")
        f.write(f"Num inference steps: {args.steps}\n")
        f.write(f"Total generated samples: {total}\n")
        f.write(f"Counting hallucination rate: {hallu_rate:.4f}\n")
        f.write(f"Counting hallucination counts: {len(hallu_indices)}\n")
        f.write(f"Visual failure rate: {lq_rate:.4f}\n")
        f.write(f"Visual failure counts: {len(lq_indices)}\n")
        f.write(f"Total failure rate: {hallu_rate + lq_rate:.4f}\n")
        f.write(f"Total failure counts: {len(hallu_indices) + len(lq_indices)}\n")
    print(f"Done. Results saved to {results_file}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=str, required=True, choices=["sd", "auto"])
    parser.add_argument("--model", type=str, required=True,
                        help="Pipeline path (sd) or model id / alias (auto).")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--prompt", type=str, default="A single human hand.")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Text file with one prompt per line (overrides --prompt).")
    parser.add_argument("--num_samples", type=int, default=5050,
                        help="Images generated per prompt.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--solver", type=str, default="default",
                        choices=["default", "dpm-1", "dpm-2", "flow-heun"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bf16", choices=list(DTYPES))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--save_root", type=str, required=True)

    parser.add_argument("--dataset_name", type=str, default="realhand")
    parser.add_argument("--counting_model_path", type=str, default=None)
    parser.add_argument("--quality_cls_model_path", type=str, default=None)
    parser.add_argument("--use_hub_model", action="store_true")
    parser.add_argument("--hub_counting_model_id", type=str, default=None)
    parser.add_argument("--hub_quality_cls_model_id", type=str, default=None)
    parser.add_argument("--save_detections", action="store_true",
                        help="Also save YOLO detection overlays.")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
