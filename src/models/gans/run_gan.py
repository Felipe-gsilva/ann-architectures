"""
Entrypoint for GAN training and augmentation.

Supports multiple GAN architectures via --gan flag.
All architectures share the same hyperparameter space and config system.

Usage examples:
    python gan.py --train --gan dcgan
    python gan.py --train --gan wcgan
    python gan.py --train          # trains all registered GANs
    python gan.py --generate --gan dcgan --num_aug 2000
    python gan.py --scheduler --gan wcgan
"""

import os
import logging

from utils.types import Hyperparams, AvailableMetrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_HYPERPARAM_LIST = [
    {"LR": 2e-4, "BATCH_SIZE": 32, "NUM_EPOCHS": 256},
    # {"LR": 1e-4, "BATCH_SIZE": 32, "NUM_EPOCHS": 512},
]

import gc
import time
import torch

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional
from gans import DCGAN, WCGAN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry: add new GAN modules here — no other changes needed
# ---------------------------------------------------------------------------
GAN_REGISTRY = {
    "dcgan": DCGAN,
    "wcgan": WCGAN,
}


def _resolve_modules(gan_name: str | None) -> list:
    """Return list of (name, module) pairs for the requested GAN(s)."""
    if gan_name is None:
        return list(GAN_REGISTRY.items())
    key = gan_name.lower()
    if key not in GAN_REGISTRY:
        raise ValueError(
            f"Unknown GAN '{gan_name}'. Available: {list(GAN_REGISTRY.keys())}"
        )
    return [(key, GAN_REGISTRY[key])]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def gan_training(
    resume: bool,
    gan_name: str | None,
    datasets: list[str],
    metrics: List[AvailableMetrics],
    modules: Optional[List] = None,
):
    """Train the selected GAN(s) for every dataset."""
    if modules is None:
        modules = _resolve_modules(gan_name)

    for dataset_name in datasets:
        base_path = Path(f"assets/data/baseline/{dataset_name}")
        input_path = (
            base_path / "images" if (base_path / "images").exists() else base_path
        )
        train_ds, _, validate_ds = preprocess(
            dataset_path=input_path,
            name=dataset_name,
            step="baseline",
            image_size=config.img_size,
        )

        for name, module in modules:
            output_path = Path(
                f"assets/data/gan_training/{dataset_name}/{name.upper()}"
            )
            logger.info(f"Training {name.upper()} for {dataset_name}.")
            module.train(
                dataset_obj=train_ds,
                validate_ds=validate_ds,
                output_path=output_path,
                resume=resume,
                metrics=metrics,
            )

            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def gan_augmentation(
    num_aug: int,
    gan_name: str | None,
    datasets: list[str],
    modules: Optional[list] = None,
):
    """Generate augmented images using trained GAN(s)."""
    if modules is None:
        modules = _resolve_modules(gan_name)

    for dataset_name in datasets:
        base_path = Path(f"assets/data/baseline/{dataset_name}")
        input_path = (
            base_path / "images" if (base_path / "images").exists() else base_path
        )
        gan_base = Path(f"assets/data/gan_training/{dataset_name}/")
        output_base = Path(f"assets/data/gan_aug/{dataset_name}/")

        train_ds, _, _ = preprocess(
            dataset_path=input_path,
            name=dataset_name,
            step="baseline",
            image_size=config.img_size,
        )
        count = num_aug if num_aug > 0 else len(train_ds.image_data)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        for name, module in modules:
            logger.info(
                f"Augmenting {dataset_name} with {name.upper()}: {count} images."
            )
            module.augment(
                dataset_obj=train_ds,
                output_path=output_base / f"{name.upper()}_{timestamp}",
                gan_training_path=gan_base / name.upper(),
                num_augmented_images=count,
            )

        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Grid Search / Hyperparameter Tuning
# ---------------------------------------------------------------------------


def run_hyperparameter_search(
    hyperparams_list: List[Hyperparams],
    gan_name: str | None,
    metrics: List[AvailableMetrics],
    num_aug: int = 1000,
    modules: Optional[list] = None,
):
    """Run sequential training with different hyperparameters."""
    if modules is None:
        modules = _resolve_modules(gan_name)

    logger.info(
        f"Starting hyperparameter search: {len(hyperparams_list)} configs × {len(modules)} GAN(s)."
    )

    for i, hyperparams in enumerate(hyperparams_list):
        logger.info(f"\n--- Config {i + 1}/{len(hyperparams_list)} --- {hyperparams}")
        for dataset in config.datasets_name_list:
            base_path = Path(f"assets/data/baseline/{dataset}")
            input_path = (
                base_path / "images" if (base_path / "images").exists() else base_path
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            train_ds, _, validate_ds = preprocess(
                dataset_path=input_path,
                name=dataset,
                step="baseline",
                image_size=config.img_size,
            )
            count = num_aug or len(train_ds.image_data)

            for name, module in modules:
                try:
                    train_out = Path(
                        f"assets/data/gan_training/{dataset}/{name.upper()}_{timestamp}"
                    )
                    module.train(
                        dataset_obj=train_ds,
                        validate_ds=validate_ds,
                        output_path=train_out,
                        resume=False,
                        hyperparams=hyperparams,
                        metrics=metrics,
                    )

                    aug_out = Path(
                        f"assets/data/gan_aug/{dataset}/{name.upper()}_{timestamp}"
                    )
                    module.augment(
                        dataset_obj=train_ds,
                        output_path=aug_out,
                        gan_training_path=train_out,
                        num_augmented_images=count,
                        config=hyperparams,
                    )
                    # clean up GPU memory after each run
                    torch.cuda.empty_cache()
                    gc.collect()

                except torch.cuda.OutOfMemoryError as e:
                    import traceback

                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.error(
                        f"Error — {name.upper()} config {i + 1} on {dataset}: {e}"
                    )
                    traceback.print_exc()

                except Exception as e:
                    import traceback

                    logger.error(
                        f"Error — {name.upper()} config {i + 1} on {dataset}: {e}"
                    )
                    traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()


def run_gan(
    should_train: bool = False,
    should_generate: bool = False,
    should_hyperparams_search: bool = False,
    resume: bool = False,
    gan: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[AvailableMetrics]] = None,
    num_aug: int = 1000,
    hyperparams_list: Optional[List[Hyperparams]] = None,
):
    if hyperparams_list is None:
        hyperparams_list = DEFAULT_HYPERPARAM_LIST

    if datasets is None:
        datasets = config.datasets_name_list
    if metrics is None:
        metrics = [AvailableMetrics.FID]

    if not any([should_train, should_generate, should_hyperparams_search]):
        logger.warning("No action specified. Use --train, --generate, or --scheduler.")
        return

    modules = _resolve_modules(gan)

    if should_train:
        gan_training(resume, gan, datasets, metrics, modules)
    if should_generate:
        gan_augmentation(num_aug, gan, datasets, modules)
    if should_hyperparams_search:
        run_hyperparameter_search(hyperparams_list, gan, metrics, num_aug, modules)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="GAN training / augmentation entrypoint")
    parser.add_argument("--train", action="store_true", help="Train GAN(s)")
    parser.add_argument(
        "--generate", action="store_true", help="Generate augmented images"
    )
    parser.add_argument(
        "--scheduler", action="store_true", help="Run hyperparameter search"
    )
    parser.add_argument(
        "--gan",
        type=str,
        default=None,
        choices=list(GAN_REGISTRY.keys()),
        help="Which GAN architecture to use. Omit to run all registered GANs.",
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=1000,
        help="Number of augmented images to generate per dataset (--generate only)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=config.datasets_name_list,
        help="List of datasets to process (default: all in config)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["FID"],
        help="Metrics to evaluate during hyperparameter search (default: ['FID'])",
        choices=AvailableMetrics._member_names_,
    )
    args = parser.parse_args()

    run_gan(
        should_train=args.train,
        should_generate=args.generate,
        should_hyperparams_search=args.scheduler,
        resume=args.resume,
        gan=args.gan,
        datasets=args.datasets,
        metrics=args.metrics,
        num_aug=args.num_aug,
    )
