"""
Abstract base class for GAN architectures.

All concrete GAN implementations (DCGAN, WCGAN, etc.) should inherit from this class
and implement the abstract methods. Shared hyperparameters and best-model saving logic
live here so they are never duplicated across architectures.
"""

import json
import torch
import lpips as lps

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from utils.config import config
from utils.types import Hyperparams
from utils.types import AvailableMetrics, Metrics

# ---------------------------------------------------------------------------
# Shared / default hyperparameters — every GAN subclass starts from this dict.
# Architecture-specific keys (e.g. LAMBDA_GP, N_CRITIC) should be added by the
# subclass's own DEFAULT_CONFIG, which is then merged on top of these values.
# ---------------------------------------------------------------------------
SHARED_CONFIG: Hyperparams = {
    "BATCH_SIZE": config.batch_size,
    "NUM_EPOCHS": config.epochs,
    "LR": 2e-4,
    "IMG_SIZE": config.img_size,
    "IMG_CHANNELS": config.img_channels,
    "NOISE_DIM": 128,
    "NGF": 64,
    "NDF": 64,
    "LABEL_EMB_DIM": 50,
    "LEAKY_RELU_SLOPE": 0.2,
    "ADAM_BETA1": 0.5,
    "ADAM_BETA2": 0.9,
}


class GAN(ABC):
    """
    Abstract GAN wrapper.

    Subclasses must:
      - Define a class-level DEFAULT_CONFIG dict (merged on top of SHARED_CONFIG).
      - Implement `build_models()` to create and return (generator, discriminator).
      - Implement `train_epoch()` to run one epoch and return (d_loss, g_loss).
      - Implement `evaluate()` to return a metrics dict.
      - Implement `augment()` to generate and persist augmented images.

    The base class handles:
      - Config merging and hyperparameter overrides.
      - Device selection.
      - Best-model tracking and saving (by FID, PSNR, LPIPS, SSIM, etc).
      - Checkpoint resume logic.
      - MetricsLogger lifecycle.
    """

    # Override in subclass with architecture-specific defaults.
    DEFAULT_CONFIG: Hyperparams = {}

    def __init__(
        self,
        dataset_obj,
        validate_obj,
        output_path: Path,
        hyperparams: Optional[Hyperparams] = None,
        resume: bool = False,
        metrics_to_calculate: Optional[List[AvailableMetrics | str]] = None,
    ):
        """
        Args:
            dataset_obj:   Pre-processed ImageDataset for training.
            validate_obj:  Pre-processed ImageDataset for validation.
            output_path:   Directory for checkpoints, best model, and logs.
            hyperparams:   Optional overrides for any config key.
            resume:        Whether to resume from an existing checkpoint.
            metrics_to_calculate: List of metric names to calculate during evaluation.
        """
        # Merge: shared defaults <- architecture defaults <- caller overrides
        self.training_config: Dict[str, Any] = {
            **SHARED_CONFIG,
            **self.__class__.DEFAULT_CONFIG,
        }
        if hyperparams:
            self.training_config.update(hyperparams)

        self.dataset_obj = dataset_obj
        self.validate_obj = validate_obj
        self.output_path = Path(output_path)
        self.resume = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Dispositivo: {self.device}")
        self.num_classes: int = len(dataset_obj.image_data.classes)
        self.data_loader = dataset_obj.data_loader
        self._best_epoch: int = -1
        # Built lazily in run()
        self.generator: Optional[torch.nn.Module] = None
        self.discriminator: Optional[torch.nn.Module] = None
        raw_metrics = metrics_to_calculate or [AvailableMetrics.FID]
        self.metrics_to_calculate = [
            self._normalize_metric(metric) for metric in raw_metrics
        ]

        if AvailableMetrics.LPIPS in self.metrics_to_calculate:
            self.lpips_model = lps.LPIPS(net="alex").to(self.device)

        self.save_best_model_metric = self.metrics_to_calculate[0]
        self._best_metric_value: float = (
            float("inf")
            if self.save_best_model_metric
            in [AvailableMetrics.FID, AvailableMetrics.LPIPS]
            else float("-inf")
        )

    @staticmethod
    def _normalize_metric(metric: AvailableMetrics | str) -> AvailableMetrics:
        if isinstance(metric, AvailableMetrics):
            return metric

        normalized = metric.strip()
        if not normalized:
            raise ValueError("Metric name cannot be empty.")

        member = AvailableMetrics.__members__.get(normalized.upper())
        if member is not None:
            return member

        for available_metric in AvailableMetrics:
            if available_metric.value.lower() == normalized.lower():
                return available_metric

        raise ValueError(f"Unknown metric '{metric}'.")

    @staticmethod
    def _get_metric_value(
        metrics: Metrics,
        metric: AvailableMetrics,
        default: float,
    ) -> float:
        for key, value in metrics.items():
            if isinstance(key, AvailableMetrics) and key == metric:
                return float(value)

            if isinstance(key, str) and key.lower() in (
                metric.name.lower(),
                metric.value.lower(),
            ):
                return float(value)

        return default

    @staticmethod
    def _serialize_metrics(metrics: Metrics) -> dict[str, float]:
        serialized: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(key, AvailableMetrics):
                serialized[key.value] = float(value)
            else:
                serialized[str(key)] = float(value)
        return serialized

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def build_models(self):
        """
        Instantiate and return (generator, discriminator) on self.device.
        Weight initialisation should also happen here.
        """
        ...

    @abstractmethod
    def build_optimizers(self):
        """
        Instantiate and return (g_optimizer, d_optimizer) using self.training_config.
        Called after build_models().
        """
        ...

    @abstractmethod
    def train_epoch(self, g_optimizer, d_optimizer) -> Dict[str, float]:
        """
        Run one full training epoch.

        Returns:
            A dict with at least {"d_loss": float, "g_loss": float}.
            Additional scalar metrics (e.g. gradient_penalty) are welcome.
        """
        ...

    @abstractmethod
    def evaluate(self) -> Metrics:
        """
        Evaluate the current generator against the validation set.

        Returns:
            A dict including at least {"fid": float}.  Additional metrics
            (psnr, ssim, lpips, …) are encouraged but optional.
        """
        ...

    @abstractmethod
    def augment(
        self,
        output_path: Path,
        gan_training_path: Path,
        num_augmented_images: int = 1000,
    ) -> None:
        """
        Load a trained generator from gan_training_path and generate
        num_augmented_images images, saving them to output_path via
        dataset_obj.save_generated_images_to_disk().
        """
        ...

    # ------------------------------------------------------------------
    # Shared logic — do not override unless you have a strong reason
    # ------------------------------------------------------------------

    def _checkpoint_paths(self, suffix: str = ""):
        """Return (generator_path, discriminator_path) for a given suffix."""
        base = self.output_path
        tag = f"_{suffix}" if suffix else ""
        return base / f"generator{tag}.pth", base / f"discriminator{tag}.pth"

    def save_checkpoint(self, suffix: str = ""):
        """Save generator + discriminator weights with optional filename suffix."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        g_path, d_path = self._checkpoint_paths(suffix)
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)

    @staticmethod
    def resolve_generator_checkpoint_path(gan_training_path: Path) -> Path:
        generator_best = gan_training_path / "generator_best.pth"
        if generator_best.exists():
            return generator_best

        generator_latest = gan_training_path / "generator.pth"
        if generator_latest.exists():
            return generator_latest

        raise FileNotFoundError(
            f"No generator checkpoint found in '{gan_training_path}'. "
            "Expected 'generator_best.pth' or 'generator.pth'."
        )

    def load_checkpoint(self, suffix: str = ""):
        """Load generator + discriminator weights if the files exist."""
        g_path, d_path = self._checkpoint_paths(suffix)
        if g_path.exists() and d_path.exists():
            self.generator.load_state_dict(
                torch.load(g_path, map_location=self.device, weights_only=False)
            )
            self.discriminator.load_state_dict(
                torch.load(d_path, map_location=self.device, weights_only=False)
            )
            print(
                f"[{self.__class__.__name__}] Checkpoint '{suffix or 'latest'}' carregado."
            )
        else:
            print(
                f"[{self.__class__.__name__}] Nenhum checkpoint encontrado em {self.output_path}."
            )

    def save_best_model(self, metrics: Metrics, epoch: int) -> bool:
        """
        Compare current metrics against the best so far, update if improved, save the
        generator/discriminator as 'best' checkpoints and persist a JSON summary.

        Args:
            metrics:  Dict returned by evaluate()
            epoch:    Current epoch (1-indexed).

        Returns:
            True if a new best was recorded, False otherwise.
        """
        if self.save_best_model_metric == AvailableMetrics.FID:
            fid = self._get_metric_value(metrics, AvailableMetrics.FID, float("inf"))
            if fid < self._best_metric_value:
                self._best_metric_value = fid
                self._best_epoch = epoch
                self.save_checkpoint(suffix="best")

                summary = {"epoch": epoch, **self._serialize_metrics(metrics)}
                best_json_path = self.output_path / "best_metrics.json"
                with open(best_json_path, "w") as f:
                    json.dump(summary, f, indent=4)

                return True

        elif self.save_best_model_metric == AvailableMetrics.PSNR:
            psnr = self._get_metric_value(metrics, AvailableMetrics.PSNR, float("-inf"))
            if psnr > self._best_metric_value:  # Higher PSNR is better
                self._best_metric_value = psnr
                self._best_epoch = epoch
                self.save_checkpoint(suffix="best")

                summary = {"epoch": epoch, **self._serialize_metrics(metrics)}
                best_json_path = self.output_path / "best_metrics.json"
                with open(best_json_path, "w") as f:
                    json.dump(summary, f, indent=4)

                return True

        elif self.save_best_model_metric == AvailableMetrics.SSIM:
            ssim = self._get_metric_value(metrics, AvailableMetrics.SSIM, float("-inf"))
            if ssim > self._best_metric_value:  # Higher SSIM is better
                self._best_metric_value = ssim
                self._best_epoch = epoch
                self.save_checkpoint(suffix="best")

                summary = {"epoch": epoch, **self._serialize_metrics(metrics)}
                best_json_path = self.output_path / "best_metrics.json"
                with open(best_json_path, "w") as f:
                    json.dump(summary, f, indent=4)

                return True

        elif self.save_best_model_metric == AvailableMetrics.LPIPS:
            lpips = self._get_metric_value(
                metrics, AvailableMetrics.LPIPS, float("inf")
            )
            if lpips < self._best_metric_value:
                self._best_metric_value = lpips
                self._best_epoch = epoch
                self.save_checkpoint(suffix="best")

                summary = {"epoch": epoch, **self._serialize_metrics(metrics)}
                best_json_path = self.output_path / "best_metrics.json"
                with open(best_json_path, "w") as f:
                    json.dump(summary, f, indent=4)

                return True
        elif self.save_best_model_metric == AvailableMetrics.WASSERSTEIN_DISTANCE:
            w_dist = self._get_metric_value(
                metrics, AvailableMetrics.WASSERSTEIN_DISTANCE, float("-inf")
            )
            if w_dist > self._best_metric_value:
                self._best_metric_value = w_dist
                self._best_epoch = epoch
                self.save_checkpoint(suffix="best")

                summary = {"epoch": epoch, **self._serialize_metrics(metrics)}
                best_json_path = self.output_path / "best_metrics.json"
                with open(best_json_path, "w") as f:
                    json.dump(summary, f, indent=4)

                return True
        else:
            print(
                "Could not determine which metric to use for best model saving. No checkpoint saved."
            )
        return False

    def run(self):
        """
        Full training loop shared across all GAN subclasses.

        Workflow per epoch:
          1. train_epoch()  → losses
          2. evaluate()     → metrics dict
          3. save_best_model() if metric improved → saves 'best' checkpoints + JSON summary
          4. Periodic checkpoint every CHECKPOINT_EVERY epochs (default: 50)
        """
        from utils.metrics import MetricsLogger, format_metrics

        self.generator, self.discriminator = self.build_models()
        g_optimizer, d_optimizer = self.build_optimizers()

        if self.resume:
            self.load_checkpoint()

        checkpoint_every: int = self.training_config.get("CHECKPOINT_EVERY", 50)
        num_epochs: int = self.training_config["NUM_EPOCHS"]

        metrics_logger = MetricsLogger(
            model_name=self.__class__.__name__,
            hyperparams=self.training_config,
        )

        self.output_path.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(
            range(num_epochs),
            desc=f"{self.__class__.__name__}",
            unit="epoch",
        )
        pbar.set_postfix(
            {
                "loss": "N/A",
                **{metric.value: "N/A" for metric in self.metrics_to_calculate},
            }
        )

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(g_optimizer, d_optimizer)
            eval_metrics = self.evaluate()

            all_metrics = {"epoch": epoch, **train_metrics, **eval_metrics}
            metrics_logger.log(**all_metrics)
            # config.logger.info(
            #     "[Metrics][GAN][%s] %s",
            #     self.__class__.__name__,
            #     format_metrics(all_metrics),
            # )

            self.save_best_model(eval_metrics, epoch)

            if epoch % checkpoint_every == 0 or epoch == num_epochs:
                self.save_checkpoint()

            pbar.update(1)
            pbar.set_postfix(
                {
                    "g_loss": f"{train_metrics.get('g_loss', float('nan')):.4f}",
                    "d_loss": f"{train_metrics.get('d_loss', float('nan')):.4f}",
                    # every subscribed metric
                    **{
                        metric.value: f"{self._get_metric_value(all_metrics, metric, float('nan')):.4f}"
                        for metric in self.metrics_to_calculate
                    },
                }
            )

        # Final checkpoint (already saved above if epoch == num_epochs)
        metrics_logger.close()
        pbar.close()

        print(
            f"\n[{self.__class__.__name__}] Training concluded. "
            f"{self.metrics_to_calculate[0].value}={self._best_metric_value:.4f} at epoch {self._best_epoch}."
        )
