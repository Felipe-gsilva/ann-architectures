"""
Deep Convolutional Generative Adversarial Network (DCGAN) para geração condicional de imagens.

The Generator receives a noise vector and a class as input, and produces an image.
The Discriminator (Critic) receives an image and a class, and outputs a real-valued score (no sigmoid).

Inherits from GAN to share configuration, training loop, and best-model logic.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from src.dataset.ImageDataset import ImageDataset
from gans.base import GAN, SHARED_CONFIG
from validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips
from src.utils.types import AvailableMetrics, Hyperparams


# Architecture-specific defaults — merged on top of SHARED_CONFIG inside GAN.__init__
DEFAULT_CONFIG = {
    **SHARED_CONFIG,
    "NOISE_DIM": 128,
    "NGF": 64,
    "NDF": 64,
    "LABEL_EMB_DIM": 50,
    "LEAKY_RELU_SLOPE": 0.2,
    "ADAM_BETA1": 0.5,
    "ADAM_BETA2": 0.9,
    "LR": 2e-4,
}


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, img_size: int, cfg: dict = DEFAULT_CONFIG):
        super().__init__()
        self.nc = cfg["IMG_CHANNELS"]
        self.nf = cfg["NDF"]
        self.img_size = img_size

        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        self.network = nn.Sequential(
            nn.Conv2d(self.nc + 1, self.nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 8, self.nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 16),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(labels.size(0), 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_emb), 1)
        output = self.network(d_in)
        return output.view(output.shape[0], -1).mean(1)


class Generator(nn.Module):
    def __init__(self, num_classes: int, cfg: dict = DEFAULT_CONFIG):
        super().__init__()
        self.noise_dim = cfg["NOISE_DIM"]
        self.label_emb_dim = cfg["LABEL_EMB_DIM"]
        self.nf = cfg["NGF"]

        self.label_embedding = nn.Embedding(num_classes, self.label_emb_dim)

        init_size = cfg["IMG_SIZE"] // 16
        self.l1 = nn.Sequential(
            nn.Linear(self.noise_dim + self.label_emb_dim, self.nf * 8 * init_size**2)
        )
        self._init_size = init_size

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.nf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 8, self.nf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.nf * 4, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.nf * 2, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 2, self.nf, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.nf, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf, cfg["IMG_CHANNELS"], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.nf * 8, self._init_size, self._init_size)
        return self.conv_blocks(out)


# ---------------------------------------------------------------------------
# DCGAN class
# ---------------------------------------------------------------------------


class DCGANModel(GAN):
    """Conditional DCGAN using BCEWithLogitsLoss."""

    def build_models(self) -> Tuple[nn.Module, nn.Module]:
        cfg = self.training_config
        generator = Generator(self.num_classes, cfg).to(self.device)
        discriminator = Discriminator(self.num_classes, cfg["IMG_SIZE"], cfg).to(
            self.device
        )

        def weights_init(m):
            classname = m.__class__.__name__
            if "Conv" in classname:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif "BatchNorm" in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        generator.apply(weights_init)
        discriminator.apply(weights_init)
        return generator, discriminator

    def build_optimizers(self):
        cfg = self.training_config
        g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=cfg["LR"],
            betas=(cfg["ADAM_BETA1"], cfg["ADAM_BETA2"]),
        )
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg["LR"],
            betas=(cfg["ADAM_BETA1"], cfg["ADAM_BETA2"]),
        )
        return g_opt, d_opt

    def train_epoch(self, g_optimizer, d_optimizer) -> Dict[str, float]:
        cfg = self.training_config
        criterion = nn.BCEWithLogitsLoss()
        noise_dim: int = cfg["NOISE_DIM"]

        d_loss_sum = 0.0
        g_loss_sum = 0.0
        steps = 0

        for real_images, labels in self.data_loader:
            real_images = real_images.to(self.device)
            labels = labels.to(self.device)
            batch_size = real_images.size(0)

            # --- Discriminator ---
            self.discriminator.train()
            d_optimizer.zero_grad()

            real_validity = self.discriminator(real_images, labels)
            real_loss = criterion(real_validity, torch.ones_like(real_validity))

            z = torch.randn(batch_size, noise_dim).to(self.device)
            gen_labels = torch.randint(0, self.num_classes, (batch_size,)).to(
                self.device
            )
            fake_images = self.generator(z, gen_labels).detach()
            fake_validity = self.discriminator(fake_images, gen_labels)
            fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # --- Generator ---
            self.generator.train()
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, noise_dim).to(self.device)
            gen_labels = torch.randint(0, self.num_classes, (batch_size,)).to(
                self.device
            )
            fake_images = self.generator(z, gen_labels)
            validity = self.discriminator(fake_images, gen_labels)
            g_loss = criterion(validity, torch.ones_like(validity))
            g_loss.backward()
            g_optimizer.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            steps += 1

        return {"d_loss": d_loss_sum / steps, "g_loss": g_loss_sum / steps}

    def evaluate(
        self,
    ) -> Dict[str, float]:
        cfg = self.training_config
        noise_dim: int = cfg["NOISE_DIM"]

        self.generator.eval()
        self.discriminator.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        real_images_list, fake_images_list = [], []

        with torch.no_grad():
            for real_imgs, _ in self.validate_obj.data_loader:
                real_images_list.append(real_imgs.cpu())
                batch_size = real_imgs.size(0)
                z = torch.randn(batch_size, noise_dim).to(self.device)
                gen_labels = torch.randint(0, self.num_classes, (batch_size,)).to(
                    self.device
                )
                fake_imgs = self.generator(z, gen_labels)
                fake_images_list.append(fake_imgs.cpu())

        # [-1,1] -> [0, 255] uint8
        # pre-instantiate LPIPS model to avoid overhead during metric calculation
        real_cat = ((torch.cat(real_images_list) + 1) / 2 * 255).to(torch.uint8)
        fake_cat = ((torch.cat(fake_images_list) + 1) / 2 * 255).to(torch.uint8)

        if AvailableMetrics.FID in self.metrics_to_calculate:
            fid = calculate_fid(real_cat, fake_cat, batch_size=cfg["BATCH_SIZE"])
        else:
            fid = -1.0  # Sentinel value when FID is not calculated

        if any(
            m in self.metrics_to_calculate
            for m in [
                AvailableMetrics.PSNR,
                AvailableMetrics.SSIM,
                AvailableMetrics.LPIPS,
            ]
        ):
            self.lpips_model.to(self.device)
            psnr, ssim, lpips = calculate_psnr_ssim_lpips(
                real_cat,
                fake_cat,
                self.lpips_model,
                device=self.device,
                batch_size=cfg["BATCH_SIZE"],
                metrics_to_compute=[
                    metrics.value.lower() for metrics in self.metrics_to_calculate
                ],
            )
        else:
            psnr, ssim, lpips = (
                -1.0,
                -1.0,
                -1.0,
            )  # Sentinel values when metrics are not calculated

        return {"fid": fid, "psnr": psnr, "ssim": ssim, "lpips": lpips}

    def augment(
        self,
        output_path: Path,
        gan_training_path: Path,
        num_augmented_images: int = 1000,
    ) -> None:
        cfg = self.training_config
        noise_dim: int = cfg["NOISE_DIM"]
        batch_size: int = cfg["BATCH_SIZE"]

        generator = Generator(self.num_classes, cfg).to(self.device)
        generator_checkpoint = self.resolve_generator_checkpoint_path(gan_training_path)
        generator.load_state_dict(
            torch.load(
                generator_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
        )
        generator.eval()

        augmented_images, augmented_labels = [], []
        with torch.no_grad():
            while sum(t.shape[0] for t in augmented_images) < num_augmented_images:
                z = torch.randn(batch_size, noise_dim).to(self.device)
                labels = torch.randint(0, self.num_classes, (batch_size,)).to(
                    self.device
                )
                fake_imgs = generator(z, labels).cpu()
                augmented_images.append(fake_imgs)
                augmented_labels.append(labels.cpu())

        full_res = torch.cat(augmented_images)[:num_augmented_images]
        full_labels = torch.cat(augmented_labels)[:num_augmented_images]
        self.dataset_obj.save_generated_images_to_disk(
            output_path, full_res, full_labels
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward-compatible with gan.py)
# ---------------------------------------------------------------------------


def train(
    dataset_obj: ImageDataset,
    validate_ds: ImageDataset,
    output_path: Path,
    resume: bool = False,
    hyperparams: Optional[Hyperparams] = None,
    metrics: Optional[AvailableMetrics] = None,
) -> DCGANModel:
    model = DCGANModel(
        dataset_obj=dataset_obj,
        validate_obj=validate_ds,
        output_path=output_path,
        hyperparams=hyperparams,
        resume=resume,
        metrics_to_calculate=metrics,
    )
    model.run()
    return model


def augment(
    dataset_obj: ImageDataset,
    output_path: Path,
    gan_training_path: Path,
    num_augmented_images: int = 1000,
    config: Optional[dict] = None,
) -> None:
    model = DCGANModel(
        dataset_obj=dataset_obj,
        validate_obj=dataset_obj,  # validate_obj not needed for augmentation
        output_path=output_path,
        hyperparams=config,
    )
    model.augment(output_path, gan_training_path, num_augmented_images)
