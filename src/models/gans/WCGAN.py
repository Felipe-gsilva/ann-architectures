"""
Wasserstein Conditional GAN with Gradient Penalty (WCGAN-GP).

The Generator receives a noise vector and a class as input, and produces an image.
The Discriminator (Critic) receives an image and a class, and outputs a real-valued score (no sigmoid).

Inherits from GAN to share configuration, training loop, and best-model logic.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.autograd as autograd
import torch.nn as nn

from dataset.ImageDataset import ImageDataset
from gans.base import GAN, SHARED_CONFIG
from utils.types import AvailableMetrics, Hyperparams, Metrics
from validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips


# Architecture-specific defaults — merged on top of SHARED_CONFIG inside GAN.__init__
DEFAULT_CONFIG = {
    **SHARED_CONFIG,
    "LR": 1e-4,
    "IMG_CHANNELS": 3,
    "NOISE_DIM": 100,
    "NGF": 64,
    "NDF": 64,
    "LABEL_EMB_DIM": 50,
    "LAMBDA_GP": 10,  # Gradient penalty coefficient
    "N_CRITIC": 5,  # Discriminator steps per generator step
    "LEAKY_RELU_SLOPE": 0.2,
    "ADAM_BETA1": 0.5,
    "ADAM_BETA2": 0.9,
}


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """Critic (no sigmoid) using InstanceNorm for WGAN compatibility."""

    def __init__(self, num_classes: int, img_size: int, cfg: dict = DEFAULT_CONFIG):
        super().__init__()
        self.img_size = img_size
        self.nc = cfg["IMG_CHANNELS"]
        self.nf = cfg["NDF"]

        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        self.network = nn.Sequential(
            nn.Conv2d(self.nc + 1, self.nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.nf * 2, affine=True),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.nf * 4, affine=True),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.nf * 8, affine=True),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Conv2d(self.nf * 8, self.nf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.nf * 16, affine=True),
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
        self.nf = cfg["NDF"]

        self.label_embedding = nn.Embedding(num_classes, self.label_emb_dim)

        init_size = cfg["IMG_SIZE"] // 16
        self._init_size = init_size
        self.l1 = nn.Sequential(
            nn.Linear(self.noise_dim + self.label_emb_dim, self.nf * 16 * init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.nf * 16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 16, self.nf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.nf * 8, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 8, self.nf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.nf * 4, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.nf * 2, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(cfg["LEAKY_RELU_SLOPE"], inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.nf * 2, cfg["IMG_CHANNELS"], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.nf * 16, self._init_size, self._init_size)
        return self.conv_blocks(out)


# ---------------------------------------------------------------------------
# Gradient penalty helper
# ---------------------------------------------------------------------------


def _compute_gradient_penalty(
    discriminator: Discriminator,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    d_interpolates = discriminator(interpolates, labels)
    fake_out = torch.ones(real_samples.size(0), requires_grad=False, device=device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ---------------------------------------------------------------------------
# WCGAN class
# ---------------------------------------------------------------------------


class WCGANModel(GAN):
    """Wasserstein Conditional GAN with Gradient Penalty."""

    DEFAULT_CONFIG = DEFAULT_CONFIG

    def build_models(self) -> Tuple[nn.Module, nn.Module]:
        cfg = self.training_config
        generator = Generator(self.num_classes, cfg).to(self.device)
        discriminator = Discriminator(self.num_classes, cfg["IMG_SIZE"], cfg).to(
            self.device
        )
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
        noise_dim: int = cfg["NOISE_DIM"]
        n_critic: int = cfg["N_CRITIC"]
        lambda_gp: float = cfg["LAMBDA_GP"]

        data_iter = iter(self.data_loader)
        batch_idx = 0
        d_loss_val = g_loss_val = 0.0
        gp_val = 0.0
        gen_steps = 0

        def _next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                return next(data_iter)

        while batch_idx < len(self.data_loader):
            # Train critic N_CRITIC times
            for _ in range(n_critic):
                real_images, labels = _next_batch()
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                bs = real_images.size(0)

                self.discriminator.train()
                d_optimizer.zero_grad()

                real_validity = self.discriminator(real_images, labels)
                z = torch.randn(bs, noise_dim, device=self.device)
                with torch.no_grad():
                    # Detach from generator graph — critic doesn't need it
                    fake_images = self.generator(z, labels).detach()

                fake_validity = self.discriminator(fake_images, labels)

                gp = _compute_gradient_penalty(
                    self.discriminator,
                    real_images,
                    fake_images,
                    labels,
                    self.device,
                )
                d_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + lambda_gp * gp
                )
                d_loss.backward()
                d_optimizer.step()

                d_loss_val = d_loss.item()
                gp_val = gp.item()
                last_bs = bs  # preserve for the generator step below

                # Free intermediates to reduce VRAM pressure between critic steps
                del (
                    real_images,
                    labels,
                    real_validity,
                    fake_images,
                    fake_validity,
                    gp,
                    d_loss,
                    z,
                )
                torch.cuda.empty_cache()

                batch_idx += 1
                if batch_idx >= len(self.data_loader):
                    break

            # Train generator once
            if batch_idx < len(self.data_loader):
                self.generator.train()
                g_optimizer.zero_grad()
                z = torch.randn(last_bs, noise_dim, device=self.device)
                gen_labels = torch.randint(
                    0, self.num_classes, (last_bs,), device=self.device
                )
                fake_images = self.generator(z, gen_labels)
                fake_validity = self.discriminator(fake_images, gen_labels)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                g_optimizer.step()
                g_loss_val = g_loss.item()
                gen_steps += 1

                del z, gen_labels, fake_images, fake_validity, g_loss
                torch.cuda.empty_cache()

        return {"d_loss": d_loss_val, "g_loss": g_loss_val, "gradient_penalty": gp_val}

    def evaluate(self) -> Dict[str, float]:
        """
        WCGAN evaluate: we compute Wasserstein distance approximation as proxy metric.
        FID is reported too (requires validate_obj to provide images).

        Note: if calculate_fid / calculate_psnr_ssim_lpips are available we use them;
        otherwise we fall back to the W-distance only.
        """
        cfg = self.training_config
        noise_dim: int = cfg["NOISE_DIM"]

        self.generator.eval()
        self.discriminator.eval()

        real_scores, fake_scores = [], []
        real_images_list, fake_images_list = [], []

        with torch.no_grad():
            for real_imgs, labels in self.validate_obj.data_loader:
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)
                bs = real_imgs.size(0)

                rv = self.discriminator(real_imgs, labels)
                real_scores.append(rv.cpu())
                real_images_list.append(real_imgs.cpu())

                z = torch.randn(bs, noise_dim, device=self.device)
                gen_labels = torch.randint(
                    0, self.num_classes, (bs,), device=self.device
                )
                fake_imgs = self.generator(z, gen_labels)
                fv = self.discriminator(fake_imgs, gen_labels)
                fake_scores.append(fv.cpu())
                fake_images_list.append(fake_imgs.cpu())

        w_dist = (torch.cat(real_scores).mean() - torch.cat(fake_scores).mean()).item()

        metrics: Dict[str, float] = {"wasserstein_distance": w_dist}

        try:
            # this is needed because of tanh activation, which will return normalized data from range(-1, 1)
            # [-1,1] -> [0, 255] uint8
            real_cat = ((torch.cat(real_images_list) + 1) / 2 * 255).to(torch.uint8)
            fake_cat = ((torch.cat(fake_images_list) + 1) / 2 * 255).to(torch.uint8)

            if AvailableMetrics.FID in self.metrics_to_calculate:
                fid = calculate_fid(real_cat, fake_cat)
            else:
                fid = -1.0

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
                        m.value.lower() for m in self.metrics_to_calculate
                    ],
                )
            else:
                psnr = ssim = lpips = -1.0

            metrics.update({"fid": fid, "psnr": psnr, "ssim": ssim, "lpips": lpips})
        except Exception:
            # If evaluation metrics are unavailable, use negative W-distance as FID proxy
            # (higher real score = better generator, lower value wins in our comparator)
            metrics["fid"] = -w_dist

        return metrics

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
                current_bs = min(
                    batch_size,
                    num_augmented_images - sum(t.shape[0] for t in augmented_images),
                )
                if current_bs <= 0:
                    break
                z = torch.randn(current_bs, noise_dim, device=self.device)
                labels = torch.randint(
                    0, self.num_classes, (current_bs,), device=self.device
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
    output_path: Path,
    validate_ds: Optional[ImageDataset] = None,
    resume: bool = False,
    hyperparams: Optional[Hyperparams] = None,
    metrics: Optional[Metrics] = None,
) -> WCGANModel:
    model = WCGANModel(
        dataset_obj=dataset_obj,
        validate_obj=validate_ds or dataset_obj,
        output_path=output_path,
        hyperparams=hyperparams,
        resume=resume,
        metrics_to_calculate=metrics
        if metrics
        else [AvailableMetrics.WASSERSTEIN_DISTANCE, AvailableMetrics.FID],
    )
    model.run()
    return model


def augment(
    dataset_obj: ImageDataset,
    output_path: Path,
    gan_training_path: Path,
    num_augmented_images: int = 1000,
    config: dict = None,
) -> None:
    model = WCGANModel(
        dataset_obj=dataset_obj,
        validate_obj=dataset_obj,
        output_path=output_path,
        hyperparams=config,
    )
    model.augment(output_path, gan_training_path, num_augmented_images)
