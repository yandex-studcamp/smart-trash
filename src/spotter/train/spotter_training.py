from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config.spotter_config import SpotterConfig
from ..data.spotter_dataset import CorruptedNormalSpotterDataset, build_train_val_image_lists
from ..inference.spotter_inference import run_spotter_calibration, tensor_to_rgb_image
from ..models.spotter_model import SpotterDAAE, count_model_parameters
from ..utils.spotter_utils import ExperimentPaths, save_json, select_device, set_seed
from .spotter_losses import SpotterReconstructionLoss


def train_spotter_model(
    config: SpotterConfig,
    experiment_paths: ExperimentPaths,
) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)

    train_paths, val_paths = build_train_val_image_lists(
        train_normal_dir=config.data.train_normal_dir,
        val_normal_dir=config.data.val_normal_dir,
        image_extensions=config.data.image_extensions,
        val_split=config.data.val_split,
        seed=config.seed,
        max_train_images=config.data.max_train_images,
    )

    train_dataset = CorruptedNormalSpotterDataset(
        image_paths=train_paths,
        image_size=config.data.image_size,
        augmentation=config.augmentation,
        enable_corruption=True,
    )
    val_dataset = CorruptedNormalSpotterDataset(
        image_paths=val_paths,
        image_size=config.data.image_size,
        augmentation=config.augmentation,
        enable_corruption=False,
    )

    pin_memory = config.data.pin_memory and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    model = SpotterDAAE(config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = SpotterReconstructionLoss(
        mse_weight=config.training.mse_weight,
        ssim_weight=config.training.ssim_weight,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    print(
        f"[spotter] start training exp='{config.exp_name}' "
        f"device={device} train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
        f"epochs={config.training.epochs}"
    )

    for epoch in range(1, config.training.epochs + 1):
        print(f"\n[spotter] epoch {epoch}/{config.training.epochs}")
        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=config.training.grad_clip_norm,
            stage_name="train",
            epoch=epoch,
            total_epochs=config.training.epochs,
            leave_progress=True,
        )
        val_metrics = _run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip_norm=None,
            stage_name="val",
            epoch=epoch,
            total_epochs=config.training.epochs,
            leave_progress=False,
        )

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_ssim_loss": train_metrics["ssim_loss"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_ssim_loss": val_metrics["ssim_loss"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)

        if val_metrics["loss"] < best_val_loss:
            previous_best = best_val_loss
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            _save_checkpoint(
                output_path=experiment_paths.weights_dir / "best_spotter_daae.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )
            if previous_best == float("inf"):
                print(
                    f"[spotter] saved best model: epoch={epoch} val_loss={best_val_loss:.6f} "
                    f"path={config.exp_name + '/best_spotter_daae.pt'}"
                )
            else:
                print(
                    f"[spotter] saved best model: epoch={epoch} val_loss={best_val_loss:.6f} "
                    f"(prev_best={previous_best:.6f}) "
                    f"path={experiment_paths.weights_dir / 'best_spotter_daae.pt'}"
                )

        if epoch % config.training.save_every_n_epochs == 0 or epoch == config.training.epochs:
            _save_checkpoint(
                output_path=experiment_paths.weights_dir / "last_spotter_daae.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )
            _save_reconstruction_preview(
                model=model,
                dataloader=train_loader,
                device=device,
                output_path=experiment_paths.artifacts_dir / f"reconstruction_epoch_{epoch:03d}.png",
                max_images=config.training.sample_visualizations,
            )

        print(
            f"[spotter] epoch {epoch}/{config.training.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_mse={train_metrics['mse']:.6f} "
            f"train_ssim={train_metrics['ssim_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_ssim={val_metrics['ssim_loss']:.6f}"
        )

    _save_history(history, experiment_paths.artifacts_dir / "train_history.csv")
    _save_history_plot(history, experiment_paths.artifacts_dir / "train_history.png")

    calibration_summary: dict[str, Any] | None = None
    if config.data.val_normal_dir and config.data.val_anomaly_dir:
        calibration_summary = run_spotter_calibration(
            config=config,
            experiment_paths=experiment_paths,
            weights_path=experiment_paths.weights_dir / "best_spotter_daae.pt",
        )
        print(
            f"[spotter] validation calibration complete "
            f"threshold={calibration_summary['image_threshold']:.6f} "
            f"f1={calibration_summary['f1']:.4f}"
        )
    else:
        print("[spotter] validation calibration skipped: val_anomaly_dir is not configured.")

    summary = {
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": config.training.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "parameter_count": count_model_parameters(model),
        "best_weights_path": str(experiment_paths.weights_dir / "best_spotter_daae.pt"),
        "last_weights_path": str(experiment_paths.weights_dir / "last_spotter_daae.pt"),
        "calibration_threshold": calibration_summary["image_threshold"] if calibration_summary else None,
    }
    save_json(summary, experiment_paths.artifacts_dir / "train_summary.json")
    return summary


def _run_epoch(
    model: SpotterDAAE,
    dataloader: DataLoader,
    criterion: SpotterReconstructionLoss,
    device: torch.device,
    optimizer: AdamW | None,
    grad_clip_norm: float | None,
    stage_name: str,
    epoch: int,
    total_epochs: int,
    leave_progress: bool,
) -> dict[str, float]:
    metrics_sum: dict[str, float] = defaultdict(float)
    total_samples = 0

    if optimizer is None:
        model.eval()
    else:
        model.train()

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"{stage_name} {epoch}/{total_epochs}",
        leave=leave_progress,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    with torch.set_grad_enabled(optimizer is not None):
        for batch in progress_bar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            batch_size = inputs.size(0)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(inputs)
            total_loss, loss_components = criterion(predictions, targets)

            if optimizer is not None:
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            metrics_sum["loss"] += float(total_loss.detach().item()) * batch_size
            metrics_sum["mse"] += float(loss_components["mse"].item()) * batch_size
            metrics_sum["ssim_loss"] += float(loss_components["ssim_loss"].item()) * batch_size
            total_samples += batch_size

            running_metrics = {
                key: value / total_samples
                for key, value in metrics_sum.items()
            }
            progress_bar.set_postfix(
                loss=f"{running_metrics['loss']:.4f}",
                mse=f"{running_metrics['mse']:.4f}",
                ssim=f"{running_metrics['ssim_loss']:.4f}",
            )

    progress_bar.close()

    if total_samples == 0:
        raise ValueError("Empty dataloader detected during training.")

    return {
        key: value / total_samples
        for key, value in metrics_sum.items()
    }


def _save_checkpoint(
    output_path: Path,
    model: SpotterDAAE,
    optimizer: AdamW,
    epoch: int,
    best_val_loss: float,
    config: SpotterConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
        },
        output_path,
    )


def _save_reconstruction_preview(
    model: SpotterDAAE,
    dataloader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_images: int,
) -> None:
    model.eval()
    batch = next(iter(dataloader))
    inputs = batch["input"][:max_images].to(device)
    targets = batch["target"][:max_images]
    with torch.no_grad():
        reconstructions = model(inputs).cpu()

    rows = inputs.shape[0]
    if rows == 0:
        return

    plt.figure(figsize=(9, 3 * rows))
    for row_index in range(rows):
        input_image = tensor_to_rgb_image(inputs[row_index].cpu())
        target_image = tensor_to_rgb_image(targets[row_index])
        reconstruction_image = tensor_to_rgb_image(reconstructions[row_index])

        plt.subplot(rows, 3, row_index * 3 + 1)
        plt.imshow(input_image)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(rows, 3, row_index * 3 + 2)
        plt.imshow(target_image)
        plt.title("Target")
        plt.axis("off")

        plt.subplot(rows, 3, row_index * 3 + 3)
        plt.imshow(reconstruction_image)
        plt.title("Reconstruction")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_history(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _save_history_plot(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [int(entry["epoch"]) for entry in history]
    train_loss = [float(entry["train_loss"]) for entry in history]
    val_loss = [float(entry["val_loss"]) for entry in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="train_loss", color="#2563eb", linewidth=2)
    plt.plot(epochs, val_loss, label="val_loss", color="#dc2626", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Spotter DAAE Training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
