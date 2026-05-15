"""Federated learning orchestration for HeartMRI-FL."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from .data import HeartMRIClientData, build_client_data
from .model import SimpleHeartMRIClassifier, SimpleHeartMRIAutoencoder


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for val in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.from_numpy(new_param).to(param.device, dtype=param.dtype)


def clip_and_add_noise(
    old_params: List[np.ndarray],
    new_params: List[np.ndarray],
    clipping_norm: float,
    noise_multiplier: float,
) -> List[np.ndarray]:
    deltas = [new_param - old_param for old_param, new_param in zip(old_params, new_params)]
    flat_delta = np.concatenate([delta.ravel() for delta in deltas]).astype(np.float64)
    norm = float(np.linalg.norm(flat_delta))
    if norm > clipping_norm and norm > 0.0:
        scale = clipping_norm / norm
        deltas = [delta * scale for delta in deltas]

    noise_scale = noise_multiplier * clipping_norm
    noisy_params: List[np.ndarray] = []
    for old_param, delta in zip(old_params, deltas):
        noise = np.random.normal(0.0, noise_scale, size=delta.shape)
        noisy_param = old_param + delta + noise
        noisy_params.append(noisy_param.astype(old_param.dtype))

    return noisy_params


def build_fit_config(
    batch_size: int,
    local_epochs: int,
    lr: float,
    clipping_norm: float,
    noise_multiplier: float,
    task: str,
) -> Dict[str, str]:
    return {
        "batch_size": str(batch_size),
        "local_epochs": str(local_epochs),
        "lr": str(lr),
        "clipping_norm": str(clipping_norm),
        "noise_multiplier": str(noise_multiplier),
        "task": task,
    }


class HeartMRIClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        client_data: HeartMRIClientData,
        task: str = "autoencoder",
    ) -> None:
        self.model = model
        self.client_data = client_data
        self.task = task
        self.loss_fn = nn.MSELoss() if task == "autoencoder" else nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        old_params = get_model_parameters(self.model)
        batch_size = int(config.get("batch_size", 16))
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-3))
        clipping_norm = float(config.get("clipping_norm", 1.0))
        noise_multiplier = float(config.get("noise_multiplier", 0.0))

        train_loader = self.client_data.train_loader(batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for _ in range(local_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(x)
                if self.task == "autoencoder":
                    target = x
                else:
                    target = y
                loss = self.loss_fn(outputs, target)
                loss.backward()
                optimizer.step()

        new_params = get_model_parameters(self.model)
        if noise_multiplier > 0.0:
            new_params = clip_and_add_noise(old_params, new_params, clipping_norm, noise_multiplier)
            set_model_parameters(self.model, new_params)

        return new_params, len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        batch_size = int(config.get("batch_size", 16))
        val_loader = self.client_data.val_loader(batch_size)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metrics = {}

        if self.task == "autoencoder":
            recon_errors: List[torch.Tensor] = []
            labels: List[torch.Tensor] = []

            with torch.no_grad():
                for x, y in val_loader:
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, x)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
                    recon_errors.append(((outputs - x) ** 2).mean(dim=[1, 2, 3]))
                    labels.append(y)

            if total_samples > 0:
                errors = torch.cat(recon_errors).cpu()
                label_tensor = torch.cat(labels).cpu()
                threshold = float(errors.mean() + errors.std())
                preds = errors > threshold
                true_anomaly = label_tensor == 1
                tp = int(((preds == 1) & (true_anomaly == 1)).sum())
                fp = int(((preds == 1) & (true_anomaly == 0)).sum())
                fn = int(((preds == 0) & (true_anomaly == 1)).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                metrics.update(
                    {
                        "reconstruction_threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    }
                )
        else:
            correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
                    correct += int((outputs.argmax(dim=1) == y).sum())
            if total_samples > 0:
                metrics["accuracy"] = correct / total_samples

        average_loss = total_loss / max(total_samples, 1)
        return float(average_loss), total_samples, metrics


def run_federated(
    data_root: Path = Path("./data"),
    num_clients: int = 3,
    subclients_per_hospital: int = 1,
    num_rounds: int = 3,
    image_size: Tuple[int, int] = (128, 128),
    num_classes: int = 2,
    batch_size: int = 16,
    local_epochs: int = 1,
    lr: float = 1e-3,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    task: str = "autoencoder",
    ray_init_args: dict[str, int] | None = None,
) -> None:
    client_data = build_client_data(
        root=data_root,
        image_size=image_size,
        num_hospitals=num_clients,
        subclients_per_hospital=subclients_per_hospital,
        num_classes=num_classes,
        samples_per_client=32,
    )

    client_ids = sorted(client_data.keys())
    print(f"Prepared {len(client_ids)} federated clients from {data_root}")
    print("Starting federated learning simulation...")

    def client_fn(context: object) -> fl.client.Client:
        if hasattr(context, "cid"):
            client_index = int(getattr(context, "cid"))
        elif hasattr(context, "node_config") and isinstance(context.node_config, dict):
            partition_id = context.node_config.get("partition-id")
            if partition_id is not None:
                client_index = int(partition_id)
            else:
                client_index = int(getattr(context, "node_id", 0))
        elif hasattr(context, "node_id"):
            client_index = int(getattr(context, "node_id"))
        else:
            raise AttributeError("Unsupported Flower Context: missing cid/node_id/partition-id")

        client_id = client_ids[client_index]
        model = (
            SimpleHeartMRIAutoencoder()
            if task == "autoencoder"
            else SimpleHeartMRIClassifier(num_classes=num_classes)
        )
        return HeartMRIClient(model, client_data[client_id], task=task).to_client()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(client_ids),
        min_evaluate_clients=len(client_ids),
        min_available_clients=len(client_ids),
        on_fit_config_fn=lambda rnd: build_fit_config(
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            task=task,
        ),
        on_evaluate_config_fn=lambda rnd: {"batch_size": str(batch_size), "task": task},
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args or {
            "num_cpus": 1,
            "object_store_memory": 100_000_000,
            "system_reserved_memory": 0,
        },
    )

    return history
