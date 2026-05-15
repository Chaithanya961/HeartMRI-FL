"""Command-line entrypoint for HeartMRI-FL."""

import argparse
from pathlib import Path

from .federated import run_federated


def main() -> None:
    parser = argparse.ArgumentParser(description="HeartMRI-FL federated learning runner")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root folder for client MRI data")
    parser.add_argument("--clients", type=int, default=3, help="Number of hospitals participating in federated training")
    parser.add_argument(
        "--subclients-per-hospital",
        type=int,
        default=1,
        help="Number of adaptive intermediary sub-clients per hospital",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per round")
    parser.add_argument("--batch-size", type=int, default=16, help="Local batch size")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], help="Image size for 2D slices")
    parser.add_argument("--classes", type=int, default=2, help="Number of output classes")
    parser.add_argument(
        "--task",
        choices=["autoencoder", "classifier"],
        default="autoencoder",
        help="Training task: autoencoder anomaly detection or classifier",
    )
    parser.add_argument(
        "--clipping-norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for DP clipping of client updates",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Gaussian noise multiplier for client-level differential privacy",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Local optimizer learning rate")
    args = parser.parse_args()

    print("HeartMRI-FL starting...")
    run_federated(
        data_root=Path(args.data_dir),
        num_clients=args.clients,
        subclients_per_hospital=args.subclients_per_hospital,
        num_rounds=args.rounds,
        image_size=(args.image_size[0], args.image_size[1]),
        num_classes=args.classes,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.learning_rate,
        clipping_norm=args.clipping_norm,
        noise_multiplier=args.noise_multiplier,
        task=args.task,
    )


if __name__ == "__main__":
    main()
