from __future__ import annotations

import json
import mimetypes
import threading
import sys
import time
import urllib.parse
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from heartmri_fl.federated import run_federated
except ImportError as exc:
    run_federated = None
    TRAINING_IMPORT_ERROR = str(exc)

TRAINING_STATE = {
    "running": False,
    "thread": None,
    "started_at": None,
    "completed_at": None,
    "message": "idle",
    "progress": 0,
    "history": None,
    "command": None,
    "results": None,
}


class DatasetApiHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/images":
            self.handle_image_list()
            return

        if parsed.path.startswith("/api/images/"):
            self.handle_image_file(parsed.path[len("/api/images/"):])
            return

        if parsed.path == "/api/metadata":
            self.handle_metadata()
            return

        if parsed.path == "/api/stats":
            self.handle_stats()
            return

        if parsed.path == "/api/train":
            self.handle_train()
            return

        if parsed.path == "/api/train-status":
            self.handle_train_status()
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/train":
            self.handle_train()
            return

        super().do_POST()

    def handle_image_list(self) -> None:
        image_files = sorted(DATA_ROOT.rglob("*.jpg"))
        sample_files = image_files[:16]
        payload = [
            {
                "id": image.relative_to(DATA_ROOT).as_posix(),
                "hospital": image.relative_to(DATA_ROOT).parts[0] if len(image.relative_to(DATA_ROOT).parts) > 1 else "unknown",
                "name": image.name,
                "description": f"Sample MRI slice from {image.relative_to(DATA_ROOT).parts[0]}",
                "url": f"/api/images/{image.relative_to(DATA_ROOT).as_posix()}"
            }
            for image in sample_files
        ]
        self.send_json(payload)

    def handle_image_file(self, raw_path: str) -> None:
        safe_path = Path(urllib.parse.unquote(raw_path.lstrip("/")))
        full_path = (DATA_ROOT / safe_path).resolve()

        try:
            full_path.relative_to(DATA_ROOT)
        except Exception:
            self.send_error(403, "Forbidden")
            return

        if not full_path.exists() or not full_path.is_file():
            self.send_error(404, "Not Found")
            return

        mime_type, _ = mimetypes.guess_type(str(full_path))
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()

        with open(full_path, "rb") as f:
            self.wfile.write(f.read())

    def handle_metadata(self) -> None:
        image_files = sorted(DATA_ROOT.rglob("*.jpg"))
        hospitals = sorted({path.relative_to(DATA_ROOT).parts[0] for path in image_files if len(path.relative_to(DATA_ROOT).parts) > 0})
        hospital_counts = {
            hospital: sum(1 for path in image_files if path.relative_to(DATA_ROOT).parts[0] == hospital)
            for hospital in hospitals
        }
        payload = {
            "total_images": len(image_files),
            "hospitals": hospitals,
            "hospital_counts": hospital_counts,
            "sample_count": min(len(image_files), 16),
        }
        self.send_json(payload)

    def handle_stats(self) -> None:
        image_files = sorted(DATA_ROOT.rglob("*.jpg"))
        hospitals = sorted({path.relative_to(DATA_ROOT).parts[0] for path in image_files if len(path.relative_to(DATA_ROOT).parts) > 0})
        hospital_counts = {
            hospital: sum(1 for path in image_files if path.relative_to(DATA_ROOT).parts[0] == hospital)
            for hospital in hospitals
        }
        augmented_patterns = [
            "flipped",
            "rotated",
            "sheared",
            "augmented",
        ]
        augmentation_breakdown = {
            "flipped": sum(1 for path in image_files if "flipped" in path.name.lower()),
            "rotated": sum(1 for path in image_files if "rotated" in path.name.lower()),
            "sheared": sum(1 for path in image_files if "sheared" in path.name.lower()),
            "augmented": sum(1 for path in image_files if "augmented" in [part.lower() for part in path.parts] or "augmented" in path.name.lower()),
        }
        augmented_image_paths = {
            path
            for path in image_files
            if any(pattern in path.name.lower() or pattern in [part.lower() for part in path.parts] for pattern in augmented_patterns)
        }
        augmented_images = len(augmented_image_paths)

        training_count = sum(
            1
            for path in image_files
            if any(part.lower() == "training" or part.lower().startswith("train") for part in path.parts)
        )
        testing_count = sum(
            1
            for path in image_files
            if any(part.lower() == "testing" or part.lower().startswith("test") for part in path.parts)
        )

        total_size_bytes = sum(path.stat().st_size for path in image_files)
        payload = {
            "total_images": len(image_files),
            "hospitals": hospitals,
            "hospital_counts": hospital_counts,
            "sample_count": min(len(image_files), 16),
            "augmented_images": augmented_images,
            "augmentation_breakdown": augmentation_breakdown,
            "training_count": training_count,
            "testing_count": testing_count,
            "dataset_size_bytes": total_size_bytes,
            "dataset_size_mb": round(total_size_bytes / 1024 / 1024, 2),
        }
        self.send_json(payload)

    def handle_train(self) -> None:
        if run_federated is None:
            # For demonstration, provide mock training when federated learning is not available
            print("Federated learning not available, using mock training for demonstration")
            if TRAINING_STATE["running"]:
                self.send_json({
                    "status": "running",
                    "message": "Training already in progress",
                    "started_at": TRAINING_STATE["started_at"],
                })
                return

            TRAINING_STATE["running"] = True
            TRAINING_STATE["message"] = "Training started"
            TRAINING_STATE["started_at"] = datetime.utcnow().isoformat() + "Z"
            TRAINING_STATE["completed_at"] = None
            TRAINING_STATE["command"] = "python -m heartmri_fl.federated --data_root ./data --num_clients 3 --subclients_per_hospital 1 --num_rounds 3 --image_size 128,128 --num_classes 2 --batch_size 16 --local_epochs 1 --lr 0.001 --clipping_norm 1.0 --noise_multiplier 1.0 --task autoencoder"

            def mock_training_worker() -> None:
                print("Mock training worker started!")
                try:
                    TRAINING_STATE["message"] = "Training running"
                    TRAINING_STATE["progress"] = 15

                    # Simulate progress updates during training
                    def update_progress() -> None:
                        steps = [25, 40, 60, 75, 90]
                        for step in steps:
                            if TRAINING_STATE["running"]:
                                TRAINING_STATE["progress"] = step
                                time.sleep(3)  # Simulate training time between steps

                    progress_thread = threading.Thread(target=update_progress, daemon=True)
                    progress_thread.start()

                    # Simulate training time
                    time.sleep(10)

                    # Create mock history object for demonstration
                    class MockHistory:
                        def __init__(self):
                            self.losses_distributed = [(i+1, 0.8 - i*0.1) for i in range(3)]
                            self.metrics_distributed = {
                                "accuracy": [(i+1, 0.75 + i*0.05) for i in range(3)],
                                "precision": [(i+1, 0.78 + i*0.04) for i in range(3)],
                                "recall": [(i+1, 0.76 + i*0.03) for i in range(3)],
                                "f1_score": [(i+1, 0.77 + i*0.035) for i in range(3)]
                            }
                    history = MockHistory()
                    TRAINING_STATE["history"] = history

                    # Extract mock training results
                    results = {
                        "rounds_completed": len(history.losses_distributed),
                        "final_loss": history.losses_distributed[-1][1],
                        "metrics": {}
                    }

                    if history.metrics_distributed:
                        for metric_name, metric_values in history.metrics_distributed.items():
                            if metric_values:
                                results["metrics"][metric_name] = {
                                    "final_value": metric_values[-1][1],
                                    "values": [val for _, val in metric_values],
                                    "rounds": [rnd for rnd, _ in metric_values]
                                }

                    TRAINING_STATE["results"] = results
                    TRAINING_STATE["message"] = "Training completed (with mock results for demonstration)"
                    TRAINING_STATE["progress"] = 100
                    print("Mock training completed successfully!")
                except Exception as exc:
                    print(f"Mock training failed: {exc}")
                    TRAINING_STATE["message"] = f"Training failed: {exc}"
                    TRAINING_STATE["progress"] = 100
                finally:
                    TRAINING_STATE["completed_at"] = datetime.utcnow().isoformat() + "Z"
                    TRAINING_STATE["running"] = False

            training_thread = threading.Thread(target=mock_training_worker, daemon=True)
            TRAINING_STATE["thread"] = training_thread
            training_thread.start()

            self.send_json({
                "status": "started",
                "message": "Training has started in background",
                "running": True,
                "started_at": TRAINING_STATE["started_at"],
                "progress": TRAINING_STATE["progress"],
            })

    def handle_train_status(self) -> None:
        self.send_json({
            "status": TRAINING_STATE["message"].lower().replace(' ', '_'),
            "message": TRAINING_STATE["message"],
            "running": TRAINING_STATE["running"],
            "started_at": TRAINING_STATE["started_at"],
            "completed_at": TRAINING_STATE["completed_at"],
            "progress": TRAINING_STATE["progress"],
            "command": TRAINING_STATE["command"],
            "results": TRAINING_STATE["results"],
        })

    def send_json(self, payload: object) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(port: int = 8000) -> None:
    handler = lambda *args, **kwargs: DatasetApiHandler(*args, directory=str(ROOT), **kwargs)
    server = HTTPServer(("127.0.0.1", port), handler)
    print(f"Serving HeartMRI-FL API and dataset files at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
