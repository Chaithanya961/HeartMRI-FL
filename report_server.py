from __future__ import annotations

import argparse
import base64
import io
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from heartmri_fl.federated import run_federated


def figure_to_data_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("utf-8")


def build_plot_histories(history) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    if history.losses_distributed:
        rounds, losses = zip(*history.losses_distributed)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds, losses, marker="o", color="#1f77b4")
        ax.set_title("Distributed Evaluation Loss")
        ax.set_xlabel("Server Round")
        ax.set_ylabel("Loss")
        ax.grid(True)
        plots["loss"] = figure_to_data_uri(fig)
        plt.close(fig)

    for metric_name, metric_values in history.metrics_distributed.items():
        rounds, values = zip(*metric_values)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds, values, marker="o", color="#ff7f0e")
        ax.set_title(f"Distributed Metric: {metric_name}")
        ax.set_xlabel("Server Round")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        plots[f"metric_{metric_name}"] = figure_to_data_uri(fig)
        plt.close(fig)

    return plots


def build_html(history, plots: Dict[str, str], generated_at: str) -> str:
    metric_rows = ""
    if history.metrics_distributed:
        for metric_name, metric_values in history.metrics_distributed.items():
            metric_rows += "<tr>"
            metric_rows += f"<td>{metric_name}</td>"
            metric_rows += f"<td>{', '.join(f'{val:.4f}' for _, val in metric_values)}</td>"
            metric_rows += f"<td>{', '.join(str(rnd) for rnd, _ in metric_values)}</td>"
            metric_rows += "</tr>"
    else:
        metric_rows = "<tr><td colspan=3>No distributed metrics recorded.</td></tr>"

    loss_rows = ""
    if history.losses_distributed:
        for rnd, loss in history.losses_distributed:
            loss_rows += f"<tr><td>{rnd}</td><td>{loss:.6f}</td></tr>"
    else:
        loss_rows = "<tr><td colspan=2>No distributed loss records available.</td></tr>"

    plot_images = ""
    if "loss" in plots:
        plot_images += f"<div class=plot><h2>Distributed Evaluation Loss</h2><img src='{plots['loss']}' alt='Loss plot'></div>"
    for key, uri in plots.items():
        if key != "loss":
            metric_name = key.replace("metric_", "")
            plot_images += f"<div class=plot><h2>{metric_name}</h2><img src='{uri}' alt='{metric_name} plot'></div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HeartMRI-FL Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f6fa; color: #222; }}
        .container {{ max-width: 1100px; margin: auto; }}
        h1 {{ color: #2c3e50; }}
        .meta {{ margin-bottom: 24px; padding: 18px; background: #ffffff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .plot {{ margin-bottom: 32px; padding: 18px; background: #ffffff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #e1e4e8; text-align: left; }}
        th {{ background: #ecf0f1; }}
        .small {{ font-size: 0.95rem; color: #555; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HeartMRI-FL Local Report</h1>
        <div class="meta">
            <p class="small">Generated at: {generated_at}</p>
            <p><strong>Federated clients:</strong> {len(history.losses_distributed) and len(history.losses_distributed) or 'Unknown'} server rounds recorded.</p>
        </div>
        {plot_images}
        <div class="plot">
            <h2>Distributed Evaluation Loss Table</h2>
            <table>
                <thead>
                    <tr><th>Server Round</th><th>Loss</th></tr>
                </thead>
                <tbody>
                    {loss_rows}
                </tbody>
            </table>
        </div>
        <div class="plot">
            <h2>Distributed Metrics Summary</h2>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Values</th><th>Rounds</th></tr>
                </thead>
                <tbody>
                    {metric_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""


def save_report(history, output_path: Path) -> None:
    plots = build_plot_histories(history)
    html = build_html(history, plots, generated_at=str(Path().resolve()))
    output_path.write_text(html, encoding="utf-8")


class LocalReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: Path | None = None, **kwargs):
        super().__init__(*args, directory=str(directory) if directory else None, **kwargs)


def start_server(port: int, directory: Path) -> None:
    handler = lambda *args, **kwargs: LocalReportHandler(*args, directory=directory, **kwargs)
    server = HTTPServer(("127.0.0.1", port), handler)
    print(f"Serving report at http://127.0.0.1:{port}/report.html")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and serve HeartMRI-FL reports locally.")
    parser.add_argument("--port", type=int, default=8000, help="Local port to serve the report on")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds to run")
    parser.add_argument("--clients", type=int, default=3, help="Number of federated clients to simulate")
    parser.add_argument("--batch-size", type=int, default=16, help="Local batch size")
    parser.add_argument("--output", type=Path, default=ROOT / "report.html", help="Output HTML report file")
    args = parser.parse_args()

    history = run_federated(
        data_root=ROOT / "data",
        num_clients=args.clients,
        num_rounds=args.rounds,
        image_size=(128, 128),
        num_classes=2,
        batch_size=args.batch_size,
    )

    save_report(history, args.output)
    print(f"Report generated at {args.output}")
    start_server(args.port, ROOT)


if __name__ == "__main__":
    main()
