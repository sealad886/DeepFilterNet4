#!/usr/bin/env python3
"""
plot_benchmark_results.py

Generate a self-contained HTML dashboard (Vega-Lite) from the benchmark CSV
produced by scripts/benchmark_backends.sh. No local plotting dependencies are
required; the HTML loads Vega/Vega-Lite from CDNs.

Outputs (to --out-dir, default logs/):
  - benchmark_dashboard.html  (interactive charts)
  - benchmark_summary.txt     (top-line winners)

Usage:
  ./scripts/plot_benchmark_results.py --csv logs/backend_benchmark_latest.csv --out-dir logs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_data(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric fields
            casted: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    if "." in v:
                        casted[k] = float(v)
                    else:
                        casted[k] = int(v)
                except ValueError:
                    casted[k] = v
            rows.append(casted)
    return rows


def top_lines(rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []

    def best(key: str, reverse: bool = True) -> Optional[Dict[str, Any]]:
        candidates = [r for r in rows if isinstance(r.get(key), (int, float))]
        if not candidates:
            return None
        return sorted(candidates, key=lambda r: r[key], reverse=reverse)[0]

    r = best("avg_throughput", True)
    if r:
        lines.append(
            f"Fastest: {r['backbone']} bs={r['batch_size']} "
            f"({r['avg_throughput']:.1f} samples/s, eval={r['eval_frequency']})"
        )

    r = best("accuracy", True)
    if r:
        metric = r.get("accuracy_metric", "accuracy")
        lines.append(f"Best quality ({metric}): {r['backbone']} bs={r['batch_size']} " f"({r['accuracy']:.2f})")

    r = best("avg_rss_mb", False)
    if r:
        lines.append(
            f"Lowest memory: {r['backbone']} bs={r['batch_size']} "
            f"(avg {r['avg_rss_mb']:.1f} MB, peak {r.get('max_rss_mb', 0):.1f} MB)"
        )

    return lines


def build_chart_spec(
    data: List[Dict[str, Any]],
    title: str,
    x: str,
    y: str,
    color: str = "backbone",
    tooltip_extra: List[str] | None = None,
    orientation: str = "vertical",
) -> Dict[str, Any]:
    tooltip = [x, y, color, "batch_size", "eval_frequency"]
    if tooltip_extra:
        tooltip.extend(tooltip_extra)
    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": title,
        "data": {"values": data},
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "x": {"field": x, "type": "ordinal", "title": x},
            "y": {"field": y, "type": "quantitative", "title": y},
            "color": {"field": color, "type": "nominal"},
            "tooltip": tooltip,
        },
        "title": title,
    }
    if orientation == "horizontal":
        spec["encoding"]["x"], spec["encoding"]["y"] = spec["encoding"]["y"], spec["encoding"]["x"]
    return spec


def build_html(data: List[Dict[str, Any]], summary_lines: List[str]) -> str:
    throughput_spec = build_chart_spec(
        data,
        "Avg Throughput (samples/s)",
        x="run",
        y="avg_throughput",
        tooltip_extra=["max_throughput"],
    )
    accuracy_spec = build_chart_spec(
        data,
        "Accuracy (prefers SI-SDR)",
        x="run",
        y="accuracy",
        tooltip_extra=["accuracy_metric"],
    )
    memory_spec = build_chart_spec(
        data,
        "Memory (avg MB, peak as tooltip)",
        x="run",
        y="avg_rss_mb",
        tooltip_extra=["max_rss_mb"],
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Benchmark Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; margin: 24px; background: #0d1117; color: #e6edf3; }}
    h1 {{ margin-bottom: 4px; }}
    .card {{ background: #161b22; padding: 16px 18px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .summary li {{ margin: 4px 0; }}
    a {{ color: #58a6ff; }}
  </style>
</head>
<body>
  <h1>Backend Benchmark Dashboard</h1>
  <p style="opacity:0.8">Interactive charts built from benchmark_backends.sh output.</p>

  <div class="card">
    <h3>Highlights</h3>
    <ul class="summary">
      {''.join(f'<li>{line}</li>' for line in summary_lines) if summary_lines else '<li>No runs found.</li>'}
    </ul>
  </div>

  <div class="grid">
    <div class="card" id="chart-throughput"></div>
    <div class="card" id="chart-accuracy"></div>
    <div class="card" id="chart-memory"></div>
  </div>

  <script type="text/javascript">
    const throughputSpec = {json.dumps(throughput_spec)};
    const accuracySpec = {json.dumps(accuracy_spec)};
    const memorySpec = {json.dumps(memory_spec)};
    vegaEmbed('#chart-throughput', throughputSpec, {{actions:false, theme:'dark'}}).catch(console.error);
    vegaEmbed('#chart-accuracy', accuracySpec, {{actions:false, theme:'dark'}}).catch(console.error);
    vegaEmbed('#chart-memory', memorySpec, {{actions:false, theme:'dark'}}).catch(console.error);
  </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate Vega-Lite dashboard from benchmark CSV.")
    parser.add_argument("--csv", type=Path, default=Path("logs/backend_benchmark_latest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_data(args.csv)
    if not rows:
        print(f"No data found in {args.csv}")
        return

    for r in rows:
        r.setdefault(
            "run",
            f"{r.get('backbone', '?')}-bs{r.get('batch_size', '?')}-e{r.get('eval_frequency', '?')}",
        )

    data = rows
    summary_lines = top_lines(rows)

    html = build_html(data, summary_lines)
    html_path = args.out_dir / "benchmark_dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    txt_path = args.out_dir / "benchmark_summary.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Dashboard written to:", html_path)
    print("Summary written to:  ", txt_path)


if __name__ == "__main__":
    main()
