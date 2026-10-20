"""Command line entry point: python -m ghtriage.cli <command>."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from ghtriage.collect import run_collector
from ghtriage.config import load_config
from ghtriage.demo_data import make_demo_data
from ghtriage.drift import drift_report, live_auc
from ghtriage.lake import build_table
from ghtriage.registry import load_current, load_reference
from ghtriage.storage import connect, read_table, write_frame

EXIT_RETRAIN_RECOMMENDED = 2


def cmd_init_demo(cfg: dict, args: argparse.Namespace) -> int:
    con = connect(cfg["data"]["db_path"])
    try:
        n = write_frame(con, cfg["data"]["table"], make_demo_data(n_rows=args.rows, shift=args.shift))
    finally:
        con.close()
    print(f"Wrote {n} synthetic rows to table '{cfg['data']['table']}'.")
    return 0


def cmd_collect(cfg: dict, args: argparse.Namespace) -> int:
    inserted, updated = run_collector(args.name, cfg)
    print(f"Collector '{args.name}': {inserted} inserted, {updated} updated.")
    return 0


def cmd_build(cfg: dict, args: argparse.Namespace) -> int:
    """Load the lake into the DuckDB table that training reads.

    Projects with real feature engineering set `data.builder: "package.module:function"`; the function
    receives the config, writes its own tables, and returns the number of training rows.
    """
    builder = cfg["data"].get("builder")
    if builder:
        module_name, _, func_name = builder.partition(":")
        rows = getattr(importlib.import_module(module_name), func_name)(cfg)
        print(f"Built '{cfg['data']['table']}' via {builder}: {rows} training rows.")
        return 0
    source = args.source or cfg["data"]["source"]
    con = connect(cfg["data"]["db_path"])
    try:
        n = build_table(con, cfg["lake"]["root"], source, cfg["data"]["table"])
    finally:
        con.close()
    print(f"Built table '{cfg['data']['table']}' from lake source '{source}': {n} rows.")
    return 0


def cmd_train(cfg: dict, args: argparse.Namespace) -> int:
    from ghtriage.train import train  # imported lazily: needs the optional "train" extra (MLflow)

    card = train(cfg)
    print(f"Promoted {card['version']} ({card['champion']}), test AUC {card['metrics']['champion_auc']:.3f}")
    print(card["selection_reason"])
    return 0


MIN_MONITOR_ROWS = 30


def _current_window(cfg: dict, window: int, after: str | None = None):
    """Most recent rows to monitor. With `after` (the model's training cutoff) only rows newer than that
    count, so live metrics are never computed on data the model was trained on."""
    con = connect(cfg["data"]["db_path"])
    try:
        df = read_table(con, cfg["data"]["table"], order_by=cfg["data"]["timestamp_col"])
    finally:
        con.close()
    if after is not None:
        newer = df[df[cfg["data"]["timestamp_col"]] > pd.Timestamp(after)]
        if len(newer) >= MIN_MONITOR_ROWS:
            return newer.tail(window)
        print(
            f"Warning: only {len(newer)} rows are newer than the training cutoff ({after}); "
            "falling back to the most recent rows, which overlap the training data. "
            "Live AUC below is optimistic until more data arrives.",
            file=sys.stderr,
        )
    return df.tail(window)


def cmd_drift(cfg: dict, args: argparse.Namespace) -> int:
    registry = cfg["model"]["registry_dir"]
    model, card, version = load_current(registry)
    reference = load_reference(registry)
    current = _current_window(cfg, args.window, card["data"]["train_end"])
    auc = live_auc(model, current, card["features"], card["target"])
    report = drift_report(
        reference,
        current,
        card["features"],
        cfg["drift"],
        live_auc=auc,
        baseline_auc=card["metrics"]["champion_auc"],
    )
    report["model_version"] = version
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Drift status: {report['overall_status']} | retrain recommended: {report['retrain_recommended']}")
    for reason in report["reasons"]:
        print(f" - {reason}")
    print(f"Report written to {out}")
    return EXIT_RETRAIN_RECOMMENDED if report["retrain_recommended"] else 0


def cmd_monitor_report(cfg: dict, args: argparse.Namespace) -> int:
    from ghtriage.monitor_report import evidently_html

    registry = cfg["model"]["registry_dir"]
    _, card, _ = load_current(registry)
    out = evidently_html(
        load_reference(registry),
        _current_window(cfg, args.window, card["data"]["train_end"]),
        card["features"],
        args.out,
    )
    print(f"Evidently report written to {out}")
    return 0


def cmd_explain(cfg: dict, args: argparse.Namespace) -> int:
    from ghtriage.explain import global_importance, make_explainer, save_importance_plot

    registry = cfg["model"]["registry_dir"]
    model, card, version = load_current(registry)
    reference = load_reference(registry)
    explainer = make_explainer(model, reference, card["features"])
    importance = global_importance(explainer, _current_window(cfg, args.window), card["features"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shap_importance.json").write_text(
        json.dumps({"model_version": version, "importance": importance}, indent=2)
    )
    plot = save_importance_plot(importance, out_dir / "shap_importance.png")
    print(f"Top features: {', '.join(list(importance)[:5])}")
    print(f"Wrote {out_dir / 'shap_importance.json'} and {plot}")
    return 0


def cmd_serve(cfg: dict, args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ghtriage.serve:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ghtriage")
    parser.add_argument("--config", default=None, help="Path to config YAML (default config/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-demo", help="Load synthetic placeholder data straight into the warehouse")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--shift", type=float, default=0.0, help="Shift feature means to simulate drift")
    p.set_defaults(func=cmd_init_demo)

    p = sub.add_parser("collect", help="Run a registered data collector (writes to the Parquet lake)")
    p.add_argument("name")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("build", help="Load the lake source into the DuckDB table used for training")
    p.add_argument("--source", default=None, help="Lake source name (default: data.source in config)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("train", help="Train, track in MLflow, and promote the champion")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("drift", help="Check data and performance drift (exit 2 if retraining advised)")
    p.add_argument("--window", type=int, default=1000, help="Most recent rows to compare")
    p.add_argument("--out", default="reports/drift_report.json")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("monitor-report", help="Evidently HTML drift report (needs the 'monitor' extra)")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--out", default="reports/evidently_drift.html")
    p.set_defaults(func=cmd_monitor_report)

    p = sub.add_parser("explain", help="Global SHAP importance (needs the 'explain' extra)")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--out-dir", default="reports")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("serve", help="Start the prediction API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
