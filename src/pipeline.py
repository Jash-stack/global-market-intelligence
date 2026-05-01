"""
End-to-end pipeline orchestrator.
Run:  python src/pipeline.py [--steps all|ingest|clean|outlier|features|forecast|anomaly|eval]
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline")

STEPS = ["ingest", "clean", "outlier", "features", "forecast", "anomaly", "eval"]


def run_step(name: str) -> float:
    t0 = time.perf_counter()
    if name == "ingest":
        from src.ingestion         import run_ingestion;         run_ingestion()
    elif name == "clean":
        from src.cleaning          import run_cleaning;          run_cleaning()
    elif name == "outlier":
        from src.outlier_detection import run_outlier_detection; run_outlier_detection()
    elif name == "features":
        from src.feature_engineering import run_feature_engineering; run_feature_engineering()
    elif name == "forecast":
        from src.forecasting       import run_forecasting;       run_forecasting()
    elif name == "anomaly":
        from src.anomaly_detection import run_anomaly_detection; run_anomaly_detection()
    elif name == "eval":
        from src.evaluation        import run_evaluation;        run_evaluation()
    else:
        raise ValueError(f"Unknown step: {name}")
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Global Market Intelligence Pipeline")
    parser.add_argument(
        "--steps", default="all",
        help="Comma-separated steps or 'all'.  Options: " + ", ".join(STEPS),
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated steps to skip.",
    )
    args = parser.parse_args()

    chosen = STEPS if args.steps.strip() == "all" else [s.strip() for s in args.steps.split(",")]
    skip   = {s.strip() for s in args.skip.split(",") if s.strip()}
    to_run = [s for s in chosen if s not in skip]

    log.info("Pipeline will run: %s", " → ".join(to_run))
    timings = {}
    errors  = {}

    for step in to_run:
        log.info("━━━ Starting step: %s ━━━", step.upper())
        try:
            elapsed = run_step(step)
            timings[step] = round(elapsed, 1)
            log.info("✓ %s completed in %.1fs", step, elapsed)
        except Exception as exc:
            log.exception("✗ Step '%s' FAILED: %s", step, exc)
            errors[step] = str(exc)

    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    for step, t in timings.items():
        status = "✓" if step not in errors else "✗"
        log.info("  %s %-12s  %.1fs", status, step, t)
    if errors:
        log.error("Failed steps: %s", list(errors.keys()))
    else:
        log.info("All steps completed successfully.")
    log.info("Total: %.1fs", sum(timings.values()))


if __name__ == "__main__":
    main()
