"""Command-line entry point: python -m poreconn.run --stages all

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings

from .config import Config

STAGES = ["extract", "air", "segment", "qa", "topology", "coarsen", "rev", "predict", "transport", "lbm", "estimates",
          "wall", "grains", "imaging", "budget", "validation", "report", "figures"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="all",
                    help="comma-separated list from: " + ", ".join(STAGES) + " (or 'all', or 'analysis' = all but extract)")
    ap.add_argument("--samples", default="", help="comma-separated sample ids (default: all)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test on small VOIs (outputs_quick/)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(line_buffering=True)
    import logging
    logging.disable(logging.WARNING)
    cfg = Config(a.config, quick=a.quick)
    if a.stages == "all":
        stages = STAGES
    elif a.stages == "analysis":
        stages = STAGES[1:]
    else:
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    only = [s.strip() for s in a.samples.split(",") if s.strip()] or None
    for st in stages:
        if st not in STAGES:
            sys.exit(f"unknown stage {st}")
        t0 = time.time()
        print(f"== stage {st}", flush=True)
        mod = importlib.import_module(f".{st}", __package__)
        if st in ("report", "figures"):
            mod.run(cfg)
        else:
            mod.run(cfg, only=only)
        print(f"== stage {st} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
