"""Run one experiment with two CPU threads, a GPU cap, and a memory watchdog."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-log", type=Path, required=True)
    parser.add_argument("--min-available-gib", type=float, default=24)
    parser.add_argument("--max-rss-gib", type=float, default=80)
    parser.add_argument("--gpu-fraction", type=float, default=0.32)
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1")
    os.nice(10)
    import psutil
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False
    args.resource_log.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    process = psutil.Process()

    def monitor():
        with args.resource_log.open("x", buffering=1) as handle:
            while not stop.is_set():
                available = psutil.virtual_memory().available / 2**30
                rss = process.memory_info().rss / 2**30
                record = dict(time=time.time(), available_gib=available, rss_gib=rss)
                if torch.cuda.is_available():
                    record.update(gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                                  gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30)
                handle.write(json.dumps(record) + "\n")
                if available < args.min_available_gib or rss > args.max_rss_gib:
                    print(f"Memory watchdog stopped this experiment: {record}", file=sys.stderr, flush=True)
                    os._exit(75)
                stop.wait(2)

    if psutil.virtual_memory().available / 2**30 < args.min_available_gib:
        raise RuntimeError("Insufficient available memory to start safely")
    if args.resource_log.exists():
        raise FileExistsError(args.resource_log)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    sys.argv = [args.script, *args.args]
    sys.path.insert(0, str(Path(args.script).resolve().parent))
    try:
        runpy.run_path(args.script, run_name="__main__")
    finally:
        stop.set()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
