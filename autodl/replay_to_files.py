"""Atomically replay NDJSON telemetry into Spark's file-stream inbox."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--records-per-file", type=int, default=10000)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    args.inbox.mkdir(parents=True, exist_ok=True)
    written = files = 0
    with args.source.open(encoding="utf-8") as source:
        while True:
            chunk = [source.readline() for _ in range(args.records_per_file)]
            chunk = [line for line in chunk if line]
            if not chunk:
                break
            temporary = args.inbox / f".batch-{files:06d}.tmp"
            final = args.inbox / f"batch-{files:06d}.json"
            temporary.write_text("".join(chunk), encoding="utf-8")
            shutil.move(temporary, final)  # Spark only discovers the completed file.
            files += 1; written += len(chunk)
            if args.delay_seconds:
                time.sleep(args.delay_seconds)
    print({"files": files, "records": written, "inbox": str(args.inbox)})


if __name__ == "__main__": main()
