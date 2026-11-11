"""Run the G1 PPO experiment followed by FlashSAC and publish job status."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", choices=("ppo", "flashsac"), default=["ppo", "flashsac"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "sequence.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    stages = [
        {"algorithm": "ppo", "num_envs": 8192, "steps": 200_000_000},
        {"algorithm": "flashsac", "num_envs": 1024, "steps": 50_000_896},
    ]
    stages = [stage for stage in stages if stage["algorithm"] in args.algorithms]
    state = {"supervisor_pid": os.getpid(), "status": "starting", "stages": stages}
    current = None

    def publish():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = output / ".status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def stop(signum, frame):
        del signum, frame
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
            try:
                current.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(current.pid, signal.SIGKILL)
                current.wait()
        state["status"] = "stopped"
        publish()
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    python = root / ".venv/bin/python"
    publish()
    for stage in stages:
        # The published 10M-entry FlashSAC buffer alone needs ~17.2 GiB.
        env["XLA_CLIENT_MEM_FRACTION"] = "0.90" if stage["algorithm"] == "flashsac" else "0.25"
        stage_output = output / stage["algorithm"]
        stage_output.mkdir(exist_ok=True)
        command = [
            str(python), "-u", "train.py", "--task", "g1-flat",
            "--algorithm", stage["algorithm"], "--platform", "gpu",
            "--num-envs", str(stage["num_envs"]), "--steps", str(stage["steps"]),
            "--output", str(stage_output),
        ]
        stage.update(command=command, started_at=datetime.now(timezone.utc).isoformat())
        with (stage_output / "train.log").open("w") as log:
            current = subprocess.Popen(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            os.setpriority(os.PRIO_PROCESS, current.pid, 10)
            stage.update(pid=current.pid, status="running")
            state.update(status="running", active_algorithm=stage["algorithm"])
            publish()
            print(f"Started {stage['algorithm']} PID {current.pid}", flush=True)
            while current.poll() is None:
                try:
                    gpu = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                         "--format=csv,noheader,nounits", "-i", "0"],
                        text=True, timeout=5,
                    ).strip()
                    state["gpu"] = gpu
                except (OSError, subprocess.SubprocessError) as error:
                    state["gpu"] = str(error)
                publish()
                time.sleep(10)
            stage.update(
                exit_code=current.returncode,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status="complete" if current.returncode == 0 else "failed",
            )
        if current.returncode:
            state["status"] = "failed"
            publish()
            raise SystemExit(current.returncode)
        print(f"Completed {stage['algorithm']}", flush=True)
        publish()
    state.update(status="complete", active_algorithm=None)
    publish()


if __name__ == "__main__":
    main()
