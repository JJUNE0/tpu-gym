"""Command-line entry point for JAX training."""

import argparse
from datetime import datetime
from importlib import import_module
import os
from pathlib import Path

TASKS = {"g1-flat": "tpu_gym.tasks.locomotion.g1.flat"}


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "flashsac"), required=True)
    parser.add_argument("--task", choices=TASKS, default="g1-flat")
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu", "tpu"), default="auto")
    parser.add_argument("--num-envs", type=_positive_int)
    parser.add_argument("--steps", type=_positive_int, help="Total environment transitions")
    parser.add_argument("--episode-length", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--num-evals", type=_positive_int, help="Evaluation and checkpoint count")
    parser.add_argument("--replay-capacity", type=_positive_int, help="FlashSAC replay capacity")
    parser.add_argument("--replay-warmup", type=_positive_int, help="FlashSAC minimum replay size")
    parser.add_argument("--seed", type=int, help="Default: PPO 42, FlashSAC 0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Use a small training run with reduced network sizes")
    args = parser.parse_args()
    if args.algorithm == "ppo" and (args.replay_capacity or args.replay_warmup):
        parser.error("replay options apply to FlashSAC")

    if args.platform != "auto":
        # The locked GPU extra supplies CUDA. JAX's gpu alias also tries ROCm.
        os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else args.platform

    import jax

    try:
        devices = jax.devices()
    except (AssertionError, RuntimeError) as error:
        parser.error(str(error) or f"No JAX device available for --platform {args.platform}")
    print(f"JAX backend={jax.default_backend()} devices={devices}", flush=True)
    task = import_module(TASKS[args.task])
    env, default_episode_length = task.make_environment(algorithm=args.algorithm)
    ppo_defaults = task.ppo_defaults()
    defaults = ppo_defaults if args.algorithm == "ppo" else task.flashsac_defaults()
    default_envs = defaults["num_envs"]
    default_steps = defaults["num_timesteps"]
    num_envs = args.num_envs or (2 if args.smoke else default_envs)
    steps = args.steps or (16 if args.smoke else default_steps)
    episode_length = args.episode_length or (4 if args.smoke else default_episode_length)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or Path("outputs/jax") / args.task / args.algorithm / timestamp
    settings = dict(
        num_envs=num_envs, steps=steps, episode_length=episode_length,
        seed=args.seed if args.seed is not None else (42 if args.algorithm == "ppo" else 0),
        output=output.resolve(), smoke=args.smoke, batch_size=args.batch_size,
    )
    print(f"task={args.task} algorithm={args.algorithm} envs={num_envs} target_steps={steps}", flush=True)
    try:
        if args.algorithm == "ppo":
            from tpu_gym.algorithm import ppo

            ppo.train(env, ppo_defaults, **settings, num_evals=args.num_evals)
        else:
            from tpu_gym.algorithm import flashsac

            flashsac.train(
                env, defaults, **settings, replay_capacity=args.replay_capacity,
                replay_warmup=args.replay_warmup, num_evals=args.num_evals,
            )
    except ValueError as error:
        parser.error(str(error))
    print(f"Checkpoints saved in {output}", flush=True)


if __name__ == "__main__":
    main()
