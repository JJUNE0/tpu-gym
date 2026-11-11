"""Train a Playground task with Brax PPO."""

from functools import partial
import json

import jax
from brax.training.agents.ppo import networks, train as ppo
from mujoco_playground import wrapper


def train(env, defaults, *, num_envs, steps, episode_length, seed, output, smoke, batch_size, num_evals):
    settings = dict(defaults)
    network_settings = dict(settings.pop("network_factory"))
    # Brax 0.14.2 cannot reload the default null mean initializer from a checkpoint.
    network_settings.setdefault("mean_kernel_init_fn", jax.nn.initializers.lecun_uniform)
    devices = 1 if smoke else jax.local_device_count()
    if num_envs % devices:
        raise ValueError(f"PPO --num-envs must be divisible by the number of local devices ({devices})")

    settings.update(
        num_envs=num_envs,
        num_timesteps=steps,
        episode_length=episode_length,
        batch_size=batch_size or (num_envs if smoke else defaults["batch_size"]),
        num_minibatches=1 if smoke else defaults["num_minibatches"],
        num_evals=num_evals or (1 if smoke else defaults["num_evals"]),
        num_eval_envs=min(32, num_envs),
        deterministic_eval=True,
        seed=seed,
        max_devices_per_host=devices,
        save_checkpoint_path=str(output),
        run_evals=not smoke,
    )
    if smoke:
        settings.update(unroll_length=2, num_updates_per_batch=1)
        network_settings = {
            **network_settings,
            "policy_hidden_layer_sizes": (32, 32),
            "value_hidden_layer_sizes": (32, 32),
        }
    if settings["batch_size"] * settings["num_minibatches"] % num_envs:
        raise ValueError("PPO batch size times the number of minibatches must be divisible by --num-envs")

    output.mkdir(parents=True, exist_ok=True)

    def progress(step, metrics):
        record = {"steps": int(step), **{key: float(value) for key, value in metrics.items()}}
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        if "eval/episode_reward" in metrics:
            print(
                f"steps={step} eval_reward={metrics['eval/episode_reward']:.4f} "
                f"episode_length={metrics['eval/avg_episode_length']:.1f}",
                flush=True,
            )

    ppo.train(
        environment=env,
        wrap_env_fn=partial(wrapper.wrap_for_brax_training, full_reset=True),
        network_factory=partial(networks.make_ppo_networks, **network_settings),
        progress_fn=progress,
        **settings,
    )
