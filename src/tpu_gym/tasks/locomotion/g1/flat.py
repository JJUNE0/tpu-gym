"""Unitree G1 flat-ground velocity tracking from MuJoCo Playground."""

from mujoco_playground import locomotion
from mujoco_playground.config import locomotion_params

PLAYGROUND_NAME = "G1JoystickFlatTerrain"


def make_environment(*, algorithm="ppo"):
    config = locomotion.get_default_config(PLAYGROUND_NAME)
    if algorithm == "flashsac":
        # FlashSAC's run_mujoco_playground.sh uses the no-push benchmark.
        config.push_config.enable = False
        config.push_config.magnitude_range = [0.0, 0.0]
    env = locomotion.load(PLAYGROUND_NAME, config=config, config_overrides={"impl": "jax"})
    return env, config.episode_length


def ppo_defaults():
    return locomotion_params.brax_ppo_config(PLAYGROUND_NAME, impl="jax").to_dict()


def flashsac_defaults():
    """Resolved overrides in FlashSAC/scripts/run_mujoco_playground.sh."""
    return dict(
        num_envs=1024,
        num_timesteps=50_000_896,
        discount=0.97,
        n_step=1,
        replay_capacity=10_000_000,
        replay_warmup=100_000,
        batch_size=2048,
        updates_per_collection=2.0,
        num_eval_envs=50,
        num_evals=11,
    )
