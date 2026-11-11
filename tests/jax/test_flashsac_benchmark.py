"""Regression checks for the MJX adapter and the published benchmark overrides."""

from pathlib import Path
import re

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_gym.algorithm.flashsac import _critic_context, _make_rollout
from tpu_gym.tasks.locomotion.g1.flat import flashsac_defaults
from flash_chord.training.flash_sac.config import NetworkConfig, ReplayConfig, TrainingConfig
from flash_chord.training.flash_sac.learner import create_learner


def test_defaults_match_upstream_launch_overrides():
    root = Path(__file__).resolve().parents[2]
    script = (root / "third_party/FlashSAC/scripts/run_mujoco_playground.sh").read_text()
    overrides = dict(re.findall(r"--overrides ([\w.]+)=([\d_.]+)", script))
    mapping = dict(
        num_envs="num_train_envs", num_timesteps="num_env_steps", discount="gamma",
        n_step="n_step", replay_capacity="agent.buffer_max_length",
        replay_warmup="agent.buffer_min_length", batch_size="agent.sample_batch_size",
        updates_per_collection="updates_per_interaction_step", num_eval_envs="num_eval_envs",
    )
    defaults = flashsac_defaults()
    for name, override in mapping.items():
        assert defaults[name] == float(overrides[override].replace("_", "")), name


@flax.struct.dataclass
class State:
    data: jax.Array
    obs: dict
    reward: jax.Array
    done: jax.Array
    info: dict


class TinyEnv:
    def reset(self, key):
        del key
        return State(
            data=jnp.float32(0), obs=self.observation(jnp.float32(0)),
            reward=jnp.float32(0), done=jnp.float32(0),
            info={"counter": jnp.int32(0)},
        )

    @staticmethod
    def observation(data):
        return {"state": data[None], "privileged_state": jnp.stack((data, data + 10))}

    def step(self, state, action):
        data = state.data + 1 + 100 * state.done
        return state.replace(
            data=data, obs=self.observation(data), reward=jnp.float32(1),
            done=(action[0] > 0).astype(jnp.float32),
            info={**state.info, "counter": state.info["counter"] + 1},
        )


def test_replay_uses_terminal_observation_and_cached_reset_keeps_task_info():
    reset, advance = _make_rollout(TinyEnv(), 2, episode_length=2)
    state = reset(jax.random.split(jax.random.PRNGKey(0), 2))
    # A timeout and true termination occur in the same collection.
    state = state.replace(
        data=jnp.full(2, 3.0), obs=jax.vmap(TinyEnv.observation)(jnp.full(2, 3.0)),
        info={**state.info, "counter": jnp.full(2, 7, dtype=jnp.int32)},
    )
    next_state, transition, age, key = advance(
        state, jnp.asarray([[0.0], [1.0]]), jnp.ones(2, dtype=jnp.int32), jax.random.PRNGKey(1),
    )
    np.testing.assert_array_equal(transition.terminated, [0, 1])
    np.testing.assert_array_equal(transition.truncated, [1, 0])
    np.testing.assert_array_equal(transition.next_observation, [[4], [4]])
    np.testing.assert_array_equal(transition.next_critic_context, [[14], [14]])
    np.testing.assert_array_equal(transition.post_reset_observation, [[0], [0]])
    np.testing.assert_array_equal(transition.post_reset_critic_context, [[10], [10]])
    np.testing.assert_array_equal(next_state.info["counter"], [8, 8])
    np.testing.assert_array_equal(age, [0, 0])
    _, following, _, _ = advance(next_state, jnp.zeros((2, 1)), age, key)
    np.testing.assert_array_equal(following.next_observation, [[1], [1]])


def test_critic_input_reconstructs_exact_privileged_observation():
    observation = {"state": jnp.ones((2, 103)), "privileged_state": jnp.ones((2, 216))}
    context = _critic_context(observation)
    assert context.shape == (2, 113)
    np.testing.assert_array_equal(
        jnp.concatenate((observation["state"], context), axis=-1), observation["privileged_state"],
    )


@pytest.mark.parametrize("context_width", [2, 113])
def test_generic_critic_context_does_not_apply_tracking_curriculum(context_width):
    config = TrainingConfig(
        world_count=2, total_environment_steps=16, mixed_precision=False,
        network=NetworkConfig(actor_hidden_dim=8, critic_hidden_dim=8, actor_blocks=1, critic_blocks=1),
        replay=ReplayConfig(capacity=8, minimum_size=2, batch_size=2, n_step=1),
    )
    context = jnp.ones((2, context_width), dtype=jnp.float32)
    learner, functions = create_learner(
        config, jnp.zeros((2, 3), dtype=jnp.float32), context,
        action_dim=1, objective_term_count=1, critic_context_names=None,
    )
    from flash_chord.training.flash_sac.learner import Transition

    transition = Transition(
        action=jnp.zeros((2, 1)), objective_terms=jnp.ones((2, 1)),
        terminated=jnp.zeros(2, dtype=jnp.int32), truncated=jnp.zeros(2, dtype=jnp.int32),
        episode_reference_progress=jnp.zeros(2), termination_diagnostics=jnp.zeros((2, 0)),
        next_observation=jnp.ones((2, 3)), next_critic_context=context,
        post_reset_observation=jnp.ones((2, 3)), post_reset_critic_context=context,
    )
    learner, _, metrics = functions.consume_transition(
        learner, transition, jnp.ones(1), jnp.float32(1), jnp.float32(4), jnp.bool_(True), jnp.bool_(False),
    )
    np.testing.assert_array_equal(learner.critic_context, context)
    assert int(metrics.critic_update_count) > 0
