"""Collect MJX rollouts for FlashCHORD's JAX FlashSAC learner."""

from dataclasses import asdict
from importlib.metadata import version
import json
import math
import platform

from tpu_gym._vendor import register_flash_chord

register_flash_chord()

import jax
import jax.numpy as jnp
from brax.training.acting import Evaluator
from mujoco_playground import wrapper
from flash_chord.training.flash_sac.checkpoint import save_learner_checkpoint
from flash_chord.training.flash_sac.config import NetworkConfig, ReplayConfig, TrainingConfig
from flash_chord.training.flash_sac.learner import Transition, create_learner
from flash_chord.training.flash_sac.network import Actor, deterministic_action
from flash_chord.training.flash_sac.replay import ReplaySpec, replay_memory


def _critic_context(obs):
    # Playground's privileged observation starts with the policy observation.
    return obs["privileged_state"][:, obs["state"].shape[-1]:]


def _make_rollout(env, num_envs, episode_length):
    raw_reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)

    @jax.jit
    def reset(keys):
        state = raw_reset(keys)
        # Match upstream AutoResetWrapper: cache only data/obs, retaining the
        # running RNG and task info across episode boundaries.
        state.info["benchmark_first_data"] = state.data
        state.info["benchmark_first_obs"] = state.obs
        return state

    @jax.jit
    def advance(state, action, age, key):
        terminal_state = step(state.replace(done=jnp.zeros_like(state.done)), action)
        age = age + 1
        terminated = terminal_state.done.astype(bool)
        truncated = (age >= episode_length) & ~terminated
        done = terminated | truncated

        def select(initial, current):
            mask = done.reshape(done.shape + (1,) * (current.ndim - 1))
            return jnp.where(mask, initial, current)

        next_state = terminal_state.replace(
            data=jax.tree.map(select, state.info["benchmark_first_data"], terminal_state.data),
            obs=jax.tree.map(select, state.info["benchmark_first_obs"], terminal_state.obs),
            done=done.astype(terminal_state.done.dtype),
        )
        transition = Transition(
            action=action,
            objective_terms=terminal_state.reward[:, None],
            terminated=terminated.astype(jnp.int32),
            truncated=truncated.astype(jnp.int32),
            episode_reference_progress=jnp.zeros_like(terminal_state.reward),
            termination_diagnostics=jnp.zeros((num_envs, 0), dtype=jnp.float32),
            next_observation=terminal_state.obs["state"],
            next_critic_context=_critic_context(terminal_state.obs),
            post_reset_observation=next_state.obs["state"],
            post_reset_critic_context=_critic_context(next_state.obs),
        )
        return next_state, transition, jnp.where(done, 0, age), key

    return jax.jit(reset), advance


def train(
    env, defaults, *, num_envs, steps, episode_length, seed, output, smoke,
    batch_size, replay_capacity, replay_warmup, num_evals,
):
    replay = ReplayConfig(
        capacity=replay_capacity or (64 if smoke else defaults["replay_capacity"]),
        minimum_size=replay_warmup or (4 if smoke else defaults["replay_warmup"]),
        batch_size=batch_size or (4 if smoke else defaults["batch_size"]),
        n_step=defaults["n_step"],
    )
    config = TrainingConfig(
        world_count=num_envs,
        total_environment_steps=steps,
        seed=seed,
        updates_per_collection=1.0 if smoke else defaults["updates_per_collection"],
        discount=defaults["discount"],
        mixed_precision=not smoke and jax.default_backend() == "gpu",
        compute_dtype="float16" if not smoke and jax.default_backend() == "gpu" else "float32",
        device=f"{jax.default_backend()}:0",
        network=NetworkConfig(actor_hidden_dim=32, critic_hidden_dim=64) if smoke else NetworkConfig(),
        replay=replay,
    )
    warmup_collections = (replay.minimum_size + num_envs - 1) // num_envs
    first_replay_step = (replay.n_step - 1 + warmup_collections) * num_envs
    if steps < first_replay_step:
        raise ValueError(f"FlashSAC needs at least {first_replay_step} steps to fill its replay warmup")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise ValueError(f"Output already contains an experiment: {output}")

    reset, advance = _make_rollout(env, num_envs, episode_length)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    state = reset(jax.random.split(reset_key, num_envs))
    age = jnp.zeros(num_envs, dtype=jnp.int32)
    context = _critic_context(state.obs)
    spec = ReplaySpec.from_config(
        replay, discount=config.discount, world_count=num_envs,
        observation_dim=state.obs["state"].shape[-1], action_dim=env.action_size,
        objective_term_count=1, critic_context_dim=context.shape[-1],
    )
    memory = replay_memory(spec)
    manifest = {
        "configuration": asdict(config),
        "environment": env._config.to_dict(),
        "actor_observation_dim": state.obs["state"].shape[-1],
        "critic_observation_dim": state.obs["privileged_state"].shape[-1],
        "reset_mode": "cached_data_and_observation",
        "num_eval_envs": min(defaults["num_eval_envs"], num_envs),
        "num_evals": num_evals or defaults["num_evals"],
        "episode_length": episode_length,
        "versions": {name: version(name) for name in ("jax", "brax", "flax", "mujoco", "mujoco-mjx", "playground")},
        "python": platform.python_version(),
        "devices": [str(device) for device in jax.devices()],
        "replay_memory": asdict(memory),
        "reference": "third_party/FlashSAC/scripts/run_mujoco_playground.sh",
        "limitation": "Uses Playground 0.2.0 / MuJoCo 3.11.0; upstream pins 0.0.5 / 3.3.3. Not a verified replication.",
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"replay_ring_gib={memory.ring_bytes / 2**30:.3f} discount={config.discount} n_step={replay.n_step}", flush=True)
    learner, functions = create_learner(
        config,
        state.obs["state"],
        context,
        action_dim=env.action_size,
        objective_term_count=1,
        critic_context_names=None,
    )
    actor = Actor(action_dim=env.action_size, config=config.network, dtype=jnp.dtype(config.compute_dtype))

    def make_policy(parameters):
        params, batch_stats = parameters

        def policy(observation, rng):
            del rng
            distribution = actor.apply(
                {"params": params, "batch_stats": batch_stats},
                observation["state"], training=False,
            )
            return deterministic_action(distribution), {}

        return policy

    evaluator = None
    if not smoke:
        evaluator = Evaluator(
            wrapper.wrap_for_brax_training(env, episode_length, full_reset=False),
            make_policy, min(defaults["num_eval_envs"], num_envs), episode_length, 1,
            jax.random.PRNGKey(seed + 1),
        )

    def record(step, metrics, evaluate=False):
        if evaluate and evaluator is not None:
            metrics = evaluator.run_evaluation(
                (learner.actor.params, learner.actor.batch_stats), metrics,
            )
            print(
                f"steps={step} eval_reward={metrics['eval/episode_reward']:.4f} "
                f"episode_length={metrics['eval/avg_episode_length']:.1f}",
                flush=True,
            )
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps({"steps": step, **{key: float(value) for key, value in metrics.items()}}) + "\n")

    if evaluator is not None:
        record(0, {}, evaluate=True)
    action, exploration, learner_key = functions.initial_action(
        learner.actor.params, learner.actor.batch_stats, learner.replay.size,
        learner.observation, learner.exploration, learner.key,
    )
    learner = learner.replace(exploration=exploration, key=learner_key)
    reward_weight = jnp.ones(1, dtype=jnp.float32)
    # Playground already integrates its reward over the control interval.
    reward_dt = jnp.float32(1)
    target_voc_scale = jnp.float32(0)
    stage_changed = jnp.bool_(False)
    reset_logging = jnp.bool_(False)
    log_every = max(1, config.collection_count // 10)
    eval_every = max(1, math.ceil(config.collection_count / max(1, (num_evals or defaults["num_evals"]) - 1)))
    metadata = {
        "network": json.dumps(asdict(config.network)),
        "compute_dtype": config.compute_dtype,
        "actor_observation": "state",
        "critic_observation": "privileged_state",
        "configuration": json.dumps(asdict(config)),
    }
    for index in range(1, config.collection_count + 1):
        state, transition, age, key = advance(state, action, age, key)
        learner, action, metrics = functions.consume_transition(
            learner, transition, reward_weight, reward_dt, target_voc_scale,
            stage_changed, reset_logging,
        )
        evaluate = not smoke and (index % eval_every == 0 or index == config.collection_count)
        if index % log_every == 0 or evaluate or index == config.collection_count:
            updates, loss, skips = jax.device_get((
                metrics.critic_update_count, metrics.critic_loss_mean_per_update,
                metrics.critic_skipped_fraction,
            ))
            print(
                f"steps={index * num_envs} critic_updates={updates} "
                f"critic_loss={loss:.4f} skipped={skips:.4f}",
                flush=True,
            )
            record(index * num_envs, {
                "training/critic_updates": updates,
                "training/critic_loss": loss,
                "training/critic_skipped_fraction": skips,
                "training/episode_reward": metrics.episode_return_mean_per_completed_episode,
                "training/episode_length": metrics.episode_length_mean_per_completed_episode,
            }, evaluate=evaluate)
        if evaluate or index == config.collection_count:
            save_learner_checkpoint(output, learner, metadata=metadata)
