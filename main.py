import tqdm, wandb, os, json, random, jax
from collections import defaultdict
from absl import app, flags
from ml_collections import config_flags
import numpy as np

from qso.agents import agents
from qso.utils.datasets import Dataset, GCDataset, HGCDataset, HRGCDataset, ReplayBuffer
from qso.utils.env_utils import make_env_and_datasets
from qso.utils.flax_utils import save_agent
from qso.utils.evaluation import evaluate
from qso.utils.log_utils import (
    CSVLogger,
    get_exp_name,
    get_flag_dict,
    setup_wandb,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_string("wandb_mode", "online", "wandb mode: online, offline or disabled.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "antmaze-giant-stitch-v0", "Environment (dataset) name.")
flags.DEFINE_string("dataset_dir", None, "Local OGBench dataset directory.")
flags.DEFINE_string("save_dir", "exp", "Save directory.",)

flags.DEFINE_integer("offline_train_steps", 1000000, "Number of offline steps.") 
flags.DEFINE_integer("online_train_steps", 1000000, "Number of online steps.")
flags.DEFINE_integer("online_start_training", 5000, "when online training starts")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 50000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_boolean("eval_on_start", True, "Whether to evaluate on start.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval.")

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes for each task.") 
flags.DEFINE_float("eval_temperature", 0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")

config_flags.DEFINE_config_file("agent", "qso/agents/qso.py", lock_config=False)


def main(_):
    # --- set up logging. ---
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project="OGBench", group=FLAGS.run_group, name=exp_name, mode=FLAGS.wandb_mode)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # --- data loading. ---
    config = FLAGS.agent
    env, offline_train_dataset, offline_val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.dataset_dir)
    eval_env = make_env_and_datasets(FLAGS.env_name, env_only=True)

    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
        "HRGCDataset": HRGCDataset,
    }[config["dataset_class"]]
    offline_train_dataset = dataset_class(Dataset.create(**offline_train_dataset), config, env, FLAGS.env_name)
    offline_val_dataset = dataset_class(Dataset.create(**offline_val_dataset), config, env, FLAGS.env_name)
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
        )
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)

    # --- initialise agent. ---
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    example_batch = offline_train_dataset.sample(1)
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # --- train agent. ---
    train_logger = CSVLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CSVLogger(os.path.join(FLAGS.save_dir, "eval.csv"))

    # --- offline RL. --- 
    for i in tqdm.tqdm(range(1, FLAGS.offline_train_steps + 1)):

        # --- for OPAL, freeze the option model once pretraining ends (pretraining steps count towards offline budget). ---
        if getattr(agent, "phase", None) == "pretrain" and i > config["rep_pretrain_steps"]:
            agent = agent.end_pretraining()

        # --- sample batch and update agent. ---
        batch = offline_train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        # --- log train and val metrics. ---
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            val_batch = offline_val_dataset.sample(config["batch_size"])
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update(
                {f"validation/{k}": v for k, v in val_info.items()}
            )
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # --- evaluate agent. ---
        if (FLAGS.eval_on_start and i == 1) or (i == FLAGS.offline_train_steps) or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            
            # --- create evaluation dictionary for each task. ---
            print(f"Evaluating agent at step {i} of offline training")
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            
            # --- evaluate on each task. --- 
            for eval_task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[eval_task_id - 1]["task_name"]

                # --- for the AC policy, during evaluation, the action chunk is executed fully. ---
                eval_info, _ = evaluate(
                    agent=eval_agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    task_id=eval_task_id,
                    num_eval_episodes=FLAGS.eval_episodes,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                metric_names = ["success"]

                print(f"Evaluation info: {eval_info}")
                eval_metrics.update(
                    {
                        f"evaluation/{task_name}_{k}": v
                        for k, v in eval_info.items()
                        if k in metric_names
                    }
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            
            # --- overall evaluation metrics. --- 
            for k, v in overall_metrics.items():
                eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

            # --- log metrics. --- 
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
        
        # --- save agent ---
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    # --- transition to online RL (create dataset and reset environment) ---
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(offline_train_dataset.dataset), size=max(FLAGS.buffer_size, offline_train_dataset.size + 1)
    )
    online_dataset = dataset_class(replay_buffer, config, env, FLAGS.env_name)
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    update_info = {}
    task_id = 1
    observation, reset_info = env.reset(options=dict(task_id=task_id))
    goal = reset_info.get("goal")
    action_queue = []
    options = None
    option_step = 0

    # --- online RL. ---
    for i in tqdm.tqdm(range(FLAGS.offline_train_steps + 1, FLAGS.offline_train_steps + FLAGS.online_train_steps + 1)):

        # --- for option policies, commit to each option for `subgoal_steps` steps. ---
        action_kwargs = {}
        if hasattr(agent, "sample_options"):
            if option_step % config["subgoal_steps"] == 0:
                online_rng, option_rng = jax.random.split(online_rng)
                options = agent.sample_options(observations=observation, goals=goal, seed=option_rng)
            action_kwargs["options"] = options
            option_step += 1

        # --- for chunked policies, execute all actions in the sequence. ---
        if len(action_queue) == 0:
            online_rng, action_rng = jax.random.split(online_rng)
            action = agent.sample_actions(
                observations=observation,
                goals=goal,
                seed=action_rng,
                temperature=FLAGS.eval_temperature,
                **action_kwargs,
                )
            action_queue.extend(np.array(action).reshape(-1, example_batch["actions"].shape[-1]))
        action = action_queue.pop(0)

        # --- step environment. ---
        next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
        done = terminated or truncated

        # --- handle transition. ---
        replay_buffer.add_transition(dict(
            observations=observation,
            actions=action,
            terminals=float(done),
            valids=1.0,
        ))
        if done:
            replay_buffer.add_transition(dict(
                observations=next_observation,
                actions=np.zeros_like(action),
                terminals=1.0,
                valids=0.0,
            ))
            replay_buffer.end_episode()
            task_id = task_id + 1 if task_id < num_tasks else 1
            observation, reset_info = env.reset(options=dict(task_id=task_id))
            goal = reset_info.get("goal")
            action_queue = []
            options = None
            option_step = 0
        else:
            observation = next_observation

        # --- update policy. ---
        if i >= FLAGS.offline_train_steps + FLAGS.online_start_training:
            batch = online_dataset.sample(config["batch_size"])
            agent, update_info = agent.update(batch)
            
        # --- log train and val metrics. ---
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            val_batch = offline_val_dataset.sample(config["batch_size"])
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update(
                {f"validation/{k}": v for k, v in val_info.items()}
            )
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # --- evaluate agent. ---
        if (i == FLAGS.offline_train_steps + FLAGS.online_train_steps) or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):

            # --- create evaluation dictionary for each task. ---
            print(f"Evaluating agent at step {i - FLAGS.offline_train_steps} of online training")
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            
            # --- evaluate on each task. --- 
            for eval_task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[eval_task_id - 1]["task_name"]

                # --- for the AC policy, during evaluation, the action chunk is executed fully. ---
                eval_info, _ = evaluate(
                    agent=eval_agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    task_id=eval_task_id,
                    num_eval_episodes=FLAGS.eval_episodes,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                metric_names = ["success"]

                print(f"Evaluation info: {eval_info}")
                eval_metrics.update(
                    {
                        f"evaluation/{task_name}_{k}": v
                        for k, v in eval_info.items()
                        if k in metric_names
                    }
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            
            # --- overall evaluation metrics. --- 
            for k, v in overall_metrics.items():
                eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

            # --- log metrics. --- 
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
        
        # --- save agent ---
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
    
    train_logger.close()
    eval_logger.close()

if __name__ == "__main__":
    app.run(main)