import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback

RUN_ID = "p4_PPO"
DO_WANDB = False

POLICY_KWARGS = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(
        pi=[256, 256], # extra linear layer added to action network
        vf=[256, 256], # value network
    )
)

my_config = {
    "run_id": RUN_ID,
    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/p4_model",

    "epoch_num": 20000,
    "timesteps_per_epoch": 5_000, 
    "eval_episode_num": 20,
    "learning_rate": 1e-4, # try 5e-4, 1e-4, 5e-5
    "policy_kwargs": POLICY_KWARGS,
    "device": 'auto', # use cuda if possible, else cpu
}

# https://blog.csdn.net/CCCDeric/article/details/125428787
def make_lunar_env():
    env = gym.make("LunarLander-v2")
    return env

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):

        # Uncomment to enable wandb logging
        callback = None
        if (DO_WANDB):
            callback = WandbCallback(
                            gradient_save_freq=100,
                            verbose=2,
                        )

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False, 
            callback=callback,
        )
        

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        
        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()
        if (DO_WANDB):
            wandb.log(
                {"avg_highest": avg_highest,
                "avg_score": avg_score}
            )
            

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            alg_name = type(model).__name__
            model.save(f"{save_path}/best_{int(current_best)}_{alg_name}")

        print("---------------")

if __name__ == '__main__':
    # create training env
    num_train_envs = 2
    train_env = DummyVecEnv([make_lunar_env for _ in range(num_train_envs)])
    eval_env = DummyVecEnv([make_lunar_env])

    # define model
    model = my_config["algorithm"](
        policy=my_config["policy_network"],
        env=train_env,
        batch_size=my_config["batch_size"],
        learning_rate=my_config["learning_rate"],
        verbose=2,
        tensorboard_log=my_config["run_id"],
        device=my_config["device"],
        policy_kwards=my_config["policy_kwargs"],
    )