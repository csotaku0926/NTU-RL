import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC

# to use CNN feature extractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

# log to wandb
DO_WANDB = False
RUN_ID = "run_2layers_CNN_DQN"

# record past best episodes
DO_RECORD = True
_boards = []
_actions = []
_actions_str = ["Up", "Right", "Down", "Left"]

LOAD_BEST = True
LOAD_WHAT = "models/sample_model/best_3430_DQN"

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# 10.25.. PROBLEM is now the policy tend not to go left
# [[ 0  0  2 16]
#  [ 0  0  8 32]
#  [ 0  0  4 16]
#  [ 0  2  8  8]]
# Down
# dont use log reward its useless

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

"""for CnnPolicy"""
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        
        """network parameters"""
        # input channel depth (should be 16)
        self.input_d = observation_space.shape[0]
        # first conv layer channel depth
        self.d1 = 128
        # second conv
        self.d2 = 128
        # fc output
        self.fc_in_d = 2*4*self.d2*2 + 3*3*self.d2*2 + 4*3*self.d1*2 # = 7424
        self.fc_out_d = features_dim

        """CNN layers"""
        self.l1_cnn1 = torch.nn.Conv2d(self.input_d, self.d1, kernel_size=(1, 2)) 
        self.l1_cnn2 = torch.nn.Conv2d(self.input_d, self.d1, kernel_size=(2, 1)) 

        self.l2_cnn1 = torch.nn.Conv2d(self.d1, self.d2, kernel_size=(1, 2)) 
        self.l2_cnn2 = torch.nn.Conv2d(self.d1, self.d2, kernel_size=(2, 1))

        """fc layers"""
        self.fc = torch.nn.Linear(self.fc_in_d, self.fc_out_d, bias=True)

        self.relu = torch.nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # layer 1
        conv11 = self.l1_cnn1(observations)
        conv12 = self.l1_cnn2(observations)

        relu11 = self.relu(conv11) # (B, 128, 4, 3)
        relu12 = self.relu(conv12) # (B, 128, 3, 4)

        # layer 2
        conv21 = self.l2_cnn1(relu11) 
        conv22 = self.l2_cnn2(relu11) 
        conv23 = self.l2_cnn1(relu12) 
        conv24 = self.l2_cnn2(relu12) 

        relu21 = self.relu(conv21) # (B, 128, 4, 2)
        relu22 = self.relu(conv22) # (B, 128, 3, 3)
        relu23 = self.relu(conv23) # (B, 128, 3, 3)
        relu24 = self.relu(conv24) # (B, 128, 2, 4)

        B = relu11.shape[0] 
        # reshape & concat
        hidden11 = torch.reshape(relu11, (B, -1))
        hidden12 = torch.reshape(relu12, (B, -1))

        hidden21 = torch.reshape(relu21, (B, -1))
        hidden22 = torch.reshape(relu22, (B, -1))
        hidden23 = torch.reshape(relu23, (B, -1))
        hidden24 = torch.reshape(relu24, (B, -1))

        hidden = torch.cat((hidden11, hidden12, hidden21, hidden22, hidden23, hidden24), axis=1)

        # FC
        fc = self.fc(hidden)
        fc = self.relu(fc)
        return fc


POLICY_KWARGS = dict(
    features_extractor_class=CustomCNNExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": RUN_ID,
    "algorithm": DQN,
    "policy_network": "CnnPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 20000,
    "timesteps_per_epoch": 5_000, 
    "eval_episode_num": 20,
    "learning_rate": linear_schedule(5e-4), # try 5e-4, 1e-4, 5e-5
    "policy_kwargs": POLICY_KWARGS,
    "device": 'auto', # use cuda if possible, else cpu

    # DQN only
    "exploration_fraction": 0.1,
    "gamma": 0.9,
    "batch_size": 512,
    "buffer_size": 6000,
}


def make_env():
    env = gym.make('2048-v0', render_mode="ansi")
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0

    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            _boards.append(info[0]['board'])
            _actions.append(int(action))
        
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

        _boards.append("\n-----\n")
        _actions.append("")

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num

    return avg_score, avg_highest


def record_boards(boards: list, actions: list):
    """ simply write printed history boards into txt file """
    with open(f"record_boards_{RUN_ID}.txt", "w") as f:
        for board, a in zip(boards, actions):
            print(board, file=f)
            if (isinstance(a, int)):
                print(_actions_str[a], file=f)
            print(file=f)

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

            if (DO_RECORD):
                record_boards(_boards, _actions)
        _boards.clear()
        _actions.clear()

        print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    if (DO_WANDB):
        run = wandb.init(
            project="assignment_3",
            config=my_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            id=my_config["run_id"]
        )

    # Create training environment 
    num_train_envs = 2
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    # Create evaluation environment 
    eval_env = DummyVecEnv([make_env])  

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](  # PPO
        my_config["policy_network"], # "MlpPolicy"
        train_env, 
        verbose=2,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"],

        device=my_config["device"],
        
        # DQN only
        exploration_fraction=my_config["exploration_fraction"],
        gamma=my_config["gamma"],
        batch_size=my_config["batch_size"],
        buffer_size=my_config["buffer_size"],
    )

    alg_name = type(model).__name__
    if (LOAD_BEST and alg_name == "DQN"):
        custom_obj = {'learning_rate': my_config["learning_rate"]}
        model = DQN.load(LOAD_WHAT, train_env, custom_objects=custom_obj)

    elif (LOAD_BEST and alg_name == "PPO"):
        model = PPO.load(LOAD_WHAT, train_env)


    train(eval_env, model, my_config)