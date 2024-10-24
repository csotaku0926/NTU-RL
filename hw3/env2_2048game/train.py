import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC

# log to wandb
DO_WANDB = False
RUN_ID = "run_3_tol_DQN"

# record past best episodes
DO_RECORD = True
_boards = []
_actions = []
_actions_str = ["Up", "Right", "Down", "Left"]

LOAD_BEST = True
LOAD_WHAT = "models/sample_model/best_2871_512_DQN"

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": RUN_ID,
    "algorithm": DQN,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 20000,
    "timesteps_per_epoch": 1000,
    "eval_episode_num": 20,
    "learning_rate": 5e-5, # try 5e-4, 1e-4, 5e-5
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
    )

    alg_name = type(model).__name__
    if (LOAD_BEST and alg_name == "DQN"):
        custom_obj = {'learning_rate': my_config["learning_rate"]}
        model = DQN.load(LOAD_WHAT, train_env, custom_objects=custom_obj)

    elif (LOAD_BEST and alg_name == "PPO"):
        model = PPO.load(LOAD_WHAT, train_env)


    train(eval_env, model, my_config)