import matplotlib.pyplot as plt
import numpy as np
import wandb
import csv
import os # DO NOT SUBMIT THIS FILE

from gridworld import GridWorld
from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500
# log reward
N_SEED = 50
MAX_STEP = 512000
LOG_PER_EPISODE = 100
N_SMOOTHING = 50

def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    return grid_world

# ========================= task 1 ==============================
def _run_TD_prediction(grid_world: GridWorld, seed):
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.reset()
    grid_world.reset_step_count()
    return prediction.get_all_state_values()

def _run_MonteCarlo_Prediction(grid_world: GridWorld, seed: int):
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.reset()
    grid_world.reset_step_count()
    return prediction.get_all_state_values()

def _measure(estimated_values: np.array, GT_values: np.array):
    # measure variance and bias
    v_avg = np.mean(estimated_values, axis=0)
    v_var = np.var(estimated_values, axis=0)
    v_bias = v_avg - GT_values
    return v_var, v_bias

def _plot_mc_td(mc_var, mc_bias, td_var, td_bias):
    X_axis = np.arange(len(mc_var))

    # plot variance
    plt.bar(X_axis - 0.2, mc_var, 0.4, label="MC_var")
    plt.bar(X_axis, td_var, 0.4, label="TD_var")
    plt.xlabel("State")
    plt.ylabel("Variance")
    plt.title("Variance of prediction algorithms")
    plt.legend()
    plt.savefig("plots/Variance_bar_2.png")
    plt.show()

    # plot bias
    plt.bar(X_axis - 0.2, mc_bias, 0.4, label="MC_bias")
    plt.bar(X_axis, td_bias, 0.4, label="TD_bias")
    plt.xlabel("State")
    plt.ylabel("Bias")
    plt.title("Bias of prediction algorithms")
    plt.legend()
    plt.savefig("plots/Bias_bar_2.png")
    plt.show()

def _plot_mc_td_wandb(mc_var, mc_bias, td_var, td_bias):
    x_axis = np.arange(len(mc_var))
    wandb.init()

    data = [[x, var, var2] for (x, var, var2) in zip(x_axis, mc_var, td_var)]
    table = wandb.Table(
        data=data,
        columns=["x", "MC varaince", "TD variance"]
    )

    data2 = [[x, var, var2] for (x, var, var2) in zip(x_axis, mc_bias, td_bias)]
    table2 = wandb.Table(
        data=data2,
        columns=["x", "MC bias", "TD bias"]
    )
    wandb.log({
        "varaince table": table,
        "bias table": table2
    })


def task1():        
    grid_world = init_grid_world("maze.txt", INIT_POS)
    GT_values = np.load("sample_solutions/prediction_GT.npy")

    MC_estimated_values = []
    TD_estimated_values = []
    # run MC for 50 seeds
    for seed in range(N_SEED):
        mc_v = _run_MonteCarlo_Prediction(grid_world, seed)
        td_v = _run_TD_prediction(grid_world, seed)
        MC_estimated_values.append(mc_v)
        TD_estimated_values.append(td_v)

    MC_estimated_values = np.array(MC_estimated_values)
    TD_estimated_values = np.array(TD_estimated_values)

    mc_var, mc_bias = _measure(MC_estimated_values, GT_values)
    td_var, td_bias = _measure(TD_estimated_values, GT_values)
    print("MC var:", mc_var)
    print("MC bias:", mc_bias)
    print("TD var:", td_var)
    print("TD bias:", td_bias)

    # _plot_mc_td(mc_var, mc_bias, td_var, td_bias)
    _plot_mc_td_wandb(mc_var, mc_bias, td_var, td_bias)

# ==================== task 2 =================================

def _run_MC_policy_iteration(grid_world: GridWorld, iter_num: int, epsilon):
    print("run MC control\n")
    policy_iteration = MonteCarloPolicyIteration(
        grid_world, 
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        epsilon=epsilon,
    )
    
    policy_iteration.run(max_episode=iter_num, log_per_episode=LOG_PER_EPISODE)
    grid_world.reset()    
    return policy_iteration._log_losses, policy_iteration._log_rewards

def _run_SARSA(grid_world: GridWorld, iter_num: int, epsilon):
    print("run SARSA\n")
    policy_iteration = SARSA(
        grid_world, 
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        epsilon=epsilon,
    )
    policy_iteration.run(max_episode=iter_num, log_per_episode=LOG_PER_EPISODE)
    grid_world.reset()
    return policy_iteration._log_losses, policy_iteration._log_rewards

def _run_Q_Learning(grid_world: GridWorld, iter_num: int, epsilon):
    print("run Q_Learning Policy Iteration\n")
    policy_iteration = Q_Learning(
        grid_world, 
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        epsilon=epsilon,
        buffer_size=BUFFER_SIZE,
        update_frequency=UPDATE_FREQUENCY,
        sample_batch_size=SAMPLE_BATCH_SIZE,
    )
    policy_iteration.run(max_episode=iter_num, log_per_episode=LOG_PER_EPISODE)
    grid_world.reset()
    return policy_iteration._log_losses, policy_iteration._log_rewards

def plot_controls(mc_losses, mc_rewards, sarsa_losses, sarsa_rewards, q_losses, q_rewards, epsilon):
    x_axis = np.arange(1, len(mc_losses) + 1) * LOG_PER_EPISODE

    # plt.ylim((-0.13, 0.01))

    # plot learning curve (rewards)
    plt.plot(x_axis, mc_rewards, label="MC")
    plt.plot(x_axis, sarsa_rewards, label="SARSA")
    plt.plot(x_axis, q_rewards, label="Q")
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(f"plots/learning_curve_{int(epsilon)}x{int(epsilon * 100)}.png")
    plt.clf()

    # plot loss curve
    plt.plot(x_axis, mc_losses, label="MC")
    plt.plot(x_axis, sarsa_losses, label="SARSA")
    plt.plot(x_axis, q_losses, label="Q")
    plt.xlabel("episodes")
    plt.ylabel("estimation loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(f"plots/loss_curve_{int(epsilon)}x{int(epsilon * 100)}.png")
    plt.clf()

def plot_controls_wandb(mc_losses, mc_rewards, sarsa_losses, sarsa_rewards, q_losses, q_rewards, epsilon):
    x_axis = np.arange(1, len(mc_losses) + 1) * LOG_PER_EPISODE
    wandb.init()

    data = [[x, v, v2, v3] for (x, v, v2, v3) in zip(x_axis, mc_losses, sarsa_losses, q_losses)]
    table = wandb.Table(
        data=data,
        columns=["x", "MC loss", "sarsa loss", "q loss"]
    )

    data2 = [[x, v, v2, v3] for (x, v, v2, v3) in zip(x_axis, mc_rewards, sarsa_rewards, q_rewards)]
    table2 = wandb.Table(
        data=data2,
        columns=["x", "MC reward", "sarsa reward", "q reward"]
    )
    wandb.log({
        f"loss table {epsilon}": table,
        f"reward table {epsilon}": table2
    })

# to smooth the plot
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

def task2():
    grid_world = init_grid_world("maze.txt", INIT_POS)
    
    for epsilon in [0.1, 0.2, 0.3, 0.4]:
        print(f"\n----- epsilon = {epsilon} ------\n")
        mc_losses, mc_rewards = _run_MC_policy_iteration(grid_world, MAX_STEP, epsilon)
        sarsa_losses, sarsa_rewards = _run_SARSA(grid_world, MAX_STEP, epsilon)
        q_losses, q_rewards = _run_Q_Learning(grid_world, MAX_STEP, epsilon)

        # mc_losses = moving_average(mc_losses)
        # mc_rewards = moving_average(mc_rewards)
        # sarsa_losses = moving_average(sarsa_losses)
        # sarsa_rewards = moving_average(sarsa_rewards)
        # q_losses = moving_average(q_losses)
        # q_rewards = moving_average(q_rewards)

        plot_controls_wandb(mc_losses, mc_rewards, 
                      sarsa_losses, sarsa_rewards,
                      q_losses, q_rewards, epsilon)

def read_data(filename: str, n=3, log_name="table"):
    mc_data = []
    sarsa_data = []
    q_data = []

    wandb.init()
    with open(filename, newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)
        x_axis = np.arange(len(rows)-1)

        for datas in rows[1:]:
            mc_data.append(float(datas[1]))
            sarsa_data.append(float(datas[2]))
            q_data.append(float(datas[3]))

    mc_data = moving_average(mc_data, n)
    sarsa_data = moving_average(sarsa_data, n)
    q_data = moving_average(q_data, n)

    data = [[x, v, v2, v3] for (x, v, v2, v3) in zip(x_axis, mc_data, sarsa_data, q_data)]
    table = wandb.Table(
        data=data,
        columns=["x", "Monte-Carlo", "sarsa", "q learning"]
    )
    wandb.log({
        log_name: table,
    })

    
if __name__ == '__main__':
    # task1()
    # task2()
    for csv_filename in os.listdir("log_csv"):
        read_data(os.path.join("log_csv", csv_filename), n=N_SMOOTHING, log_name=csv_filename[:-4])