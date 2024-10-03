import matplotlib.pyplot as plt
import numpy as np
import wandb

from gridworld import GridWorld
from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
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
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    # grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


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
    plt.savefig("Variance_bar.png")
    plt.show()

    # plot bias
    plt.bar(X_axis - 0.2, mc_bias, 0.4, label="MC_bias")
    plt.bar(X_axis, td_bias, 0.4, label="TD_bias")
    plt.xlabel("State")
    plt.ylabel("Bias")
    plt.title("Bias of prediction algorithms")
    plt.legend()
    plt.savefig("Bias_bar.png")
    plt.show()

if __name__ == '__main__':        
    grid_world = init_grid_world("maze.txt", INIT_POS)
    GT_values = np.load("sample_solutions/prediction_GT.npy")
    n_seed = 50

    MC_estimated_values = []
    TD_estimated_values = []
    # run MC for 50 seeds
    for seed in range(n_seed):
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

    _plot_mc_td(mc_var, mc_bias, td_var, td_bias)