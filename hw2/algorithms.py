import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter += 1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
    
        G = 0
        state_returns_sum = np.zeros(self.state_space)
        # appears for how many episode for each state 
        state_n_times = np.zeros(self.state_space)
        state_is_visited = np.zeros(self.state_space, dtype=bool)
        episode_states = [] # list of (s_t, r_{t+1})

        while self.episode_counter < self.max_episode:
            # collect a episode
            next_state, reward, done = self.collect_data()
            episode_states.append((current_state, reward))
            current_state = next_state
            if (not done):
                continue
            
            # done one episode
            tmp = state_returns_sum.copy()
            for s_t, r_t in reversed(episode_states):
                G = self.discount_factor * G + r_t
                # add count if not visited
                if (not state_is_visited[s_t]):
                    state_is_visited[s_t] = True
                    state_n_times[s_t] += 1

                # first time of each state, update values and returns
                state_returns_sum[s_t] = tmp[s_t] + G
                self.values[s_t] = state_returns_sum[s_t] / state_n_times[s_t]

            # init for next episode
            G = 0
            state_is_visited = np.zeros(self.state_space, dtype=bool)
            episode_states = [] 


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            next_state, reward, done = self.collect_data()
            self.values[current_state] += self.lr * \
                (reward + self.discount_factor * self.values[next_state] * (1-done) - self.values[current_state])
            current_state = next_state


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        reward_deque = deque() # to store n rewards
        state_deque = deque() # to store n states
        done = False
        t = 0

        while (self.episode_counter < self.max_episode or len(state_deque)):
            # step forward
            if (not done):
                next_state, reward, done = self.collect_data()
                reward_deque.append(reward)
                state_deque.append(current_state)
                current_state = next_state
                t += 1

            # update V(s_t) with n rewards
            if (t < self.n):
                continue
            # G(n)_t = R_{t+1} + \gamma * R_{t+2} + ... + \gamma^{t+n-1} * R_{t+n} + \gamma^{t+n} * V(s_{t+n})
            G = 0
            for i, r_i in enumerate(reward_deque):
                G += (self.discount_factor ** i) * r_i
            G += (self.discount_factor ** self.n) * self.values[next_state] * (1 - done)
            s_t = state_deque[0]
            self.values[s_t] += self.lr * (G - self.values[s_t])

            reward_deque.popleft()
            state_deque.popleft()

            # start a new episode
            if (len(state_deque) == 0):
                done = False
                t = 0

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stochastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

        self.seed = 1
        self.rng = np.random.default_rng(self.seed)

        # only for logging rewards
        self._log_reward = 0
        self._log_rewards = []
        self._log_loss = 0
        self._log_losses = []
        self._do_log_ctr = 0
        self._do_log = False


    def collect_data(self):
        current_state = self.grid_world.get_current_state()

        policy = self.policy[current_state]
        action = self.rng.choice(self.action_space, p=policy)

        next_state, reward, done = self.grid_world.step(action)

        return next_state, action, reward, done

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        T = len(reward_trace)
        Gs = np.zeros(T)
        G = 0

        for i in range(T):
            j = T-1-i
            r_t = reward_trace[j]
            G = self.discount_factor * G + r_t
            Gs[j] = G
            
        for i in range(T):
            s_t = state_trace[i]
            a_t = action_trace[i]
            self.q_values[s_t, a_t] += self.lr * (Gs[i] - self.q_values[s_t, a_t])    

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        argmax_a = np.argmax(self.q_values, axis=1)

        # policy(s, a) = epsilon / m + 1 - epsilon, if a optimal
        #              = epsilon / m, otherwise
        self.policy = np.ones((self.state_space, self.action_space)) * self.epsilon / self.action_space
        for s in range(self.state_space):
            _a = argmax_a[s]
            self.policy[s][_a] += (1 - self.epsilon)


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []


        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            done = False

            # start an episode
            while (not done):
                next_state, action, reward, done = self.collect_data()
                state_trace.append(next_state)
                action_trace.append(action)
                reward_trace.append(reward)

            iter_episode += 1

            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()

            # clear traces
            current_state = self.grid_world.get_current_state()
            state_trace = [current_state]
            action_trace.clear()
            reward_trace.clear()

class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        self.q_values[s, a] += self.lr * \
            (r + self.discount_factor * self.q_values[s2, a2] * (1 - is_done) - self.q_values[s, a])
        
        # epsilon-greedy policy improvement (only affect (s, a))
        self.policy[s] = np.ones(self.action_space) * self.epsilon / self.action_space
        argmax_a = np.argmax(self.q_values[s])
        self.policy[s, argmax_a] += (1 - self.epsilon)


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            
            # sample an initial state and action first
            policy = self.policy[current_state]
            current_action = self.rng.choice(self.action_space, p=policy)

            # start an episode
            n_step = 0
            while (not is_done):
                # Take action A, observe R, S'
                next_state, prev_r, is_done = self.grid_world.step(current_action)

                # assign prev variables
                prev_s = current_state
                prev_a = current_action

                # decide next state and action 
                current_state = next_state
                policy = self.policy[current_state]
                current_action = self.rng.choice(self.action_space, p=policy)

                self.policy_eval_improve(prev_s, prev_a, prev_r, current_state, current_action, is_done)
                
                n_step += 1

            # restart
            iter_episode += 1
            is_done = False


class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size
        self.transition_count  = 0

        self._tmp_loss = 0
        self._tmp_reward = 0

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        random_indices = self.rng.choice(len(self.buffer), self.sample_batch_size)
        random_samples = [self.buffer[idx] for idx in random_indices]
        
        # keep the selected items in replay buffer
        
        return random_samples


    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        q2 = max(self.q_values[s2])

        self.q_values[s, a] += self.lr * \
            (r + self.discount_factor * q2 * (1 - is_done) - self.q_values[s, a])
        
        # epsilon-greedy policy improvement (only affect (s, a))
        self.policy[s] = np.ones(self.action_space) * self.epsilon / self.action_space
        argmax_a = np.argmax(self.q_values[s])
        self.policy[s, argmax_a] += (1 - self.epsilon)

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        self.transition_count = 0

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            while (not is_done):
                prev_s = current_state
                current_state, prev_a, prev_r, is_done = self.collect_data()
                
                # store transition
                self.add_buffer(prev_s, prev_a, prev_r, current_state, is_done)
                self.transition_count += 1

                # sample batch
                if (self.transition_count == self.update_frequency):
                    random_samples = self.sample_batch()
                    self.transition_count = 0

                    for s, a, r, s2, d in random_samples:
                        self.policy_eval_improve(s, a, r, s2, d)
            
            # restart
            iter_episode += 1
            is_done = False
