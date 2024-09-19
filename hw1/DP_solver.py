import numpy as np

from gridworld import GridWorld

class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # q_\pi(s, a) = r(s,a) + \gamma \sum_s' P(s, s', a) v(s')
        # TODO: Get reward from the environment and calculate the q-value
        values = self.get_values()
        next_s, reward_s_a, done = self.grid_world.step(state, action)
        q_value = reward_s_a + self.discount_factor * values[next_s] * (1 - done)
        return q_value


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Evaluates a non-deterministic policy -- policy is prob. distribution
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state
        V_k+1(s) = \sum_a pi(s, a) \sum_s' p(s'|s, a) [r_s,a + \gamma V_k(s')] 

        Args:
            state (int)

        Returns:
            float: value
        """
        # V(s) = \sum_a \pi(s, a) * q(s, a)
        # TODO: Get the value for a state by calculating the q-values
        action_space = self.grid_world.get_action_space()
        policy = self.get_policy()

        q_for_as = [self.get_q_value(state, a) for a in range(action_space)]
        q_for_as = np.array(q_for_as)

        return np.dot(policy[state, :], q_for_as)

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        state_space = self.grid_world.get_state_space()
        values = self.get_values()
        new_values = np.empty(self.values.shape)
        delta = 0

        for s in range(state_space):
            new_values[s] = self.get_state_value(s)
            delta = max(delta, abs(new_values[s] - values[s]))
        self.values = new_values
        return delta

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while (True):
            delta = self.evaluate()
            if (delta < self.threshold):
                return


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration
        policy iteration finds an optimal deterministic policy -- policy maps state to certain action

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        # V(s) = \sum_{s',r} p(s',r | s, \pi(s)) q(s, \pi(s))
        policy = self.get_policy()
        return self.get_q_value(state, policy[state])

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        delta = 0
        values = self.get_values()
        new_values = np.empty(values.shape)
        state_space = self.grid_world.get_state_space()

        for s in range(state_space):
            new_values[s] = self.get_state_value(s)
            delta = max(delta, abs(new_values[s] - values[s]))
        self.values = new_values
        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        # pi(s) <-- argmax_a \sum_{s', r} p(s',r | s, a) q(s, a)
        policy = self.get_policy()
        policy_stable = True    
        state_space = self.grid_world.get_state_space()
        action_space = self.grid_world.get_action_space()

        for s in range(state_space):
            q_for_as = [self.get_q_value(s, a) for a in range(action_space)]
            new_policy = np.argmax(q_for_as)
            if (new_policy != policy[s]):
                policy_stable = False
            policy[s] = new_policy
        
        return policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while (True):
            # part 2
            delta = self.policy_evaluation()
            if (delta >= self.threshold):
                continue
            # part 3
            policy_stable = self.policy_improvement()
            if (policy_stable):
                return


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        output deterministic policy based on finding optimal value
        given V*(s'), we can derive V*(s):
        V*(s) = max_a R(s, a) + \gamma * \sum_s' P(s, a, s') V*(s')

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action_space = self.grid_world.get_action_space()
        q_for_as = [self.get_q_value(state, a) for a in range(action_space)]
        return max(q_for_as)

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        delta = 0
        values = self.get_values()
        new_values = np.empty(values.shape)
        state_space = self.grid_world.get_state_space()
        
        for s in range(state_space):
            new_values[s] = self.get_state_value(s)
            delta = max(delta, abs(new_values[s] - values[s]))

        self.values = new_values
        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        state_space = self.grid_world.get_state_space()
        action_space = self.grid_world.get_action_space()

        q_state_action = []
        for s in range(state_space):
            q_state_action.append(
                [self.get_q_value(s, a) for a in range(action_space)]
            )

        q_state_action = np.array(q_state_action)
        return np.argmax(q_state_action, axis=1)


    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while (True):
            delta = self.policy_evaluation()
            if (delta < self.threshold):
                break
        self.policy = self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        - in-place value iteration (dynamic programming)
        - prioritized sweeping
        - real-time value iteration (dynamic programming)

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        
        np.random.seed(13921)
        self.state_space = self.grid_world.get_state_space()
        self.action_space = self.grid_world.get_action_space()
        self.n_steps = 3
        self.q_values = np.zeros((self.state_space, self.action_space))
        self.predecessors = dict()
        self.model = dict()
        # self.pq = PriorityQueue()
        self.pq = np.zeros(self.state_space * self.action_space)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        # self.run_inplace_value_iter()
        self.run_prioritized_sweeping()
        # self.run_novel_method()

    """method 1"""
    def run_inplace_value_iter(self):
        """Run in-place value iteration"""
        while (True):
            delta = self.policy_evaluation()
            if (delta < self.threshold):
                break
        self.policy = self.policy_improvement()

    def get_state_value(self, state: int) -> float:
        """max_a r(s,a) + \gamma * v(s')"""
        q_for_as = [self.get_q_value(state, a) for a in range(self.action_space)]
        return max(q_for_as)

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        delta = 0
        values = self.get_values()
        state_space = self.grid_world.get_state_space()
        
        # async update
        for s in range(state_space):
            old_values = values[s]
            values[s] = self.get_state_value(s)
            delta = max(delta, abs(values[s] - old_values))

        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        state_space = self.grid_world.get_state_space()
        action_space = self.grid_world.get_action_space()

        q_state_action = []
        for s in range(state_space):
            q_state_action.append(
                [self.get_q_value(s, a) for a in range(action_space)]
            )

        q_state_action = np.array(q_state_action)
        return np.argmax(q_state_action, axis=1)
    
    """method 2 & 3"""
    def run_prioritized_sweeping(self):
        """run q-learning episode"""
        deltas = np.ones(self.state_space) * 100

        while (True):
            for s in range(self.state_space):
                self.pq = np.zeros(self.state_space * self.action_space)

                if (deltas[s] >= self.threshold):
                    deltas[s] = self.run_MDP_episode(start_state=s)
            
            if (np.sum(deltas < self.threshold) == len(deltas)):
                break
            
        self.policy = self.policy_improvement()

    
    def run_novel_method(self):
        """run my novel method"""
        delta = 0

        while (True):
            self.pq = np.zeros(self.state_space * self.action_space)

            delta = self.run_MDP_episode()
            
            if (delta < self.threshold):
                break

        self.policy = self.policy_improvement()

    def get_best_action(self, state):
        action_space = self.grid_world.get_action_space()
        q_values = [self.q_values[state, a] for a in range(action_space)]
        max_q = np.max(q_values)
        epsilon = 1e-6
        # if multiple actions with max q, randomly pick one
        best_actions = [ a for i, a in enumerate(range(action_space)) if abs(q_values[i]-max_q) < epsilon ]
        best_idx = np.random.choice(len(best_actions))
        return best_actions[best_idx]

    def compute_value(self, state):
        best_action = self.get_best_action(state)
        return self.q_values[state, best_action]

    def update_predecessors(self, state, action, next_state):
        if (next_state not in self.predecessors):
            self.predecessors[next_state] = []
        self.predecessors[next_state].append((state, action))

    def get_from_pq(self):
        """Priority Queue is not suffice, we need dynamic update"""
        argmax_state = np.argmax(np.abs(self.pq))
        if (self.pq[argmax_state] == 0):
            return -1
        else:
            return argmax_state
    
    def get_pq_by_idx(self, s, a):
        return self.pq[s * self.action_space + a]

    def set_pq_by_idx(self, s, a, data):
        self.pq[s * self.action_space + a] = data

    def update_ps(self, state, action, reward, next_state, done):
        """one-step updating"""
        delta = 0

        # update reward
        self.model[(state, action)] = (reward, next_state)
        # update predecessors
        self.update_predecessors(state, action, next_state)
        
        # calculate priority and put to pq
        priority = reward + self.discount_factor * self.compute_value(next_state) * (1-done) - self.q_values[state, action] 
        if (abs(priority) > self.threshold):
            self.set_pq_by_idx(state, action, priority)
        
        # update `n_steps` largest elements while pq is not empty
        for _ in range(self.n_steps):
            _idx = self.get_from_pq()
            if (_idx == -1):
                break
            _state = _idx // self.action_space
            _action = _idx % self.action_space
            
            _p = self.pq[_idx]
            delta = max(delta, abs(_p))
            
            # update value of this state-action pair
            self.q_values[_state][_action] += _p 

            # set priority to 0
            self.set_pq_by_idx(_state, _action, 0)

            # loop all preceding state
            if (_state not in self.predecessors):
                continue
            for s, a in self.predecessors[_state]:
                r, _ = self.model[(s, a)]
                _prirority = r + self.discount_factor * self.compute_value(_state) - self.q_values[s, a]
                if (abs(_prirority) > self.threshold):
                    self.set_pq_by_idx(s, a, _prirority)
        
        return delta

    def run_MDP_episode(self, start_state=0):
        state = start_state
        done = False
        delta = 0

        while (not done):
            action = self.get_best_action(state)
            next_state, reward, done = self.grid_world.step(state, action)
            d = self.update_ps(state, action, reward, next_state, done)
            delta = max(delta, d)
            state = next_state
        
        for s in range(self.state_space):
            self.values[s] = self.compute_value(s)

        return delta

    