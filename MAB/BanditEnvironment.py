import numpy as np
from typing import Callable

class BanditEnvironment:
    '''
    A simple bandit environment with n actions.
    The true action values are sampled from a normal distribution with mean 0 and standard deviation 1.
    '''
    def __init__(self, n: int, q_dist_func: Callable=lambda n: np.random.normal(0, 1, n), stochastic: bool = True) -> None:
        '''
        Initialize the BanditEnvironment.

        Parameters
        ----------
        n : int
            Number of actions.
        q_dist_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the true action values. Defaults to np.random.normal(0, 1, n).
        '''
        self.n = n
        self.q_dist_func = q_dist_func
        self.q_values = q_dist_func(n)  if stochastic else np.arange(0, n) / n # Initialize the true action values with a normal distribution
        self.stochastic = stochastic

    def get_reward(self, action: int) -> float:
        '''
        Get the reward for a given action.

        Parameters
        ----------
        action : int
            The action for which to get the reward.

        Returns
        -------
        reward : float
            The reward sampled from a normal distribution with mean q_values[action] and standard deviation 1.
        '''
        reward = self.q_values[action] 
        if self.stochastic:
            reward += np.random.normal(0, 1)
        return reward
    
    def get_optimal_action(self) -> int:
        '''
        Get the optimal action.

        Returns
        -------
        optimal_action : int
            The action with the highest true action value.
        '''
        return np.argmax(self.q_values)
    
    def get_optimal_value(self) -> float:
        '''
        Get the optimal value.

        Returns
        -------
        optimal_value : float
            The highest true action value.
        '''
        # return np.max(self.q_values)
        return self.q_values
    
    def reset(self) -> None:
        '''
        Reset the environment.
        '''
        self.q_values = self.q_dist_func(self.n) if self.stochastic else np.arange(0, self.n) / self.n



class BanditNonstationary(BanditEnvironment):
    '''
    Nonstationary bandit environment with n actions.
    '''
    def __init__(self, n: int, q_dist_func: Callable=lambda n: np.random.normal(0, 1, n), 
                 random_walk_std: float = 0.01) -> None:
        '''
        Initialize the BanditNonstationary.

        Parameters
        ----------
        n : int
            Number of actions.
        q_dist_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the true action values. Defaults to np.random.normal(0, 1, n).
        random_walk_std : float, optional
            The standard deviation of the random walk for the true action values. Defaults to 0.01.
        '''
        super().__init__(n, q_dist_func)
        self.random_walk_std = random_walk_std

    def random_walk(self) -> None:
        '''
        Update the true action values with a random walk.
        '''
        self.q_values += np.random.normal(0, self.random_walk_std, self.n)

    def get_reward(self, action: int) -> float:
        '''
        Get the reward for a given action and update the true action values with a random walk.

        Parameters
        ----------
        action : int
            The action for which to get the reward.

        Returns
        -------
        reward : float
            The reward sampled from a normal distribution with mean q_values[action] and standard deviation 1.
        '''
        self.random_walk()
        return self.q_values[action] + np.random.normal(0, 1)


class AdversarialBandit(BanditEnvironment):
    '''
    Non-stochastic adversarial bandit environment with n actions.
    '''
    def __init__(self, n: int, reward_update_func: Callable = lambda q_values, t: q_values, q_dist_func: Callable = lambda n: np.random.rand(n)) -> None:
        '''
        Initialize the AdversarialBandit.

        Parameters
        ----------
        n : int
            Number of actions.
        reward_update_func : function, optional
            A function that takes the true action values [np.ndarray] and current time t as input and returns the adversarial rewards [np.ndarrary]. Defaults to do nothing.
        q_dist_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the true action values. Can be adversarialy chosen.
            Defaults to a np.random.rand(n).
        '''
        super().__init__(n, q_dist_func)
        self.reward_update_func = reward_update_func
        self.cummulative_rewards = np.zeros(n)
        self.t = 0

    def get_reward(self, action: int) -> float:
        '''
        First adversary choose rewards. Then get the reward for a given action.

        Parameters
        ----------
        action : int
            The action for which to get the reward.

        Returns
        -------
        reward : float
            The reward equal to the true action value.
        '''
        self.q_values = self.reward_update_func(self.q_values, self.t)
        self.cummulative_rewards += self.q_values
        self.t += 1
        return self.q_values[action]

    def get_optimal_action(self) -> int:
        '''
        Get the optimal action in hindsight.

        Returns
        -------
        optimal_action : int
            The action with the highest cummulative reward.
        '''
        return np.argmax(self.cummulative_rewards)
    
    def get_optimal_value(self) -> float:
        '''
        Get the optimal values in hindsight.

        Returns
        -------
        optimal_value : float
            The cummulative rewards.
        '''
        return self.cummulative_rewards.copy()

    def reset(self) -> None:
        '''
        Reset the environment.
        '''
        self.q_values = self.q_dist_func(self.n)
        self.cummulative_rewards = np.zeros(self.n)
        self.t = 0
    

class DeterministicBanditEnvironment:
    '''
    A simple bandit environment with n actions.
    The true action values are sampled from a normal distribution with mean 0 and standard deviation 1.
    '''
    def __init__(self, n: int) -> None:
        '''
        Initialize the BanditEnvironment.

        Parameters
        ----------
        n : int
            Number of actions.
        q_dist_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the true action values.
        '''
        self.n = n
        self.q_values = np.arange(0, n) / n

    def get_reward(self, action: int) -> float:
        '''
        Get the reward for a given action.

        Parameters
        ----------
        action : int
            The action for which to get the reward.

        Returns
        -------
        reward : float
            The reward sampled from a normal distribution with mean q_values[action] and standard deviation 1.
        '''
        reward = self.q_values[action]
        # print(f"Action: {action}, True Value: {self.q_values[action]}, Reward: {reward}")
        return reward
    
    def get_optimal_action(self) -> int:
        '''
        Get the optimal action.

        Returns
        -------
        optimal_action : int
            The action with the highest true action value.
        '''
        return np.argmax(self.q_values)
    
    def get_optimal_value(self) -> float:
        '''
        Get the optimal value.

        Returns
        -------
        optimal_value : float
            The highest true action value.
        '''
        return np.max(self.q_values)
    
    def reset(self) -> None:
        '''
        Reset the environment.
        '''
        self.q_values = np.arange(0, self.n) / self.n
