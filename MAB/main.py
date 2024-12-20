'''
Train and evaluate different bandit algorithms.
'''

import numpy as np
from Algorithms import EpsilonGreedy, Algorithm, UCB, GradientBandit, EXP3, EXP3IX, EXP3IXrl
from BanditEnvironment import BanditEnvironment
from tqdm import tqdm
from typing import Union, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json

np.random.seed(0)
np.seterr(divide = 'ignore')

def main():
    num_train_steps = 10000
    num_eval_steps = 30
    num_runs = 100
    gt_algo = EXP3IXrl
    rl_algos = [
        (EXP3, (), {'time_horizon':num_train_steps}),
        (EXP3IX, (), {'time_horizon':num_train_steps}),
        (GradientBandit, (), {'alpha': 0.1, 'baseline': True}),
        (UCB, (), {'c': 2}),
        (EpsilonGreedy, (), {'epsilon': 0.1})
    ]
    zetas = [0, 5, 50, 100, 200, 500, 1000, 1500, 2000]
    path = os.path.join(os.curdir,'results','MAB')
    os.makedirs(path,exist_ok=True)
    gather_test_statistics(num_train_steps,num_eval_steps,num_runs,gt_algo,rl_algos,zetas,path)

    np.random.seed(0)
    num_train_steps = 5000
    num_eval_steps = 100
    active = False
    color = ['b','r','g']
    rl_algos = [
        (GradientBandit, (), {'alpha': 0.1, 'baseline': True}),
        (UCB, (), {'c': 2}),
        (EpsilonGreedy, (), {'epsilon': 0.1})
    ]
    visualize_stochastic_environment_learning(num_train_steps,num_eval_steps,num_runs,gt_algo,rl_algos,zetas,path,color,active)


def evaluate(bandit, gt_algorithm, rl_algorithm, num_eval_steps, certainty):
    ## EVALUATION ##
    gt_rewards = np.zeros((num_eval_steps))
    rl_rewards = np.zeros((num_eval_steps))
    gt_actions = np.zeros((num_eval_steps),dtype=int)
    rl_actions = np.zeros((num_eval_steps),dtype=int)

    rl_state_rewards = [0] * num_eval_steps
    gt_state_rewards = [0] * num_eval_steps
    for j in range(num_eval_steps):

        ## GT ##
        state_action = np.random.get_state()
        equalibrum, visits = gt_algorithm.get_equilibrium()
        gt_action = gt_algorithm.action_selection(equalibrum, visits, certainty)
        gt_actions[j] = int(gt_action)
        state_reward = np.random.get_state()
        gt_rewards[j] = bandit.get_reward(gt_action)
        gt_state_rewards[j] = bandit.get_optimal_value()

        # RL
        np.random.set_state(state_action)
        rl_action = rl_algorithm.select_action()
        rl_actions[j] = int(rl_action)
        np.random.set_state(state_reward)
        rl_rewards[j] = bandit.get_reward(rl_action)
        rl_state_rewards[j] = bandit.get_optimal_value()
    return (gt_rewards, gt_actions, gt_state_rewards), (rl_rewards,rl_actions,rl_state_rewards)

def train(bandit, gt_algorithm, rl_algorithm, num_train_steps):
    #TRAIN
    for _ in range(num_train_steps):
        action = rl_algorithm.select_action()
        reward = bandit.get_reward(action)
        rl_algorithm.train(action, reward)
        gt_algorithm.train(action, reward)
    return gt_algorithm, rl_algorithm

def training_dynamics(bandit, gt_algorithm, rl_algorithm, num_steps, certainty):
    gt_rewards = np.zeros(num_steps)
    rl_rewards = np.zeros(num_steps)

    gt_actions = np.zeros((num_steps),dtype=int)
    rl_actions = np.zeros((num_steps),dtype=int)

    gt_state_rewards = [0] * num_steps
    rl_state_rewards = [0] * num_steps

    for i in range(num_steps):
        ## GT ##
        action_state = np.random.get_state()
        equalibrum, visits = gt_algorithm.get_equilibrium()
        action = gt_algorithm.action_selection(equalibrum, visits, certainty)
        if action == -1:
            action = rl_algorithm.select_action()
        gt_actions[i] = action # Prepare for calculating the optimal action

        reward_state = np.random.get_state()
        gt_rewards[i] = bandit.get_reward(action)
        gt_state_rewards[i] = bandit.get_optimal_value() # If regret - returns cumulative rewards
        gt_algorithm.train(action, gt_rewards[i])
        
        np.random.set_state(action_state)
        ## RL ##
        rl_action = rl_algorithm.select_action()
        rl_actions[i] = rl_action # Prepare for calculating the optimal action

        np.random.set_state(reward_state)
        rl_rewards[i] = bandit.get_reward(rl_action)
        rl_state_rewards[i] = bandit.get_optimal_value() # If regret - returns cumulative rewards
        rl_algorithm.train(rl_action, rl_rewards[i])

    return (gt_rewards, gt_actions, gt_state_rewards), (rl_rewards, rl_actions, rl_state_rewards)

def compute_statistics(bandit, gt, rl, num_eval_steps):
    (_, gt_actions, gt_state_rewards) = gt
    (_, rl_actions, rl_state_rewards) = rl
    optimal_action = bandit.get_optimal_action()

    steps = np.arange(num_eval_steps)
    rl_optimal_rewards = np.array(rl_state_rewards)[steps, optimal_action]
    gt_optimal_rewards = np.array(gt_state_rewards)[steps, optimal_action]
    rl_action_rewards = np.array(rl_state_rewards)[steps, rl_actions]
    gt_action_rewards = np.array(gt_state_rewards)[steps, gt_actions]
              
    gt_weak_regret_measures = np.cumsum(gt_optimal_rewards) - np.cumsum(gt_action_rewards) # Weak regret
    rl_weak_regret_measures = np.cumsum(rl_optimal_rewards) - np.cumsum(rl_action_rewards) # Weak regret       

    gt_rewards_measures = np.cumsum(gt_action_rewards)
    rl_rewards_measures = np.cumsum(rl_action_rewards)
    
    gt_percents_optimal = gt_actions == optimal_action
    rl_percents_optimal = rl_actions == optimal_action

    return (gt_weak_regret_measures,rl_weak_regret_measures), (gt_rewards_measures,rl_rewards_measures), (gt_percents_optimal, rl_percents_optimal)


def gather_visualization_statistics(
        bandit: BanditEnvironment,
        num_train_steps: int,
        num_eval_steps: int,
        num_runs: int,
        rl_algorithm: Algorithm,
        gt_algorithm: Algorithm,
        certainty: int,
        active: bool = False,
        *args, 
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
    n = bandit.n

    rl_algorithm = rl_algorithm(n, *args, **kwargs)
    gt_algorithm = gt_algorithm()

    ##  TRAINING   ##
    # Train the gt_algorithm with the rl_algorithm
    bandit.reset()

    ## EVALUATION  ##
    rl_average_measures = np.zeros(num_eval_steps)
    rl_percents_optimal = np.zeros(num_eval_steps)
    gt_average_measures = np.zeros(num_eval_steps)
    gt_percents_optimal = np.zeros(num_eval_steps)
    
    for _ in tqdm(range(num_runs), desc=f'Training {str(gt_algorithm)} with {str(rl_algorithm)}', leave=False, position=1):
        if active:
            # Allow active data collection
            gt, rl = training_dynamics(bandit,gt_algorithm,rl_algorithm,num_train_steps,certainty)
        else:
            # Pretrain the algorithm and gather training statistics (Passive)
            gt_algorithm, rl_algorithm = train(bandit, gt_algorithm, rl_algorithm, num_train_steps)
            gt, rl = evaluate(bandit,gt_algorithm,rl_algorithm,num_eval_steps,certainty)
        (gt_weak_regret_measures,rl_weak_regret_measures), _, (gt_percents, rl_percents) = compute_statistics(bandit,gt,rl,num_eval_steps=num_eval_steps)
        
        gt_average_measures+=gt_weak_regret_measures
        rl_average_measures+=rl_weak_regret_measures
        gt_percents_optimal+=gt_percents
        rl_percents_optimal+=rl_percents

        bandit.reset()
        rl_algorithm.reset()
        gt_algorithm.reset()

    rl_average_measures /= num_runs
    rl_percents_optimal /= num_runs
    gt_average_measures /= num_runs
    gt_percents_optimal /= num_runs
    return (gt_average_measures, gt_percents_optimal), (rl_average_measures, rl_percents_optimal)

def train_and_evaluate(
        bandit: BanditEnvironment,
        num_train_steps: int,
        num_eval_steps: int,
        num_runs: int,
        rl_algorithm_class: Algorithm,
        gt_algorithm_class: Algorithm,
        certainty: int, 
        *args, 
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
    n = bandit.n

    rl_algorithm = rl_algorithm_class(n, *args, **kwargs)
    gt_algorithm = gt_algorithm_class()

    gt_rewards_measures = np.zeros((num_runs))
    rl_rewards_measures = np.zeros((num_runs))

    gt_regret_measures = np.zeros((num_runs))
    rl_regret_measures = np.zeros((num_runs))

    bandit.reset()
    for i in tqdm(range(num_runs), desc=f'Training {str(gt_algorithm)} with {str(rl_algorithm)}', leave=False, position=1):
        # Pretrain the algorithm and evaluate on a frozen model
        gt_algorithm, rl_algorithm = train(bandit, gt_algorithm, rl_algorithm, num_train_steps)
        gt, rl = evaluate(bandit,gt_algorithm,rl_algorithm,num_eval_steps,certainty)
        (gt_weak_regret_measures,rl_weak_regret_measures), (gt_rewards, rl_rewards), _ = compute_statistics(bandit,gt,rl,num_eval_steps=num_eval_steps)
        gt_regret_measures[i] = gt_weak_regret_measures[-1]
        rl_regret_measures[i] = rl_weak_regret_measures[-1]
        gt_rewards_measures[i] = gt_rewards[-1]
        rl_rewards_measures[i] = rl_rewards[-1]
        rl_algorithm.reset()
        gt_algorithm.reset()
        bandit.reset()
    
    gt_regret = {"Mean": gt_regret_measures.mean(), 
                 "Std": gt_regret_measures.std(),
                 "Max": gt_regret_measures.max(),
                 "Min": gt_regret_measures.min()}
    
    gt_reward = {"Mean": gt_rewards_measures.mean(), 
                "Std": gt_rewards_measures.std(),
                "Max": gt_rewards_measures.max(),
                "Min": gt_rewards_measures.min()}
    
    rl_regret = {"Mean": rl_regret_measures.mean(), 
                 "Std": rl_regret_measures.std(),
                 "Max": rl_regret_measures.max(),
                 "Min": rl_regret_measures.min()}
    
    rl_reward = {"Mean": rl_rewards_measures.mean(), 
                "Std": rl_rewards_measures.std(),
                "Max": rl_rewards_measures.max(),
                "Min": rl_rewards_measures.min()}
    return {f'{gt_algorithm_class.__name__}':{"Regret":gt_regret,'Reward':gt_reward},
            f'{rl_algorithm_class.__name__}':{"Regret":rl_regret,'Reward':rl_reward}}

def gather_test_statistics(num_train_steps, num_eval_steps, num_runs, gt_algo, rl_algos, zetas, path):
    certainty_bar = tqdm(reversed(zetas), leave=True, position=0, total=len(zetas))
    for z in certainty_bar:
        certainty_bar.set_description_str(f'Certainty {z}')
        state_rl = np.random.get_state()
        results = {}
        for i, (algorithm, args, kwargs) in enumerate(rl_algos):
            stochastic_bandit = BanditEnvironment(10, stochastic=True)
            stochastic = train_and_evaluate(stochastic_bandit, num_train_steps, num_eval_steps, num_runs, algorithm, gt_algo, certainty = z, *args, **kwargs)
            deterministic_bandit = BanditEnvironment(10, stochastic=False)
            deterministic = train_and_evaluate(deterministic_bandit, num_train_steps, num_eval_steps, num_runs, algorithm, gt_algo, certainty = z, *args, **kwargs)
            np.random.set_state(state_rl)
            results[rl_algos[i][0].__name__] = {'Deterministic': deterministic, 'Stochastic': stochastic}
        with open(os.path.join(path,f'results_{z}.json'), 'w') as f:
            json.dump(results, f, indent=4)


def visualize_stochastic_environment_learning(num_train_steps,num_eval_steps,num_runs,gt_algo,rl_algos,zetas,path,color,active):
    certainty_bar = tqdm(reversed(zetas), leave=True, position=0, total=len(zetas))
    for z in certainty_bar:
        certainty_bar.set_description_str(f'Certainty {z}')

        gt_average_regret = np.zeros((len(rl_algos), num_eval_steps))
        gt_percents_optimal = np.zeros((len(rl_algos), num_eval_steps))
        rl_average_regret = np.zeros((len(rl_algos), num_eval_steps))
        rl_percents_optimal = np.zeros((len(rl_algos), num_eval_steps))


        bandit = BanditEnvironment(10, stochastic=True)
        for i, (algorithm, args, kwargs) in enumerate(rl_algos):
            (gt_average_regret[i], gt_percents_optimal[i]), (rl_average_regret[i], rl_percents_optimal[i]) = gather_visualization_statistics(bandit, num_train_steps, num_eval_steps, num_runs, algorithm, gt_algo, certainty = z, active=active, *args, **kwargs)

        np.save(os.path.join(path,f'gt_average_regret_{z}_.npy'), gt_average_regret)
        np.save(os.path.join(path,f'gt_percent_optimal_{z}.npy'), gt_percents_optimal)
        np.save(os.path.join(path,f'rl_average_regret_{z}.npy'), rl_average_regret)
        np.save(os.path.join(path,f'rl_percent_optimal_{z}.npy'), rl_percents_optimal)
        tqdm.write('saved average regret and percent optimal')
        visualize(rl_algos, z, gt_average_regret, rl_average_regret, gt_percents_optimal, rl_percents_optimal, path, color)

def visualize(rl_algos, certainty, gt_average_regret, rl_average_regret, gt_percents_optimal, rl_percents_optimal, path, color):

    os.makedirs(os.path.join(path,'Regret'),exist_ok=True)
    plt.figure(figsize=(10, 20/3))
    for i,rl_algo in enumerate(rl_algos):
            plt.plot(gt_average_regret[i], f'{color[i]}-', label=f'EXP3IXrl with {rl_algo[0].__name__}')
            plt.plot(rl_average_regret[i], f'{color[i]}--', label=f'{rl_algo[0].__name__}')
    plt.xlabel('Steps')
    plt.ylabel('Average Regret')
    plt.legend()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(os.path.join(path,'Regret',f'EXP3IXrl-Regret_{certainty}.png'), bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    os.makedirs(os.path.join(path,'Freq'),exist_ok=True)
    for i,rl_algo in enumerate(rl_algos):
        if gt_percents_optimal[i].max() <= 1 or rl_percents_optimal[i].max() <= 1:
            data = np.stack([100*gt_percents_optimal[i],100*rl_percents_optimal[i]],axis=0).T
        else:
            data = np.stack([gt_percents_optimal[i],rl_percents_optimal[i]],axis=0).T
        plt.figure(figsize=(10, 20/3))
        plt.hist(data, bins=100*np.arange(0.0,1.01,0.1), weights=100*np.ones(data.shape)/data.shape[0], label = [f'EXP3IXrl with {rl_algos[i][0].__name__}',rl_algos[i][0].__name__])
        plt.xlabel(f'Frequency of Optimal Action Chosen (%)')
        plt.xticks(100*np.arange(0.0,1.01,0.1))
        plt.ylabel(f'Frequency of Occurance Within a Run (%)')
        plt.yticks(100*np.arange(0.0,1.01,0.1))
        plt.legend()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(os.path.join(path,'Freq',f'EXP3IXrl-Freq_{certainty}_{rl_algos[i][0].__name__}.png'), bbox_inches = 'tight', pad_inches = 0)
        plt.close()

if __name__ == '__main__':
    main()