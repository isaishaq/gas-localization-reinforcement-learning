## This code is to test the Deep Q Learning algorithm

import gym
from brain import Agent
from plotcurve import plot_learning_curve
import numpy as np
from gym_examples.envs import FieldEnv
from gym.wrappers import FlattenObservation
import torch
import time

POSITION = {
    1 : (1, 1),
    2 : (25, 1),
    3 : (25, 15),
    4 : (1, 15)
}

if __name__ == '__main__':
    env = gym.make('gym_examples/GridWorld-v3', render_mode="human", explore_indicator=True)
    env = FlattenObservation(env)

    # Set epsilon to 0.4 to have higher chance of making random move
    agent = Agent(gamma=0.99, epsilon=0.4, batch_size=64, n_actions=env.action_space.n, eps_end=0.01,
                  input_dims=env.observation_space.shape, lr=0.001) # input dims before = [8]
    
    
    agent.Q_eval.load_state_dict(torch.load('./model/v3/ENV4/500eps.pth'))
    agent.Q_eval.eval()
    
    scores, eps_history, total_timestep = [], [], []
    n_games = 20

    
    for i in range(n_games):
        score = 0
        done = False
        truncated = False
        observation, info = env.reset(agent_starting_position=POSITION[2])
        timestep=0
        while not done and not truncated:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info_ = env.step(action)
            timestep+=1
            score += reward
            # print('Action taken: ', action, 'Observation: ', observation, 'Reward:', reward)
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            observation = observation_
            info = info_
        total_timestep.append(timestep)
        scores.append(score)
        eps_history.append(agent.epsilon)

        # if terminated or truncated:
            # observation, info = env.reset()

        if i % 1 == 0:
            avg_score = np.mean(scores[-1:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % agent.epsilon)
    print(total_timestep)         
    x = [i+1 for i in range(n_games)]
    filename = 'test.png'
    plot_learning_curve(x, scores, eps_history, filename)
    time.sleep(10)