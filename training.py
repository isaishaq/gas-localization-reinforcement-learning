import gym
from brain import Agent
# from brain2 import Agent
from plotcurve import plot_learning_curve
import numpy as np
from gym_examples.envs import FieldEnv
from gym.wrappers import FlattenObservation
import torch
import time

SAVE_PATH = './model/v4/ENV1/500eps.pth'
FILENAME = './model/v4/ENV1/results_500eps.png'
# SAVE_PATH = './model/8actions/500eps.pth'
# FILENAME = './model/8actions/results_500eps.png'

SHOW_EVERY = 50

if __name__ == '__main__':
    env = gym.make('gym_examples/GridWorld-v4')
    env = FlattenObservation(env)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, eps_end=0.01,
                  input_dims=env.observation_space.shape, lr=0.001, eps_dec=3e-5) # input dims before = [8]
    
    scores, eps_history = [], []
    n_games = 500
    
    start_time = time.time()
    for i in range(n_games):
        score = 0
        done = False
        truncated = False
        observation, info = env.reset()
        while not done and not truncated:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info_ = env.step(action)
            score += reward
            # print('Action taken: ', action, 'Observation: ', observation, 'Reward:', reward)
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            info = info_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % SHOW_EVERY == 0:
            avg_score = np.mean(scores[-SHOW_EVERY:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % agent.epsilon)
    torch.save(agent.Q_eval.state_dict(), SAVE_PATH)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, FILENAME)
    print("--- %s seconds ---" % (time.time() - start_time))