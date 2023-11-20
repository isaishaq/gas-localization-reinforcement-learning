## This code is to test the Q-learning algorithm

import gym
from gym_examples.envs import FieldEnv
from gym.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
from plotcurve import plot_learning_curve
import time

POSITION = {
    1 : (1, 1),
    2 : (25, 1),
    3 : (25, 15),
    4 : (1, 15)
}

FILENAME = 'qtable.png'

LEARNING_RATE = 0.01
DISCOUNT = 0.99
EPISODES = 20

SHOW_EVERY = 1000

env = gym.make('gym_examples/GridWorld-v3', render_mode='human', explore_indicator=True )
env = FlattenObservation(env)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high + 1) # divide to 20 chunks or 20 buckets
discrete_os_win_size = (env.observation_space.high - env.observation_space.low + 1) / DISCRETE_OS_SIZE

epsilon = 0.4
EPSILON_DECAY = 1E-6
EPS_END = 0.01
# START_EPSILON_DECAYING = 1
# END_EPSILON_DECAYING = EPISODES // 2
# epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

table=9000
q_table = np.load(f'final-qtables/env4/{table}-qtable.npy')

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

scores, eps_history, total_timestep = [], [], []
start_time = time.time()
for episode in range(EPISODES):
    score = 0
    timestep = 0
    if episode % SHOW_EVERY == 0:
        print('Episodes:', episode)
        render = True
    else:
        render = False
    state, info = env.reset(agent_starting_position=POSITION[2])
    discrete_state = get_discrete_state(state)
    done = False
    truncated = False
    while not done and not truncated:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, truncated, _ = env.step(action)
        timestep += 1
        score += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done and not truncated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)

            q_table[discrete_state+(action,)] = new_q # update q table

        elif done:
            print(f"We made it on episode {episode}, with timestep{timestep}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
        score += reward
        # if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        #     epsilon -= epsilon_decay_value
        epsilon = epsilon - EPSILON_DECAY if epsilon > EPS_END \
            else EPS_END

    scores.append(score)
    eps_history.append(epsilon)
    ep_rewards.append(score)

    # if not episode % SHOW_EVERY == 0:
    #     average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
    #     aggr_ep_rewards['ep'].append(episode)
    #     aggr_ep_rewards['avg'].append(average_reward)
    #     aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
    #     aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

    #     print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
    
    if episode % SHOW_EVERY == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
    total_timestep.append(timestep)
    if episode % 1 == 0:
        avg_score = np.mean(scores[-1:])
        print('episode ', episode, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % epsilon)

print(total_timestep)  
x = [i+1 for i in range(EPISODES)]
plot_learning_curve(x, scores, eps_history, FILENAME)
print("--- %s seconds ---" % (time.time() - start_time))   
time.sleep(10)



