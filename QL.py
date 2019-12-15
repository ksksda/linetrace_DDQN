import numpy as np
import random
import copy
import LineTrace as sim
from collections import deque

num_episodes = 10000
max_steps = 10000
goal_level = 12
average_episodes = 10
plot_step = 300
gamma = 0.99
learning_rate = 0.1

input_size = 16*(sim.linetrace_car().o_max - sim.linetrace_car().o_min + 1)**2
output_size = sim.linetrace_car().action_fields

q_table = np.random.uniform(-1, 1, (input_size, output_size))
next_q_table = np.random.uniform(-1, 1, (input_size, output_size))

level_deque = deque(maxlen = average_episodes)
total_step = 0
islearned = False

ofs = np.array([0, 2, 4, 8, 16, 16 * (sim.linetrace_car().o_max - sim.linetrace_car().o_min + 1)])

def get_state(cart, v):
	st = []
	sense = cart.get_sense()
	for i in range(len(sense)):
		st.append(1 if sense[i]!=0 else 0)
	st.extend([cart.mtrL,cart.mtrR])
	return sum(np.array(st)*ofs)

def get_action(cart, state):
	epsilon = 0.005 + 1 / (episode/output_size + 1)
	predicted_rewards = q_table[state]
	if epsilon <= np.random.uniform(0, 1) or islearned:
		action = np.argmax(predicted_rewards)
	else:
		action = np.random.choice(range(output_size))
		r = np.random.uniform(0, sum(np.array(predicted_rewards)+2))
		lim = 0
		for i in range(output_size):
			lim += predicted_rewards[i] + 2
			if r < lim:
				action = i
				break
	return action

def get_reward(done,v):
	return done+0.1 if sum(v)!=0 else -1

def update_Qtable(state, action, reward, next_state):
	next_q_table[state][action] = (1 - learning_rate) * next_q_table[state][action] + learning_rate * (reward + gamma * max(next_q_table[next_state]))

for episode in range(num_episodes):
	cart = sim.linetrace_car(False)
	state = get_state(cart, cart.step()[0])
	stage_level = 1
	qtable = copy.deepcopy(next_q_table)

	for t in range(max_steps + 1):
		action = get_action(cart, state)
		v, action = cart.step(action)
		next_state = get_state(cart, v)
		sense = np.array(cart.get_sense())
		done = cart.check_done(sense, stage_level, t) if t < max_steps-1 else -1
		reward = get_reward(done,v)

		if done:
			level_deque.append(stage_level)
			print('%5d (%5d, %5d) %4d %2d %.2f' % (episode + 1, cart.posx, cart.posy, t + 1, stage_level, sum(level_deque)/len(level_deque)))
			break

		update_Qtable(state, action, reward, next_state)

		stage_level = min(sense[sense.nonzero()])
		total_step += 1
		state = next_state

		if islearned:
			cart.plot_state()

	if islearned:
		if done > 0 or episode+1 == num_episodes:
			cart.show_plot()
			break

	if sum(level_deque)/len(level_deque) >= goal_level or episode + 1 == num_episodes-1:
		islearned = True

