import numpy as np
import random
import LineTrace as sim
from collections import deque

num_episodes = 100
max_steps = 5000
total_step = 0
goal_level = 12
average_episodes = 100
level_deque = deque(maxlen = average_episodes)
islearned = False

def get_state(cart, v):
	return cart.get_sense()

def get_action(cart, state):
	o_l = 0
	o_r = 0
	if state[3] != 0:
		o_l = 1
	elif state[2] == 0:
		o_l = -1
	if state[0] != 0:
		o_r = 1
	elif state[1] == 0:
		o_r = -1
	if (cart.mtrL+o_l)<=0 and (cart.mtrR+o_r)<=0:
		o_l=1
		o_r=1
	if o_l == 0 and o_r == 0:
		o_l = -1 if cart.mtrL > 5 else (1 if cart.mtrL < 5 else 0)
		o_r = -1 if cart.mtrR > 5 else (1 if cart.mtrR < 5 else 0)
	return (o_l + 1) * 3 + (o_r + 1)

for episode in range(num_episodes):
	cart = sim.linetrace_car(False)
	state = get_state(cart, cart.step()[0])
	stage_level = 1

	for t in range(max_steps + 1):
		action = get_action(cart, state)
		v, action = cart.step(action)
		next_state = get_state(cart, v)
		sense = np.array(cart.get_sense())
		done = cart.check_done(sense, stage_level, t) if t < max_steps-1 else -1
		reward = done

		if done:
			level_deque.append(stage_level)
			print('%5d (%5d, %5d) %4d %2d %.2f' % (episode+1, cart.posx, cart.posy, t+1, stage_level, sum(level_deque)/len(level_deque)))
			break

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

