import numpy as np
import LineTrace as sim
import qn
from collections import deque
from keras import backend as K
from tensorflow import where
import matplotlib.pyplot as plt

def huber(y_true, y_pred):
	err = y_true - y_pred
	cond = K.abs(err) < 1.0
	L2 = 0.5 * K.square(err)
	L1 = (K.abs(err) - 0.5)
	loss = where(cond, L2, L1)
	return K.mean(loss)

num_episodes = 10000
max_steps = 10000
goal_level = 14
average_episodes = 10
memory_size = 32768
lean_step = 16
batch_size = 16
stock_size = 1000
gamma = 0.5
learning_rate = 0.000001

input_size = 6
output_size = sim.linetrace_car().action_fields
accs = []
layer_info = []
layer_info.append([input_size,'linear'])
layer_info.append([18,'sigmoid'])
layer_info.append([output_size,'linear'])

memory = deque(maxlen = memory_size)
mainQN = qn.QNetwork(layer_info, huber, gamma, learning_rate)
targetQN = qn.QNetwork(layer_info, huber, gamma, learning_rate)
mainQN.model.load_weights('weight_supervised.h5')
targetQN.model.load_weights('weight_supervised.h5')

levels=np.zeros(15)
level_deque = deque(maxlen = average_episodes)
total_step = 0
islearned = False

def get_state(cart, v):
	sense = cart.get_sense()
	for i in range(len(sense)):
		v.append(sense[i]!=0)
	return np.reshape(np.array(v),[1,6])

def get_action(cart, state, episode,mainQN):
	epsilon = 1 - (1 - 0.001) * min((episode,1000))/1000
	a_list = np.dot(cart.mtrL != cart.order_range[:, np.newaxis], cart.mtrR != cart.order_range[np.newaxis, :]).reshape(output_size) * np.arange(1, output_size + 1)
	action_list = a_list[a_list.nonzero()] - 1
	predicted_rewards = mainQN.model.predict(state)[0][action_list]
	if epsilon <= np.random.uniform(0, 1) or islearned:
		action = action_list[np.argmax(predicted_rewards)]
	else:
		action = np.random.choice(action_list)
		r = np.random.uniform(0, sum(predicted_rewards + 1))
		lim = 0
		for i in range(len(action_list)):
			lim += predicted_rewards[i] + 1
			if r < lim:
				action = action_list[i]
				break
	return action

def get_reward(state, done, v, action):
	reward = 0
	if done<0:
		reward = done
	else:
		reward += sum(v)*0.1
		reward += -0.1*(abs(action//3-1+action%3-1))+0.1
		if bool(state[0][2]) ^ bool(state[0][5]):
			reward += -0.1
	return reward

for episode in range(num_episodes):
	cart = sim.linetrace_car(False)
	#cart = sim.linetrace_car(islearned)
	state = get_state(cart, cart.step()[0])
	stage_level = 1

	for t in range(max_steps + 1):
		action = get_action(cart, state, episode, mainQN)
		v, _ = cart.step(action)
		next_state = get_state(cart, v)
		sense = np.array(cart.get_sense())
		done = cart.check_done(sense, stage_level) if t < (max_steps - 1) else -1
		reward = get_reward(next_state,done,v,action)

		memory.append((state, action, reward, next_state, done))
		total_step += 1
		state = next_state
		if (len(memory) == stock_size):
			print('train start')
		if (total_step) % lean_step == 0 and not islearned:
			if (len(memory) > stock_size):
				ret = mainQN.replay(memory, batch_size, targetQN)
				accs.append(ret.history['acc'])
			total_step = 0

		if done:
			level_deque.append(stage_level)
			levels[stage_level-1] += 1
			print('%7d (%5d, %5d) %5d  %2d  %.2f' % (episode+1, cart.posx, cart.posy, t+1, stage_level, sum(level_deque)/average_episodes))
			break

		if min(sense[sense.nonzero()]) == stage_level + 1:
			stage_level += 1

		if islearned:
			cart.plot_state()

	if islearned:
		if done > 0 or episode + 1 == num_episodes:
			plt.plot(range(len(accs)),accs)
			plt.show()
			cart.show_plot()
			print(levels)
			mainQN.model.save_weights('weight_superDQN_all.h5')
			break

	if sum(level_deque)/average_episodes >= goal_level or episode + 1 >= num_episodes - 100:
		islearned = True

