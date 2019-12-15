import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def get_action(state):
	o_l = 0
	l = 0
	o_r = 0
	r = 0
	if state[0][5] != 0:
		o_l = (state[0][0] < 9)
		l=1
	elif state[0][4] == 0:
		o_l = -1*(state[0][0] > 0)
		l=1
	if state[0][2] != 0:
		o_r = (state[0][1] < 9)
		r=1
	elif state[0][3] == 0:
		o_r = -1*(state[0][1] > 0)
		r=1
	if (state[0][0]+o_l)<=0 and (state[0][1]+o_r)<=0:
		o_l=1
		o_r=1
		l=r=1
	if l == 0 and r == 0:
		o_l = -1 if state[0][0] > 5.5 else (1 if state[0][0] < 4.5 else 0)
		o_r = -1 if state[0][1] > 5.5 else (1 if state[0][1] < 4.5 else 0)
	return (o_l + 1) * 3 + (o_r + 1)

class QNetwork:
	def __init__(self, layer_info, loss_func, gamma, learning_rate, supervise=False):
		self.model = Sequential()
		self.model.add(Dense(layer_info[1][0], activation = layer_info[1][1], input_dim = layer_info[0][0]))
		for i in range(len(layer_info) - 2):
			self.model.add(Dense(layer_info[i + 2][0], activation = layer_info[i + 1][1]))
		opt = Adam(lr = learning_rate)
		self.model.compile(optimizer = opt, loss = loss_func, metrics = ['accuracy'])
		self.gamma = gamma
		self.input_size = layer_info[0][0]
		self.output_size = layer_info[len(layer_info) - 1][0]
		#self.num_learned = 0
		self.supervise = supervise
	def replay(self, memory, batch_size, targetQN):
		inputs = np.zeros((batch_size, self.input_size))
		targets = np.zeros((batch_size, self.output_size))
		mini_batch = random.sample(memory, k = batch_size)
		for i, (state_b, action_b, reward_b, next_state_b, done_b) in enumerate(mini_batch):
			inputs[i] = state_b
			if self.supervise:
				targets[i] = [j == get_action(state_b) for j in range(self.output_size)]
			else:
				targets[i] = self.model.predict(state_b)[0]
				targets[i][action_b] = reward_b + self.gamma * np.max(targetQN.model.predict(next_state_b)[0]) * (not done_b)
		'''
		if self.num_learned > 625:
			targetQN.model.set_weights(self.model.get_weights())
			self.num_learned = 0
		else:
			self.num_learned += 1
		'''
		targetQN.model.set_weights(self.model.get_weights())
		return self.model.fit(inputs, targets, epochs=1, verbose=0)

