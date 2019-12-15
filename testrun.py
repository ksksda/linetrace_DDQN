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

num_episodes = 1000
max_steps = 10000

input_size = 6
output_size = sim.linetrace_car().action_fields
layer_info = []
layer_info.append([input_size,'linear'])
layer_info.append([18,'sigmoid'])
layer_info.append([output_size,'linear'])

mainQN = qn.QNetwork(layer_info, huber, 0, 0)

result_l=0
result_s=0
result_v=0
result_a=0
result_o=0

names=['weight_supervised.h5','weight_superDQN_center.h5','weight_superDQN_speed.h5','weight_superDQN_acceleration.h5','weight_superDQN_all.h5']
labels=['A','B','C','D','E']
colors=['gray','r','g','b','y']

def get_state(cart, v):
	sense = cart.get_sense()
	for i in range(len(sense)):
		v.append(sense[i]!=0)
	return np.reshape(np.array(v),[1,6])

def get_action(cart, state, mainQN):
	a_list = np.dot(cart.mtrL != cart.order_range[:, np.newaxis], cart.mtrR != cart.order_range[np.newaxis, :]).reshape(output_size) * np.arange(1, output_size + 1)
	action_list = a_list[a_list.nonzero()] - 1
	predicted_rewards = mainQN.model.predict(state)[0][action_list]
	action = action_list[np.argmax(predicted_rewards)]
	return action

#for i in range(5):
for i in range(4,5):
	mainQN.model.load_weights(names[i])
	for episode in range(num_episodes):
		cart = sim.linetrace_car(False)
		state = get_state(cart, cart.step()[0])
		stage_level = 1
		t_v=t_a=t_o=0
		plot_x=[]
		plot_y=[]
		for t in range(max_steps + 1):
			action = get_action(cart, state, mainQN)
			v, _ = cart.step(action)
			next_state = get_state(cart, v)
			sense = np.array(cart.get_sense())
			done = cart.check_done(sense, stage_level) if t < (max_steps - 1) else -1
	
			state = next_state
	
			t_v+=cart.mtrL+cart.mtrR
			t_a+=abs(action//3-1+action%3-1)
			t_o+=bool(state[0][2])^bool(state[0][5])
	
			if done:
				print('%7d (%5d, %5d) %5d  %2d' % (episode+1, cart.posx, cart.posy, t+1, stage_level))
				if done>0:
					result_l+=1
					result_s+=t+1
					result_v+=t_v
					result_a+=t_a
					result_o+=t_o
				break
	
			if min(sense[sense.nonzero()]) == stage_level + 1:
				stage_level += 1
'''
			plot_x.append(cart.posx+np.cos(cart.posa)*150)
			plot_y.append(cart.posy+np.sin(cart.posa)*150)
	
		if done > 0 or episode + 1 == num_episodes:
			plt.plot(plot_x,plot_y,label=labels[i],color=colors[i])
			print('x')
			break

plt.plot(cart.line_x1,cart.line_y1,color='black')
plt.plot(cart.line_x2,cart.line_y2,color='black')
plt.legend(loc='lower left')
plt.axis([-200,1800,-1400,200])
plt.axes().set_aspect('equal')
plt.show()
'''
print(float(result_l)/10.)
print(float(result_s)/float(result_l))
print(float(result_v)/2./float(result_s))
print(float(result_a)/2./float(result_s))
print(float(result_o)/float(result_s))

