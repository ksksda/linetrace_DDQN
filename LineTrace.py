import numpy as np
import matplotlib.pyplot as plt

class linetrace_car:
	sensors = np.array([[150, 22.5], [150, 7.5], [150,-7.5], [150,-22.5]])
	cart_d = 200

	o_min = 0
	o_max = 10
	order_range = np.array([o_min, np.nan, o_max])

	x_min = -10
	x_max =  10
	y_min = -10
	y_max =  10
	a_min = -0.17
	a_max =  0.17
	v_min = 0.9 * o_min
	v_max = 1.1 * o_max
	w_min = -(1.1 * o_max - 0.9 * o_min) / (2 * cart_d)
	w_max =  (1.1 * o_max - 0.9 * o_min) / (2 * cart_d)

	action_fields = 9

	'''
	line_x1=[0]
	line_y1=[-10]
	line_x2=[0]
	line_y2=[10]
	for theta in np.linspace(0,np.pi/2,45):
		line_x1.append( 140 * np.sin(theta) + 1150)
		line_y1.append( 140 * np.cos(theta) - 150)
		line_x2.append( 160 * np.sin(theta) + 1150)
		line_y2.append( 160 * np.cos(theta) - 150)
	for theta in np.linspace(0,np.pi/2,45):
		line_x1.append( 140 * np.cos(theta) + 1150)
		line_y1.append(-140 * np.sin(theta) - 890)
		line_x2.append( 160 * np.cos(theta) + 1150)
		line_y2.append(-160 * np.sin(theta) - 890)
	line_x1.append(900)
	line_y1.append(-1030)
	line_x2.append(900)
	line_y2.append(-1050)
	'''

	line_x1=[0]
	line_y1=[-10]
	line_x2=[0]
	line_y2=[10]
	for theta in np.linspace(np.pi/2,0,45):
		line_x1.append( 140 * np.cos(theta) + 300)
		line_y1.append( 140 * np.sin(theta) - 150)
		line_x2.append( 160 * np.cos(theta) + 300)
		line_y2.append( 160 * np.sin(theta) - 150)
	for theta in np.linspace(np.pi,3*np.pi/2,45):
		line_x1.append( 160 * np.cos(theta) + 600)
		line_y1.append( 160 * np.sin(theta) - 450)
		line_x2.append( 140 * np.cos(theta) + 600)
		line_y2.append( 140 * np.sin(theta) - 450)
	for theta in np.linspace(0,np.pi*2,180):
		line_x1.append(1050 + 180 * theta / np.pi)
		line_y1.append(  10 * np.cos(theta) - 620)
		line_x2.append(1050 + 180 * theta / np.pi)
		line_y2.append(  10 * np.cos(theta) - 600)
	for theta in np.linspace(np.pi/2,-np.pi/2,90):
		line_x1.append( 290 * np.cos(theta) + 1410)
		line_y1.append( 290 * np.sin(theta) - 900)
		line_x2.append( 310 * np.cos(theta) + 1410)
		line_y2.append( 310 * np.sin(theta) - 900)
	for theta in np.linspace(np.pi*2,0,180):
		line_x1.append(1050 + 180 * theta / np.pi)
		line_y1.append(  10 * np.cos(theta) - 1200)
		line_x2.append(1050 + 180 * theta / np.pi)
		line_y2.append(  10 * np.cos(theta) - 1220)
	for theta in np.linspace(3*np.pi/2,np.pi,45):
		line_x1.append( 290 * np.cos(theta) + 1050)
		line_y1.append( 290 * np.sin(theta) - 900)
		line_x2.append( 310 * np.cos(theta) + 1050)
		line_y2.append( 310 * np.sin(theta) - 900)
	for theta in np.linspace(0,np.pi/2,45):
		line_x1.append( 160 * np.cos(theta) + 600)
		line_y1.append( 160 * np.sin(theta) - 450)
		line_x2.append( 140 * np.cos(theta) + 600)
		line_y2.append( 140 * np.sin(theta) - 450)
	for theta in np.linspace(3*np.pi/2,np.pi,45):
		line_x1.append( 140 * np.cos(theta) + 300)
		line_y1.append( 140 * np.sin(theta) - 150)
		line_x2.append( 160 * np.cos(theta) + 300)
		line_y2.append( 160 * np.sin(theta) - 150)
	line_x1.append(160)
	line_y1.append(150)
	line_x2.append(140)
	line_y2.append(150)

	def __init__(self,Train_finished=True):
		self.lines = []
		if Train_finished:
			self.lines.append(lambda x: -10<=x[0]<=1150 and   -10<=x[1]<=   10)
			self.lines.append(lambda x:1150<=x[0]		and  -150<=x[1]			and 140<=np.linalg.norm(np.array(x)-np.array([1150,-150]))<=160)
			self.lines.append(lambda x:1290<=x[0]<=1310 and  -300<=x[1]<= -150)
			self.lines.append(lambda x:1290<=x[0]<=1310 and  -890<=x[1]<= -300)
			self.lines.append(lambda x:1150<=x[0]		and        x[1]<= -890	and 140<=np.linalg.norm(np.array(x)-np.array([1150,-890]))<=160)
			self.lines.append(lambda x: 900<=x[0]<=1150 and -1050<=x[1]<=-1030)
			self.lines.append(lambda x: 800<=x[0]<= 900 and -1050<=x[1]<=-1030)
		else:
			self.lines.append(lambda x: -10<=x[0]<=300 and -10<=x[1]<= 10)
			self.lines.append(lambda x: 300<=x[0] and -150<=x[1] and 140<=np.linalg.norm(np.array(x)-np.array([300,-150]))<=160)
			self.lines.append(lambda x: 440<=x[0]<=460 and -450<=x[1]<=-150)
			self.lines.append(lambda x: x[0]<=600 and x[1]<=-450 and 140<=np.linalg.norm(np.array(x)-np.array([600,-450]))<=160)
			self.lines.append(lambda x: 600<=x[0]<=1050 and -610<=x[1]<=-590)
			self.lines.append(lambda x:1050<=x[0]<=1410 and -620<=x[1]-10*np.cos((x[0]-1050)*np.pi/180)<=-600)
			self.lines.append(lambda x:1410<=x[0] and 290<=np.linalg.norm(np.array(x)-np.array([1410,-900]))<=310)
			self.lines.append(lambda x:1050<=x[0]<=1410 and -1220<=x[1]-10*np.cos((x[0]-1050)*np.pi/180)<=-1200)
			self.lines.append(lambda x: x[0]<=1050 and x[1]<=-900 and 290<=np.linalg.norm(np.array(x)-np.array([1050,-900]))<=310)
			self.lines.append(lambda x: 740<=x[0]<=760 and -900<=x[1]<=-450)
			self.lines.append(lambda x: 600<=x[0] and -450<=x[1] and 140<=np.linalg.norm(np.array(x)-np.array([600,-450]))<=160)
			self.lines.append(lambda x: 300<=x[0]<=600 and -310<=x[1]<=-290)
			self.lines.append(lambda x: x[0]<=300 and x[1]<=-150 and 140<=np.linalg.norm(np.array(x)-np.array([300,-150]))<=160)
			self.lines.append(lambda x: 140<=x[0]<=160 and -150<=x[1]<=150)
			self.lines.append(lambda x: 140<=x[0]<=160 and 150<=x[1]<=250)
		while True:
			self.posx =-150	+ np.random.uniform(self.x_min,self.x_max)
			self.posy = 0	+ np.random.uniform(self.y_min,self.y_max)
			self.posa = 0	+ np.random.uniform(self.a_min,self.a_max)
			if any(self.get_sense()):
				break
		self.mtrL = self.o_min
		self.mtrR = self.o_min
		self.sensor_plot = []
		for i in range(len(self.sensors)):
			self.sensor_plot.append([[],[]])
		self.cart_plot = [[],[]]

	def step(self, action=4):
		mL = np.clip(self.mtrL + action // 3 - 1, self.o_min, self.o_max)
		mR = np.clip(self.mtrR + action % 3 - 1, self.o_min, self.o_max)
		act_ml = np.random.uniform(1.1, 0.9) * mL
		act_mr = np.random.uniform(1.1, 0.9) * mR
		true_action = (mL - self.mtrL + 1) * 3 + (mR - self.mtrR + 1)
		if mL != 0 or mR != 0:
			mid_angle = self.posa + (act_mr - act_ml) / (4 * self.cart_d)
			self.posx += np.cos(mid_angle) * (act_mr + act_ml) / 2
			self.posy += np.sin(mid_angle) * (act_mr + act_ml) / 2
			self.posa += (act_mr - act_ml) / (2 * self.cart_d)
		self.mtrL = mL
		self.mtrR = mR
		return ([(act_ml - self.v_min)/(self.v_max - self.v_min), (act_mr - self.v_min)/(self.v_max - self.v_min)],true_action)

	def get_sense(self):
		sense = []
		angs = np.array([[np.cos(self.posa),-np.sin(self.posa)],[np.sin(self.posa),np.cos(self.posa)]])
		for i in range(len(self.sensors)):
			s = 0
			p = np.array([sum(self.sensors[i] * angs[0]) + self.posx, sum(self.sensors[i] * angs[1]) + self.posy])
			for j in range(len(self.lines)):
				if self.lines[j](p):
					s = j + 1
					break
			sense.append(s)
		return sense
	
	def check_done(self,sense,stage_level=1):
		if stage_level == len(self.lines):
			done = 1
		elif not any(sense):
			done = -1
		else:
			done = 0
		return done

	def plot_state(self):
		angs = np.array([[np.cos(self.posa),-np.sin(self.posa)],[np.sin(self.posa),np.cos(self.posa)]])
		for i in range(len(self.sensors)):
			p = np.array([sum(self.sensors[i] * angs[0]) + self.posx, sum(self.sensors[i] * angs[1]) + self.posy])
			self.sensor_plot[i][0].append(p[0])
			self.sensor_plot[i][1].append(p[1])
		self.cart_plot[0].append(self.posx)
		self.cart_plot[1].append(self.posy)

	def show_plot(self):
		for i in range(len(self.sensors)):
			plt.plot(self.sensor_plot[i][0],self.sensor_plot[i][1],label='sensor'+str(i+1))
		plt.plot(self.cart_plot[0],self.cart_plot[1],label='cart')
		plt.plot(self.line_x1,self.line_y1,color='black')
		plt.plot(self.line_x2,self.line_y2,color='black')
		plt.legend(loc='lower left')
		plt.axis([-200,1800,-1400,200])
		plt.axes().set_aspect('equal')
		plt.show()

