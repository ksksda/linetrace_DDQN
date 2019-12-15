import numpy as np
from matplotlib import pyplot as plt

class cyclone:
	def __init__(self):
		self.D1 = 160
		self.L1 = 130
		self.D2 = 200
		self.L2 = 130
		self.motor_spec = [12.5*np.pi, 3890]
		self.a = 0
		self.t = 0.01
		self.F = 0
		self.T = 0
		
		self.a1_max = 2*np.arcsin(self.L1/self.D1)
		self.a2_max = 2*np.arcsin(self.L2/self.D2)
		
		self.a1 = self.a1_max - 2*np.arcsin(np.sqrt((self.L1**2-5**2)/self.D1**2))
		self.l1 = self.D1 * np.sin((self.a1_max-(self.a1-self.a))/2)
		self.h = np.sqrt(self.L1**2-self.l1**2)
		self.l2 = np.sqrt(self.L2**2-self.h**2)
		self.a2 = 2 * (np.arcsin(self.l2/self.D2) + self.a) - self.a2_max
	
	def step(self,v):
		w1, w2 = v * self.motor_spec[0]
		T1, T2 = v * self.motor_spec[1]
		self.a += self.t/2 * (-(self.a2-self.a)*w1 + (self.a1-self.a)*w2)/(self.a1-self.a2)
		self.a1 += self.t/2 * w1
		self.a2 += self.t/2 * w2
		self.F = self.h * (
			np.sin(self.a1_max-(self.a1-self.a))/self.L1**2*T1
			+np.sin(self.a2_max+(self.a2-self.a))/self.L2**2*T2
		)
		self.T = (
			(self.D1*np.sin(self.a1_max-(self.a1-self.a))/(2*self.L1))**2*T1
			+(self.D2*np.sin(self.a2_max+(self.a2-self.a))/(2*self.L2))**2*T2
		)
		self.h += self.t * (
			self.l1*np.sqrt(self.D1**2-self.l1**2)*(self.a1-self.a)*w1
			+self.l2*np.sqrt(self.D2**2-self.l2**2)*(self.a2-self.a)*w2
		)/(2*self.h*(self.a1-self.a2))
		self.l1 = self.D1 * np.sin((self.a1_max-(self.a1-self.a))/2)
		self.l2 = self.D2 * np.sin((self.a2_max+(self.a2-self.a))/2)
		self.a += self.t/2 * (-(self.a2-self.a)*w1 + (self.a1-self.a)*w2)/(self.a1-self.a2)
		self.a1 += self.t/2 * w1
		self.a2 += self.t/2 * w2
	
	def get_eqns(self):
		return [
			[
				self.l1*np.sqrt(self.D1**2-self.l1**2)*(self.a1-self.a)/(2*self.h*(self.a1-self.a2)) * self.motor_spec[0],
				self.l2*np.sqrt(self.D2**2-self.l2**2)*(self.a2-self.a)/(2*self.h*(self.a1-self.a2)) * self.motor_spec[0]
			],
			[
				-((self.a2-self.a)/(self.a1-self.a2)) * self.motor_spec[0],
				 ((self.a1-self.a)/(self.a1-self.a2)) * self.motor_spec[0]
			],
			[
				self.h*np.sin(self.a1_max-(self.a1-self.a))/self.L1**2 * self.motor_spec[1],
				self.h*np.sin(self.a2_max+(self.a2-self.a))/self.L2**2 * self.motor_spec[1]
			],
			[
				(self.D1*np.sin(self.a1_max-(self.a1-self.a))/(2*self.L1))**2 * self.motor_spec[1],
				(self.D2*np.sin(self.a2_max+(self.a2-self.a))/(2*self.L2))**2 * self.motor_spec[1]
			]
		]

if __name__ == '__main__':
	c=cyclone()
	p_out=[]
	p_state=[]
	x=np.arange(0,1.25,0.01)
	labels=['h','a','F','T']
	for i in x:
		eqns=c.get_eqns()
		a=np.array([eqns[0],eqns[2]])
		v=np.dot(np.linalg.inv(a),np.array([[100],[9.8*3]])).reshape(2)
		c.step(v)
		p_out.append(v)
		p_state.append([c.h,c.a,c.F,c.T])
	pl=np.array(p_out).T
	plt.plot(x,pl[0])
	plt.plot(x,pl[1])
	plt.show()
	'''
	pl=np.array(p_state).T
	for i in range(len(pl)):
		plt.plot(x,pl[i],label=labels[i])
		plt.plot(x,np.zeros(len(x)))
		plt.legend()
		plt.show()
	'''

