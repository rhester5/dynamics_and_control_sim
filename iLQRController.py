import sys
sys.path.append('../ilqr/ilqr')

from ilqr import iLQR
from iLQRCost import iLQR_cost
import numpy as np

# class iLQR_Controller():
# 	def __init__(self, dynamics, trajectory, actions, linear, num_steps,):
# 		self.dynamics = dynamics
# 		self.trajectory = trajectory
# 		self.actions = actions
# 		self.linear = linear
# 		self.num_steps = num_steps

# 	def __call__(self, x, sp, k):
# 		# if not k%self.num_steps:
# 			# print(k)
# 		# traj = self.trajectory[k:(k//self.num_steps+1)*self.num_steps, :]
# 		# traj = self.trajectory[k:k+self.num_steps, :]
# 		if k+self.num_steps < self.trajectory.shape[0]:
# 			traj = self.trajectory[k:k+self.num_steps, :]
# 			us_init = self.actions[k:k+self.num_steps, :]
# 			horizon = self.num_steps
# 		else:
# 			traj = self.trajectory[k:, :]
# 			us_init = self.actions[k:, :]
# 			horizon = self.actions.shape[0]-k
# 		# print(self.trajectory.shape, self.actions.shape, traj.shape, us_init.shape)
# 		x0 = x # traj[0, :]
# 		goal = traj[-1, 0:3]
# 		cost = iLQR_cost(goal, self.linear)
# 		ilqr = iLQR(self.dynamics, cost, horizon) # traj.shape[0])
# 		# us_init = self.actions[k:(k//self.num_steps+1)*self.num_steps, :]
# 		# us_init = self.actions[k:k+self.num_steps, :]
# 		xs, us = ilqr.fit(x0, us_init)
# 		return us[0]

# ok it's stupid to re-optimize the whole trajectory (at least in python)
# I should do an initial optimization over the whole trajectory
# break it into chunks
# and re-optimize every chunk
# how do I go from one chunk to the next though without it fucking up though?
# yeah so far this doesn't work, re-optimizing the entirety of the remaining trajectory works best
# but how to do it quickly? re-optimizing every 50 steps was the smallest number of steps I could do
# re-optimizing every 10 steps was taking forever
# but the best I can do is equivalent to 2 Hz and it was not good enough, really need it to be like 40 Hz probably
# is this what tedrake was talking about? the solvers haven't caught up yet
# or maybe it would be quicker if the dynamics were hard coded instead of auto differentiated?
# obviously it's not recomputing the dynamics every time, but whatever theano object it plugs into for
# the Jacobian might take longer than just a numpy array (but that seems unlikely since all the
# autodiff and tensor stuff was designed for deep learning where training time is very important)
# ooo wait one thing I'm clearly doing wrong is setting x0 to where it's supposed to be, not where it is... let's try that
# eh that's better but x position is still a mess and the z tracking is nowhere near as good as the "re-optimize everything every 50 steps"
# 20 steps is better but still not satisfactory
# or what if we do a receding horizon? e.g. always re-optimize the next 10 steps
# struggling with the receding horizon because actions and trajectory are different length, could that have been causing problems elsewhere?
# ok receding horizon is trash, at least with 10 time steps, wonder what happens if I reduce the time step length (and increase the number of time steps looked at a time?)
# hey that actually worked pretty well, like really well with a 20 step horizon
# or honestly reducing time step length and re-optimizing the entire trajectory might be the best
# ok yeah that's almost literally perfect
# I'm very happy

# so in conclusion, re-optimizing the entire trajectory as you go rather than re-optimizing a receding horizon is better and can be done sufficiently quickly at 20 Hz instead of 100 Hz
# oh wait shit nvm I'm re-optimizing the entire trajectory every 10 time steps which in this case is... every 1/2 second lol but for some reason it works really well when it didn't before?
# anyway whaterver we're gucci, the x position control isn't perfect but it's not bad
# oof let's try it on an x step
# ok hype it works very well, z is good during the x step response
# now I'm curious what happens if the geometric controller is given the iLQR trajectory
# and I'm curious if I can get iLQR to follow some crazy ass trajectories
# need to figure out how to formulate them because the way I was making the circle did not work at all
# and then once I can make complicated trajectories instead of just steps I can initialize those trajectories with a plannnnnnerrrrrr
# and if I can get the geometric controller working better than the base iLQR then I'll have all of the parts of the pipeline that I'm interested in
# (plan -> traj opt -> control -> state estimation)
# and then I can either do it in C++ or think about research directions or shit like that
# also try turning up the noise on the kalman filter
# need to reimplement the geometric controller
# also wondering if including linear and angular acceleration in the dynamics model/trajectory/state would be helpful
# I mean you can get dv/dt and dw/dt directly from the velocity/angular velocity differences and the time step length
# so 2 main pieces now are:
# - get iLQR to generate trajectories of any shape
# - get geometric controller to follow those trajectories (look at implementation again based on kdc report)
# ^ and subject to increasing amounts of noise

class iLQR_Controller():
	def __init__(self, dynamics, actions, cost, us_init, K):
		self.dynamics = dynamics
		self.actions = actions
		self.cost = cost
		self.us_init = us_init
		self.K = K

	def __call__(self, x, sp, next_sp, k):
		if k > 0 and not k % 10:
			ilqr = iLQR(self.dynamics, self.cost, self.K-k)
			xs, us = ilqr.fit(x, self.us_init[k:, :])
			self.actions = np.zeros(self.actions.shape)
			self.actions[k:, :] = us
			return us[0]
		else:
			return self.actions[k]

	def set_gains(self, gains):
		raise TypeError('iLQR Controller does not have gains')

# class iLQR_Controller():
# 	def __init__(self, actions):
# 		self.actions = actions

# 	def __call__(self, x, sp, k):
# 		return self.actions[k]