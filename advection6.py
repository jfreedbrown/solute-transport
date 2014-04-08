#####################################
#	Table of Contents:				#
#									#
#	- Initialize Class 				#
#	- Initialize List of Points 	#
#	- Run Simulation 				#
#	- Compute Streamline 			#
#	- Voronoi Area Around a Point 	#
#	- Plotting Functions 			#
#	- Helpful Functions 			#
# 									#
#####################################

from numpy import *
import matplotlib.pyplot as plt
import itertools
import time

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from scipy.interpolate import griddata

class advect(object):
	"""The advect class is used to initialize, run, and plot results for a simulation.

In these simulations, test masses with initial mass eta * h * (voronoi area) are passively advected
with the fluid as the drop evaporates. As they advect, their mass is unchanged but their changing 
spacing changes eta and the changing drop-height (from evaporation and spatial changes) changes phi.

By interpolating between these points, we can find eta and phi throughout the drop.
"""

	####################
	# Initialize Class #
	####################

	def __init__(self, a, b, dx0, N, flowfield, gamma = .99, voronoi = False):

		start = time.time()		# Timer to see how long it takes to initialize simulation

		self.voronoi = voronoi 	# voronoi tells us whether to use the voronoi area to track M/A, or an area element

		self.flowfield = flowfield		# This is the simulation of the flow field within the drop
		
		self.tau = 1. 		# tau = 1-t/tf
		self.gamma = gamma 	# (1-gamma)*tau is the time step. tau -> gamma*tau

		self.a = a 		# major axis of ellipse
		self.b = b 		# minor axis of ellipse

		self.dx0 = dx0 	# backgroundspacing

		#self.dx = 2.*a/J 	# x-spacing for pointgenrandom 
		#self.dy = 2.*b/L 	# y-spacing

		#self.J = J 			# x-resolution
		#self.L = L 			# y-res

		self.N = N 			# number of points in weightedpoints

		[x_back, y_back] = self.backgroundpoints() 	# call to initialize lattice of test masses
		[x_weight, y_weight] = self.weightedpoints()
		[x_ring, y_ring] = self.ring()

		self.x = append(append(x_back, x_weight), x_ring)		# set x vector to be background points + weighted points
		self.y = append(append(y_back, y_weight), y_ring)		# set y vector to be background points + weighted points

		self.ring() 		# Ring the drop in points

		N = len(self.x) # reset length of array

		self.area = ones(N) 	# initialize area vector

		if self.voronoi: 		# if we're using voronoi area, compute it
			self.computearea()

		self.mass = self.area * self.h(self.x,self.y) 	# compute mass 

		self.phi = self.tau*(self.mass/self.area)/(1 - (self.x/self.a)**2 - (self.y/self.b)**2) 	# compute phi

		self.computeeta() 	# From mass and area, compute eta
		self.computephi() 	# From eta and h, compute phi

		print "Initialization Time (s): " + str(time.time()-start) #print initialization time
	
	#############################
	# Initialize List of Points #
	#############################

	def backgroundpoints(self):
		"""This is a protocol to uniformly test masses in a hexagonal grid.

The spacing between the points is specified by x0. The mass of a point is given by eta * h * (voronoi area).
"""
		J = int((2*self.a+2*self.dx0)/(self.dx0))		# Number of points in x
		L = int((2*self.b+2*self.dx0*sqrt(3.)/2)/(self.dx0*sqrt(3.)/2)) 	# Number of points in y

		N = (J+1)*(L+1) 	# Total number of points

		newx = zeros(N)		# initialize x, y arrays
		newy = zeros(N)

		for i in range(N): 	# generate the grid points 
			newx[i] = (i%(J+1))*self.dx0 - self.a - self.dx0 + (-1)**(i/(J+1))*self.dx0/4
			newy[i] = (i/(J+1)-L/2)*sqrt(3.)*self.dx0/2 # - self.b - self.dx0*sqrt(3.)/2


		test = (self.h(newx,newy)>0.) # remove nugatory points

		newx = newx[test]
		newy = newy[test]

		return [newx, newy] 	# Return x and y positions


	def weightedpoints(self):
		"""weightedpoints() generates a list of non-uniformly random sample of N initial points (x, y), with more
points located at where the height is larger. It does this by distributing around the drop using h(x,y)/vol_total as a PDF.

If this method works well, the mass of our points should be around dm = pi*a*b/(2*N).

"""
		counter = 0 	# Counter tells us which point is being generated

		newx = zeros(self.N) 		# Initialize x and y arrays
		newy = zeros(self.N)

		while counter<self.N: 		# If we haven't generated enough points (int() always rounds down) add more points until we reach N.
			newx[counter] = tempx = random.uniform(-self.a, self.a) # generate the new point	
			newy[counter] = tempy = random.uniform(-self.b, self.b)

			test = random.random() 		# accept it with probability h
			if test < self.h(tempx,tempy):
				counter += 1

		test = (self.h(newx,newy)>0) 	# remove nugatory points

		newx = newx[test]
		newy = newy[test]

		return [newx, newy]


	def ring(self):
		n = int(2*pi/(self.dx0/(self.a))) 	# divide theta ring into equal spaced points (d theta = 2*pi*a/dx0)

		theta = linspace(0, 2*pi, n) 		# initialize thetas
		newx = 1.01*self.a*cos(theta)		# set x and y points just outside ellipse
		newy = 1.01*self.b*sin(theta)

		return [newx, newy] 				# return new points


	##################
	# Run Simulation #
	##################

	def step(self):
		"""Step implements the fourth order Runge-Kutta method from Numerical Recipes (pg 908). Note that it does not update the position or tau, 
it simply returns the new position

"""
		ellipse = ((self.x/self.a)**2+(self.y/self.b)**2<=1.) # boolean of whether a point is inside the drop

		dt = self.tau-self.gamma*self.tau

		# CHANGE IN POSITION
		k1 = dt * self.flowfield.U(self.x, self.y)/(self.tau)

		[x1, y1] = [self.x+k1[0]/2., self.y+k1[1]/2.]
		k2 = dt * self.flowfield.U(x1, y1)/(self.tau-dt/2.)

		[x2, y2] = [self.x+k2[0]/2., self.y+k2[1]/2.]
		k3 = dt * self.flowfield.U(x2, y2)/(self.tau-dt/2.)

		[x3, y3] = [self.x+k3[0], self.y+k3[1]]
		k4 = dt * self.flowfield.U(x3, y3)/(self.tau-dt)

#		newx = self.x + (self.gamma**(-ux * ellipse) - 1.)*u/ux 	# new x position determind by above formula  	
#		newy = self.y + (self.gamma**(-vy * ellipse) - 1.)*v/vy 	# new y

		newx = self.x + k1[0]/6. + k2[0]/3. + k3[0]/3. + k4[0]/6.
		newy = self.y + k1[1]/6. + k2[1]/3. + k3[1]/3. + k4[1]/6.

		# CHANGE IN AREA
		g1 = dt * self.flowfield.divU(self.x, self.y)*self.area/self.tau

		area1 = self.area + g1/2.
		g2 = dt * self.flowfield.divU(x1, y1)*area1/(self.tau-dt/2.)

		area2 = self.area + g2/2.
		g3 = dt * self.flowfield.divU(x2, y2)*area2/(self.tau-dt/2.)

		area3 = self.area + g3
		g4 = dt * self.flowfield.divU(x3, y3)*area3/(self.tau-dt)

		newarea = self.area + g1/6. + g2/3. + g3/3. + g4/6.

		return array([newx, newy, newarea]) 		# return new values of x and y

	def step_voronoi(self):
		"""Step implements the fourth order Runge-Kutta method from Numerical Recipes (pg 908). Note that it does not update the position or tau, 
it simply returns the new position

"""
		ellipse = ((self.x/self.a)**2+(self.y/self.b)**2<=1.) # boolean of whether a point is inside the drop

		dt = self.tau-self.gamma*self.tau

		# CHANGE IN POSITION
		k1 = dt * self.flowfield.U(self.x, self.y)/(self.tau)

		[x1, y1] = [self.x+k1[0]/2., self.y+k1[1]/2.]
		k2 = dt * self.flowfield.U(x1, y1)/(self.tau-dt/2.)

		[x2, y2] = [self.x+k2[0]/2., self.y+k2[1]/2.]
		k3 = dt * self.flowfield.U(x2, y2)/(self.tau-dt/2.)

		[x3, y3] = [self.x+k3[0], self.y+k3[1]]
		k4 = dt * self.flowfield.U(x3, y3)/(self.tau-dt)

#		newx = self.x + (self.gamma**(-ux * ellipse) - 1.)*u/ux 	# new x position determind by above formula  	
#		newy = self.y + (self.gamma**(-vy * ellipse) - 1.)*v/vy 	# new y

		newx = self.x + k1[0]/6. + k2[0]/3. + k3[0]/3. + k4[0]/6.
		newy = self.y + k1[1]/6. + k2[1]/3. + k3[1]/3. + k4[1]/6.

		return array([newx, newy]) 		# return new values of x and y

	def run(self, tauf = .001):
		"""run(tauf = .001) runs the simulation until tau<tauf"""
		tempgamma = self.gamma

		if not self.voronoi:
			while self.tau>tauf:
				if self.gamma*self.tau < tauf:
					self.gamma = tauf/self.tau
				[self.x, self.y, self.area] = self.step()		# update positions
				self.tau = self.gamma*self.tau 		# update tau

		else:
			while self.tau>tauf:
				if self.gamma*self.tau < tauf:
					self.gamma = tauf/self.tau
				[self.x, self.y] = self.step_voronoi()		# update positions
				self.tau = self.gamma*self.tau 		# update tau
				self.computearea()
		self.gamma = tempgamma

		#self.computearea()
		self.computeeta()
		self.computephi()


	######################
	# Compute Streamline #
	######################

	def streamline(self, X, Y):
		"""Compute a streamline starting from point (X,Y). The streamline goes forward and backwards in time, 
so it runs from the center to the edge. Returns an array of points [X,Y,T]."""

		counter = 0 	# tells us which time step we are on
		gamma = 1/self.gamma 	# set gamma for time steps

		trajx = [X] 	# initialize X, Y, and T arrays
		trajy = [Y]
		trajt = [1.]
		l = 1 			# the length of the array

		while trajx[counter]**2 + trajy[counter]**2 > 10**-2: 	# While the point is sufficiently far from the center
			if counter+1 == l:	# If the next point is outside the array, double the length of the array.
				trajx += l*[0]
				trajy += l*[0]
				trajt += l*[0]

				l = 2*l

			xx = trajx[counter] # set current position (xx, yy)
			yy = trajy[counter]
			tau = trajt[counter]

			dt = tau-gamma*tau

			# CHANGE IN POSITION
			k1 = dt * self.flowfield.U(xx, yy)/(tau)

			[x1, y1] = [xx+k1[0]/2., yy+k1[1]/2.]
			k2 = dt * self.flowfield.U(x1, y1)/(tau-dt/2.)

			[x2, y2] = [xx+k2[0]/2., yy+k2[1]/2.]
			k3 = dt * self.flowfield.U(x2, y2)/(tau-dt/2.)

			[x3, y3] = [xx+k3[0], yy+k3[1]]
			k4 = dt * self.flowfield.U(x3, y3)/(tau-dt)

			trajx[counter+1] = xx + k1[0]/6. + k2[0]/3. + k3[0]/3. + k4[0]/6. 	# Take step BACKWARDS in time
			trajy[counter+1] = yy + k1[1]/6. + k2[1]/3. + k3[1]/3. + k4[1]/6.
			trajt[counter+1] = trajt[counter]*gamma

			counter+=1 # Increment counter

		trajx = trajx[0:counter+1] # concatenate array to exclude unused points
		trajy = trajy[0:counter+1]
		trajt = trajt[0:counter+1]

		trajx.reverse() 	# reverse array (put it in chronological order)
		trajy.reverse()
		trajt.reverse()

		l = len(trajx) 		# reset length
		gamma = self.gamma

		while (trajx[counter]/self.a)**2 + (trajy[counter]/self.b)**2 < 1.: # While the point is inside the ellipse
			if counter+1 == l: 	# Double length of array if necessary
				trajx += l*[0]
				trajy += l*[0]
				trajt += l*[0]

				l = 2*l

			xx = trajx[counter] 	# set current position
			yy = trajy[counter]
			tau = trajt[counter]

			dt = tau-gamma*tau

			# CHANGE IN POSITION
			k1 = dt * self.flowfield.U(xx, yy)/(tau)

			[x1, y1] = [xx+k1[0]/2., yy+k1[1]/2.]
			k2 = dt * self.flowfield.U(x1, y1)/(tau-dt/2.)

			[x2, y2] = [xx+k2[0]/2., yy+k2[1]/2.]
			k3 = dt * self.flowfield.U(x2, y2)/(tau-dt/2.)

			[x3, y3] = [xx+k3[0], yy+k3[1]]
			k4 = dt * self.flowfield.U(x3, y3)/(tau-dt)

			trajx[counter+1] = xx + k1[0]/6. + k2[0]/3. + k3[0]/3. + k4[0]/6. 	# Take step BACKWARDS in time
			trajy[counter+1] = yy + k1[1]/6. + k2[1]/3. + k3[1]/3. + k4[1]/6.
			trajt[counter+1] = trajt[counter]*gamma

			counter+=1 # Increment counter


		trajx = trajx[0:counter+1] 	# Truncate array
		trajy = trajy[0:counter+1]
		trajt = trajt[0:counter+1]


		return [trajx, trajy, trajt] 	# Return [X, Y, T]

	def ri(self, theta, taufinal = .1, eps = 10**-14):
		xi = [(self.a-eps) * cos(theta)]
		yi = [(self.b-eps) * sin(theta)]

		t = [0.]

		gamma = self.gamma

		counter = 0
		l = 1

		while 1.-t[counter] > taufinal: 
			if counter+1 == l: 	# Double length of array if necessary
				xi += l*[0]
				yi += l*[0]

				t += l*[0]

				l = 2*l


			xx1 = xi[counter] 	# set current position
			yy1 = yi[counter]

			tau = 1.-t[counter]

			dt = tau-gamma*tau

			# CHANGE IN POSITION 1
			k1 = -dt * self.flowfield.U(xx1, yy1)/(tau)

			[x1, y1] = [xx1+k1[0]/2., yy1+k1[1]/2.]
			k2 = -dt * self.flowfield.U(x1, y1)/(tau-dt/2.)

			[x2, y2] = [xx1+k2[0]/2., yy1+k2[1]/2.]
			k3 = -dt * self.flowfield.U(x2, y2)/(tau-dt/2.)

			[x3, y3] = [xx1+k3[0], yy1+k3[1]]
			k4 = -dt * self.flowfield.U(x3, y3)/(tau-dt)

			xi[counter+1] = xx1 + k1[0]/6. + k2[0]/3. + k3[0]/3. + k4[0]/6. 	# Take step BACKWARDS in time
			yi[counter+1] = yy1 + k1[1]/6. + k2[1]/3. + k3[1]/3. + k4[1]/6.


			t[counter+1] = t[counter] + dt

			counter+=1 # Increment counter
		
		l = counter+1

		xi = array(xi[0:l]) 	# Truncate list
		yi = array(yi[0:l])

		t = array(t[0:l])

		return [xi, yi, t]

	def lineardensity(self, xi1, yi1, xi2, yi2):
		l = len(xi1)

		if len(yi1)!=l or len(xi2)!=l or len(yi2)!=l:
			print "INVALID INPUTS PATHS: paths must contain the same number of steps."

			return

		lineardensity = zeros(l)
		L = zeros(l)

		lineardensity[0] = 0.
		L[0] = 0.


		ds = sqrt((xi2[0]-xi1[0])**2 + (yi2[0]-yi1[0])**2)

		#LAMBDA = 1/

		for i in range(l-1):
			a = array([xi1[i], yi1[i]])
			b = array([xi2[i], yi2[i]])
			c = array([xi2[i+1], yi2[i+1]])
			d = array([xi1[i+1], yi1[i+1]])

			eta_a = self.h(xi1[i], yi1[i])
			eta_b = self.h(xi2[i], yi2[i])
			eta_c = self.h(xi2[i+1], yi2[i+1])
			eta_d = self.h(xi1[i+1], yi1[i+1])

			mass = self.triarea(a,b,c)*(eta_a+eta_b+eta_c)/3. + self.triarea(a,d,c)*(eta_a+eta_d+eta_c)/3.

			lineardensity[i+1] = lineardensity[i] + (mass)/ds
			#L[i+1] = 




		return array(lineardensity)

	def linearDensityProfile(self, theta1, theta2, n = 2, taufinal = .1, dtheta = 10**-8, eps = 10**-14, plots = False):
		theta = linspace(theta1, theta2, n)

		xi1, yi1, t = self.ri(theta[0]-dtheta, taufinal, eps)
		xi2, yi2, _ = self.ri(theta[0]+dtheta, taufinal, eps)

		densityProfile = zeros([n, len(t)])

		densityProfile[0] = self.lineardensity(xi1, yi1, xi2, yi2)


		if plots:
			fig = plt.figure()
			ax = fig.add_subplot(111)

			temp = linspace(0,2*pi,1000)

			plt.plot(self.a*cos(temp), self.b*sin(temp), 'k')

			ax.set_xlim(-1.05*self.a, 1.05*self.a)
			ax.set_ylim(-1.05*self.b, 1.05*self.b)

			fig.set_size_inches(10*self.a, 10*self.b)

			plt.plot(.5*(xi1+xi2), .5*(yi1+yi2), linewidth = 3.0)
			
		for i in range(1,n):
			xi1, yi1, _ = self.ri(theta[i]-dtheta, taufinal, eps)
			xi2, yi2, _ = self.ri(theta[i]+dtheta, taufinal, eps)

			densityProfile[i] = self.lineardensity(xi1, yi1, xi2, yi2)

			if plots:
				plt.plot(.5*(xi1+xi2), .5*(yi1+yi2), linewidth = 3.0)

		return densityProfile, t, theta










	###############################
	# Voronoi Area Around a Point #
	###############################

	# Adapted from  http://stackoverflow.com/questions/19634993/volume-of-voronoi-cell-python

	def triarea(self,a,b,c):
	    ''' Calculates the area of a triangle with vertices a, b, and, c '''
	    area = abs(cross(c-b,a-b)/2.)
	    return area

	def areacalc(self,vor,p):
	    '''Calculate area of 2d Voronoi cell based on point p. Voronoi diagram is passed in vor.

For each point, the vertices of its voronoi cells are taken and triangulated (Delaunay triangulation).
We then sum the area of each triangle. 
'''
	    dpoints=[] 		# Points in Delauney triangulation
	    area=0 			# Initialize area

	    for v in vor.regions[vor.point_region[p]]: 		# For v, the indices of the vertices of the Voronoi region
	        dpoints.append(list(vor.vertices[v])) 		# Append the vertex of the Voronoi region to dpoints

	    tri=Delaunay(array(dpoints)) 		# Triangulate the vertices
	    for simplex in tri.simplices: 		# For each simplex of the triangulation, add its area to area 
	        area+=self.triarea(array(dpoints[simplex[0]]), array(dpoints[simplex[1]]), array(dpoints[simplex[2]]))
	    return area 		# Return area of the cell

	def computearea(self):
		"""Run areacalc() for each test mass"""

		points=zip(self.x,self.y) 	# Make list of points

		vor=Voronoi(points) 		# compute Voronoi diagram of points

		for i,p in enumerate(vor.points): 	# For each point in the diagram
		    out=False

		    for v in vor.regions[vor.point_region[i]]: 	# Look to see if the area is infinite
		        if v<=-1: # a point index of -1 is returned if the vertex is outside the Vornoi diagram
		            out=True
		    if not out:
		        self.area[i] = self.areacalc(vor,i) 	# If the area is finite, compute the area using areacalc()

		    if out: 	# If the area is infinite
		        self.area[i] = 4*self.a*self.b 	# Set it to the size of the simulation domain

	def computeeta(self):
		"""divide mass by volume (if using voronoi area, must run computearea first)"""
		self.eta = abs(self.mass/self.area)

	def computephi(self):
		"""divide mass by volume (if using voronoi area, must run computearea first)"""
		self.phi = abs(self.mass/(self.area * (1-(self.x/self.a)**2-(self.y/self.b)**2) * self.tau))


	def interpolation(self,x0,x1, n = 100):
		"""Interpolate eta and phi along a line from x0 to x1"""
		X = linspace(x0[0],x1[0],n) 	# Initialize x points
		Y = linspace(x0[1],x1[1],n) 	# Initialize y points

		interpolatedeta = griddata(zip(self.x, self.y), self.eta, (X, Y), method='linear')	# Interpolate eta
		interpolatedphi = griddata(zip(self.x, self.y), self.phi, (X, Y), method='linear') 	# Interpolate phi

		return [X, Y, interpolatedeta, interpolatedphi]		# return X, Y and interpolated data

	def exactsol(self, X, Y):
		"""For a point (or array of points) X and Y, compute exact solution of eta and phi. Only applies for a==b."""
		if self.a != self.b:
			print "Warning! If a!=b, these results do not apply!"

		exacteta = self.tau**(.5) - ((X/self.a)**2. + (Y/self.b)**2.)*self.tau
		exactphi = exacteta/(self.h(X,Y)*self.tau)

		return [exacteta, exactphi]


	######################
	# Plotting Functions #
	######################

	def etaplot(self, vm=None, imint = 'bilinear'):
		"""Plot eta over simulation domain."""

		grid_x, grid_y = mgrid[-self.a:self.a:200j, -self.b:self.b:200j]	# initialize grid where points will be plotted
		points = array([self.x, self.y]).T 		# make list of points
		values = self.mass/self.area 			# values of eta
		grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear') 	# grid_z0 are values that linearly interpolate between test masses

		# Make figure
		fig = plt.figure() 
		ax = fig.add_subplot(111)

		plt.subplot(111)
		im = plt.imshow(grid_z0.T, extent=(-self.a,self.a,-self.b,self.b), origin='lower', cmap=cm.gist_yarg)
		plt.title('ETA')
		im.set_interpolation(imint)

		im.set_clim(vmin = 0, vmax = vm)

		# Plot drop boundary
		theta = linspace(0,2*pi,100)
		plt.plot(self.a*cos(theta),self.b*sin(theta))
		fig.colorbar(im)
		
		ax.set_xlim(-1.1*self.a,1.1*self.a)
		ax.set_ylim(-1.1*self.b,1.1*self.b)
		
		fig.set_size_inches(10*self.a,10*self.b)

	def phiplot(self, vm=None, imint = 'bilinear'):
		"""Plot phi over simulation domain."""
		grid_x, grid_y = mgrid[-self.a:self.a:200j, -self.b:self.b:200j] 	# Initialize grid points
		points = array([self.x, self.y]).T 		# Make list of positions
		values = self.tau*self.mass/(self.area * (1-(self.x/self.a)**2 - (self.y/self.b)**2)) 	# values of phi
		values = values*(values>0)

		grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear') 	# values interpolated between our points

		# Make figure
		fig = plt.figure()
		ax = fig.add_subplot(111)

		plt.subplot(111)
		im = plt.imshow(grid_z0.T, extent=(-self.a,self.a,-self.b,self.b), origin='lower', cmap=cm.gist_yarg)
		plt.title('PHI')
		im.set_interpolation(imint)
		fig.colorbar(im)

		im.set_clim(vmin = 0, vmax = vm)

		# Plot drop boundary
		theta = linspace(0,2*pi,100)
		plt.plot(self.a*cos(theta),self.b*sin(theta))
		
		ax.set_xlim(-1.1*self.a,1.1*self.a)
		ax.set_ylim(-1.1*self.b,1.1*self.b)
		
		fig.set_size_inches(10*self.a,10*self.b)


	############################
	# Save and Load Simulation #
	############################

	def savesim(self, filename):
		"""Save simulation. Warning: Does not save flowfield"""
		simfile = [self.x, self.y, self.a, self.b, self.tau, self.gamma, self.mass, self.area, self.eta, self.phi, self.dx0, self.voronoi]

		save(filename, simfile)
		print( "Simulation saved!     " + filename)

	def loadsim(self, filename):
		"""Load simulation from saved file. Warning: Does not load flowfield"""
		simfile = load(filename)

		[self.x, self.y, self.a, self.b, self.tau, self.gamma, self.mass, self.area, self.eta, self.phi, self.dx0, self.voronoi] = simfile


	#####################
	# Helpful Functions #
	#####################

	def h(self,X,Y):
		"""h(x,y) returns the time independent height at points (x,y). x and y can be lists of points."""
		ellipse = ((X/self.a)**2 + (Y/self.b)**2 < 1.) 		# If (x,y) is outside of the ellipse, set to 0

		return (1 - (X/self.a)**2 - (Y/self.b)**2)*ellipse 	# Return height

	def vol(self, xx, yy, ddxx, ddyy):
		"""Compute volume in a cell centered at (xx,yy) with width (ddxx, ddyy)"""
		# vol at x = vol at -x, by symmetry
		x = abs(xx)
		y = abs(yy)

		# find bottom left corner of square
		x0 = x-ddxx/2.
		y0 = y-ddyy/2.

		# if cell lies on an axis, divide it so that it lies fully in one quadrant
		if x == 0:
			return 2*self.vol(x+ddxx/4., y, ddxx/2., ddyy)
		if y == 0:
			return 2*self.vol(x, y+ddyy/4., ddxx, ddyy/2.)

		# If the cell is outside the drop, return 0
		if 1-(x0/self.a)**2-(y0/self.b)**2<=0:
			return 0
		# if the cell is inside the drop, integrate across the entire cell
		if 1-(x+ddxx/2.)**2/self.a**2-(y+ddxx/2.)**2/self.b**2>=0:
			return ddxx*ddyy*(1-x**2/self.a**2-y**2/self.b**2-ddxx**2/(12.*self.a**2)-ddyy**2/(12.*self.b**2))#

		yint = self.b*sqrt(1-(x0)**2/self.a**2)
		xint = self.a*sqrt(1-(y0)**2/self.b**2)
        
		# If cell can be integrated to boundary exactly, do it
		if yint <= y+ddyy/2. and xint <= x+ddxx/2.:
			return x0*y0*(1-x0**2/(3.*self.a**2)-y0**2/(3.*self.b**2)) \
					+ self.b*x0*(2*x0**2/self.a**2-5.)*sqrt(1-x0**2/self.a**2)/12. \
					+ self.a*y0*(2*y0**2/self.b**2-5.)*sqrt(1-y0**2/self.b**2)/12. \
					+ self.a*self.b*(arctan(sqrt(self.b**2-y0**2)/y0) - arctan(x0/sqrt(self.a**2-x0**2)))/4.

		# Else, use recursion to evaluate small cells in terms of other cells that can be evaluated exactly.
		else:
			return self.vol(x+ddxx/2., y+ddyy/2., ddxx*2, ddyy*2) - self.vol(x+ddxx, y, ddxx, ddyy) \
					- self.vol(x, y+ddyy, ddxx, ddyy) \
					- self.vol(x+ddxx, y+ddyy, ddxx, ddyy)




