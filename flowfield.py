#####################################
#   Table of Contents:              #
#                                   #
#   - Support Functions             #
#   - Compute Flow Field            #
#   - Polynomial Fit                #
#   - Plotting Functions            #
#   - Save Simulation               #
#                                   #
#####################################

from numpy import *
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import itertools

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class flow(object):
    """The flow class is used to numerically compute the flowfield due to uniform evaporation in an elliptical drop.

An evaporating drop is subject to 2 governing equations:
(1)     div(uh) = -h_t-J_0
(2)     curl (u/h^2) = 0
Equation (1) is saying that the mass of fluid is conserved. Divergence in the flow balances the local height change
and area flux. Equation (2) tells us that the pressure ( P ~ u/h^2 ) can be described by a scalar field.

We solve these equations on a grid by recasting them in an integral form so the flux (u*h) is explicitly conserved
throughout the drop and pressure is explicitly path independent.
"""
    def __init__(self, a, b , J, L):
        """Initialize Simulation"""
        self.a = float(a)   # major axis of ellipse
        self.b = float(b)   # minor axis
        self.J = J          # resolution in x
        self.L = L          # resolution in y
        self.dx = 2.*a/J    # grid spacing in x
        self.dy = 2.*b/L    # grid spacing in y

        
    #####################
    # Support functions #
    #####################
    
    def hdx(self,x,y):
        """Compute integral of h dx between cells"""
        # only integrate to the edge of the droplet. h cannot be negative
        x1 = min(max(x-self.dx/2, -self.a*sqrt(max(1-y**2/self.b**2,0))), self.a*sqrt(max(1-y**2/self.b**2,0)))
        x2 = max(min(x+self.dx/2, self.a*sqrt(max(1-y**2/self.b**2,0))), -self.a*sqrt(max(1-y**2/self.b**2,0)))

        return (x2 - x2**3/(3*self.a**2) - (x2*y**2)/self.b**2) - (x1 - x1**3/(3*self.a**2) - (x1*y**2)/self.b**2)
    
    def hdy(self,x,y):
        """Compute integral of h dy between cells"""
        y1 = min(max(y-self.dy/2, -self.b*sqrt(max(1-x**2/self.a**2,0))), self.b*sqrt(max(1-x**2/self.a**2,0)))
        y2 = max(min(y+self.dy/2, self.b*sqrt(max(1-x**2/self.a**2,0))), -self.b*sqrt(max(1-x**2/self.a**2,0)))
    
        return (y2 - y2**3/(3*self.b**2) - (y2*x**2)/self.a**2) - (y1 - y1**3/(3*self.b**2) - (y1*x**2)/self.a**2)
        
    # Compute flux through surface (-h_t - J_0/rho)
    def source(self, xx, yy, ddxx, ddyy):
        """Exactly integrates -h_t -J_0 over a ddxx by ddyy cell centered at (xx, yy)"""

        # source at x = source at -x, by symmetry
        x = abs(xx)
        y = abs(yy)
    
        # find bottom left corner of square
        x0 = x-ddxx/2.
        y0 = y-ddyy/2.
    
        # if cell lies on an axis, divide it so that it lies fully in one quadrant
        if x == 0:
            return 2*self.source(x+ddxx/4., y, ddxx/2., ddyy)
        if y == 0:
            return 2*self.source(x, y+ddyy/4., ddxx, ddyy/2.)
    
        # If the cell is outside the drop, return 0
        if 1-(x0)**2/self.a**2-(y0)**2/self.b**2<=0:
            return 0
    
        # If the cell is inside the drop, integrate across the entire cell
        if 1-(x+ddxx/2.)**2/self.a**2-(y+ddxx/2.)**2/self.b**2>=0:
            return ddxx*ddyy*(.5-x**2/self.a**2-y**2/self.b**2-ddxx**2/(12.*self.a**2)-ddyy**2/(12.*self.b**2))

        yint = self.b*sqrt(1-(x0)**2/self.a**2)
        xint = self.a*sqrt(1-(y0)**2/self.b**2)
    
        # If cell can be integrated to boundary exactly, do it 
        if yint <= y+ddyy/2. and xint <= x+ddxx/2.:
            return x0*y0*(.5-x0**2/(3.*self.a**2)-y0**2/(3.*self.b**2)) + self.b*x0*(x0**2/self.a**2-1.)*sqrt(1-x0**2/self.a**2)/6. \
                                                          + self.a*y0*(y0**2/self.b**2-1.)*sqrt(1-y0**2/self.b**2)/6.
        # Else, use recursion to evaluate small cells in terms of other cells that can be evaluated exactly.
        else:
            return self.source(x+ddxx/2., y+ddyy/2., ddxx*2, ddyy*2) - self.source(x+ddxx, y, ddxx, ddyy)\
                                                          - self.source(x, y+ddyy, ddxx, ddyy)\
                                                          - self.source(x+ddxx, y+ddyy, ddxx, ddyy)

    ######################
    # COMPUTE FLOW FIELD #
    ######################
    def computeflowfield(self):
        """Write a matrix equation that casts the integral forms of (1) and (2) as finite difference equations of the form
    A.x = b,
where x is a vector representing our final solution (u,v); A represents the differential operators on the left of the equation;
and b represents the source term on the right. We then solve for x.

After solving for x, we know the values of u and v along the walls between cells.
"""
        a = self.a
        b = self.b
        J = self.J
        L = self.L
        dx = self.dx
        dy = self.dy
    
        # Number of equations for equation 1
        N = (J+1)*(L+1)
    
        # Total number of equations
        M = (J+1)*(L+1)+J*L
    
        # Find Velocity Field
        # Initialize the right hand side of the equation A.x = b
        bb = zeros(M)
    
        # Initialize arrays that will be used to define our sparse matrix. vV are the values (coefficients in the equation),
        # iI and jJ are the coordinates within the matrix.
        vV = zeros(4*M)
    
        iI = zeros(4*M)
        jJ = zeros(4*M)
    
        # Build Equation Matrix
    
        # Mass Conservation, (equation 1)
        for j in range(J+1):
            for l in range(L+1):
                # Find positions of grid cell
                x = j*dx - a
                y = l*dy - b
            
                # set equation index
                i = (L+1)*j+l
            
                # indices for u and v
                ui = (J)*l + j
                vi = (J)*(L+1) + (L)*j + l
        
                # Equation index
                iI[4*i:4*(i+1)] = [i, i, i, i]
            
                # compute height integrals (areas of cell walls)
                hdy1 = self.hdy(x-dx/2., y)
                hdy2 = self.hdy(x+dx/2., y)
                hdx1 = self.hdx(x, y-dy/2.)
                hdx2 = self.hdx(x, y+dy/2.)        
            
                # if any integral is nonzero, part of the cell is in the drop. Compute flux through wall. 
                if hdx1!=0 or hdx2!=0 or hdy1!=0 or hdy2!=0:
                    # set flux equal to source
                    bb[i] = self.source(x,y,dx,dy)
                
                    # if the cell has area, use to solve for ui
                    if hdy2!=0:
                        jJ[4*i+0] = ui
                        vV[4*i+0] = hdy2
                        
                    # else, set flux to eps, the "leak" for the total flux integration
                    else:
                        jJ[4*i+0] = M-1
                        vV[4*i+0] = 1
                    
                    if hdy1!=0:
                        jJ[4*i+1] = ui-1
                        vV[4*i+1] = -hdy1
                    
                    else:
                        jJ[4*i+1] = M-1
                        vV[4*i+1] = 1
                
                    if hdx2!=0:
                        jJ[4*i+2] = vi
                        vV[4*i+2] = hdx2
    
                    else:
                        jJ[4*i+2] = M-1
                        vV[4*i+2] = 1
                
                    if hdx1!=0:
                        jJ[4*i+3] = vi-1
                        vV[4*i+3] = -hdx1
    
                    else:
                        jJ[4*i+3] = M-1
                        vV[4*i+3] = 1
    
                # if the cell is entirely outside the drop, set u-velocity along appropriate wall to 0.
                else:
                    bb[i] = 0
                    vV[4*(i):4*(i+1)] = [0, 0, 0, 1]
                
                    if x<0:
                        jJ[4*(i):4*(i+1)] = [ui, ui, ui, ui]
    
                    if x>0:
                        jJ[4*(i):4*(i+1)] = [ui-1, ui-1, ui-1, ui-1]
    
        # Curl of u/h^2 = 0 (equation 2)
        for j in range(J):
            for l in range(L):
            
                # Find positions of vertices of grid
                x = j*dx - a + .5*dx
                y = l*dy - b + .5*dy
                
                i = L*j+l
                ui = J*l + j
                vi = (J)*(L+1) + L*j + l
            
                h = 1-x**2/a**2-y**2/b**2
                
                iI[4*(N+i):4*(N+i+1)] = [(J+1)*(L+1)+i, (J+1)*(L+1)+i, (J+1)*(L+1)+i, (J+1)*(L+1)+i]
            
                # if the vertex lies inside the drop, enforce equation (2)
                if h>0:
                    jJ[4*(N+i):4*(N+i+1)] = [vi+L, vi, ui+J, ui]
                    vV[4*(N+i):4*(N+i+1)] = [h/(dx)+2*x/a**2, -h/(dx)+2*x/a**2, -h/(dy)-2*y/b**2, h/(dy)-2*y/b**2]
                    
                    bb[N+i] = 0
            
                # else, set appropriate v-velocity to 0. 
                elif x<0:
                    jJ[4*(N+i):4*(N+i+1)] = [vi, vi, vi, vi]
                    vV[4*(N+i):4*(N+i+1)] = [0, 0, 0, 1]
                    
                    bb[N+i] = 0
            
                elif x>0:
                    jJ[4*(N+i):4*(N+i+1)] = [vi+L, vi+L, vi+L, vi+L]
                    vV[4*(N+i):4*(N+i+1)] = [0, 0, 0, 1]
                    
                    bb[N+i] = 0
    
        # Solve
        # Create matrix, A
        A = sparse.csr_matrix((vV,(iI,jJ)),shape=(M,M))
        
        # Solve for velocity field
        self.uvec = spsolve(A,bb)

    ##################
    # POLYNOMIAL FIT #
    ##################
    def ufit(self, order=10):
        """Fit a polynomial of the form u(x,y) = sum_{i,j}^{2*order} a_ij x^i y^j to the data.

The polynomial fit explicitly enforces symmetries of the flow (u is odd in x and even in y, v is odd in y and even in x)
"""
        # initialize arrays for the x, y, and u values of the x-velocity
        ux = zeros(self.J*(self.L+1))
        uy = zeros(self.J*(self.L+1))
        uu = zeros(self.J*(self.L+1))

        # for each cell wall, add x, y and u value to array
        for j in range(self.J):
            for l in range(self.L+1):
                x = j*self.dx-self.a+self.dx/2.
                y = l*self.dy-self.b

                ui = self.J*l + j

                ux[ui] = x
                uy[ui] = y
                uu[ui] = self.uvec[ui]

        # remove points outside drop.
        test = (ux/self.a)**2. + (uy/self.b)**2. < 1

        ux = ux[test]
        uy = uy[test]
        uu = uu[test]

        #repeat for y-velocity
        vx = zeros(self.L*(self.J+1))
        vy = zeros(self.L*(self.J+1))
        vv = zeros(self.L*(self.J+1))

        for j in range(self.J+1):
            for l in range(self.L):
                x = j*self.dx-self.a
                y = l*self.dy-self.b+self.dy/2.

                vi = self.L*j + l

                vx[vi] = x
                vy[vi] = y
                vv[vi] = self.uvec[self.J*(self.L+1) + vi]

        test = (vx/self.a)**2. + (vy/self.b)**2. < 1
       
        vx = vx[test]
        vy = vy[test]
        vv = vv[test]

        # solve using least squares fit
        ncols = (order + 1)**2      # number of coefficients
        uG = zeros((ux.size, ncols))    # initial vectors to hold data of the form ux**i * uy**j. These will be used to make fit
        vG = zeros((vx.size, ncols))
        
        ij = itertools.product(range(order+1), range(order+1))  #for {i,j}
        for k, (i,j) in enumerate(ij):
            uG[:,k] = ux**(2*i+1) * uy**(2*j)   # let uG = polynomial with symmetries enforced
            vG[:,k] = vx**(2*i) * vy**(2*j+1)

        self.um, _, _, _ = linalg.lstsq(uG, uu) # solve for coefficients using least squares fit
        self.vm, _, _, _ = linalg.lstsq(vG, vv)
        
        erru = linalg.norm(self.U(ux,uy)[0]-uu)     # return |ufit - udata| (unnormalized)
        errv = linalg.norm(self.U(vx,vy)[1]-vv)

        return (erru, errv)


    def U(self, x, y):
        """This function returns u(x,y) from the polynomial fit."""
        ellipse = ((x/self.a)**2+(y/self.b)**2<=1.) #boolean that determines whether (x,y) is inside drop

        uorder = int(sqrt(len(self.um))) - 1
        vorder = int(sqrt(len(self.um))) - 1
        
        ij = itertools.product(range(uorder+1), range(uorder+1))
        u = zeros_like(x)
        
        # evaluate polynomial for u
        for a, (i,j) in zip(self.um, ij):
            u += a * x**(2*i+1) * y**(2*j)

        ij = itertools.product(range(vorder+1), range(vorder+1))
        v = zeros_like(x)
        
        # evaluate polynomial for v
        for a, (i,j) in zip(self.vm, ij):
            v += a * x**(2*i) * y**(2*j+1)

        # return u,v
        return array([u*ellipse,v*ellipse])

    def UX(self, x, y):
        """Evaluate du/dx and dv/dy. Note: This does NOT evaluate the full Jacobian"""
#        ellipse = ((x/self.a)**2+(y/self.b)**2<=1.)

        uorder = int(sqrt(len(self.um))) - 1
        vorder = int(sqrt(len(self.um))) - 1
        
        ij = itertools.product(range(uorder+1), range(uorder+1))
        ux = zeros_like(x)
        
        # Find du/dx
        for a, (i,j) in zip(self.um, ij):
            ux += a * (2*i+1) * x**(2*i+1-1) * y**(2*j)

        ij = itertools.product(range(vorder+1), range(vorder+1))
        vy = zeros_like(x)
        
        # dv/dy
        for a, (i,j) in zip(self.vm, ij):
            vy += a * (2*j+1) * x**(2*i) * y**(2*j+1-1)

        return array([ux,vy])

    def UY(self, x, y):
        """Evaluate off diagonal terms of Jacobian. Returns du/dy, dv/dx."""
#        ellipse = ((x/self.a)**2+(y/self.b)**2<=1.)

        uorder = int(sqrt(len(self.um))) - 1
        vorder = int(sqrt(len(self.um))) - 1
        
        ij = itertools.product(range(uorder+1), range(uorder+1))
        uy = zeros_like(x)
        
        # Find du/dy
        for a, (i,j) in zip(self.um, ij):
            if j != 0:
                uy += a * (2*j) * x**(2*i+1) * y**(2*j-1)

        ij = itertools.product(range(vorder+1), range(vorder+1))
        vx = zeros_like(x)
        
        # dv/dx
        for a, (i,j) in zip(self.vm, ij):
            if i!=0:
                vx += a * (2*i) * x**(2*i-1) * y**(2*j+1)

        return array([uy,vx])

    def divU(self, x, y):
        """Evaluate div(u) at a point (or array) x,y"""
#        ellipse = ((x/self.a)**2+(y/self.b)**2<=1.)

        ux = self.UX(x,y)

        return ux[0]+ux[1]

    def dvpdn(self, x, y):
        [u, v] = self.U(x,y)

        if u==0 and v ==0:
            return 0.

        [ux, vy] = self.UX(x,y)
        [uy, vx] = self.UY(x,y)

        nrm = sqrt(u**2 + v**2)

        cosTH = u/nrm
        sinTH = v/nrm

        dvpdn = ux*sinTH**2 - vx*sinTH*cosTH - uy*sinTH*cosTH + vy*cosTH**2

        return dvpdn




    ######################
    # PLOTTING FUNCTIONS #
    ######################
    def flowplot(self,dn=0):
        """Return a vector field plot of the flow within the drop."""
        J = self.J
        L = self.L

        # set spacing (in grid cells) of vectors
        if dn==0:
            dn = max(1,J/2**3,L/2**3)
        
        # define outer edge of the drop
        theta = linspace(0, 2*pi, 100)
        ex = self.a*cos(theta)
        ey = self.b*sin(theta)        
        
        # the starting point of each vector is the center of the cell. The direction and magnitude is determined by the average of u on its edges
        [xs, ys, us, vs] = array([\
            [j*self.dx - self.a, l*self.dy - self.b,\
            ((abs(j*self.dx - self.a)+self.dx/2)**2/self.a**2 + (abs(l*self.dy - self.b)+self.dy/2)**2/self.b**2 < 1)*(self.uvec[J*l+j]+self.uvec[J*l+j-1])/2,\
            ((abs(j*self.dx - self.a)+self.dx/2)**2/self.a**2 + (abs(l*self.dy - self.b)+self.dy/2)**2/self.b**2 < 1)*(self.uvec[J*(L+1)+L*j+l]+self.uvec[J*(L+1)+L*j+l-1])/2\
            ] for j in range(0,J+1,dn) for l in range(0,L+1,dn)]).T

        # Make plot using quiver
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.quiver(xs,ys,us,vs,angles='xy',scale_units='xy', scale=2)
 
        ax.set_xlim([-self.a*1.1, self.a*1.1])
        ax.set_ylim([-self.b*1.1, self.b*1.1])
        fig.set_size_inches(10*self.a,10*self.b)

        plt.draw()
        plt.hold(True)
        plt.plot(ex,ey,'b')
        plt.hold(False)

    def fitplot(self):
        """Color map of the velocity field of the drop, as determined by ufit"""
        
        # initialize grid
        X = linspace(-self.a,self.a,200)
        Y = linspace(-self.b,self.b,200)

        X, Y = meshgrid(X, Y)

        # Find U and V on grid
        Z = self.U(X,Y)

        # Plot u
        fig = plt.figure()
        fig.add_subplot(111)
        fig.set_size_inches(10*self.a,10*self.b)
        
        imgplot = plt.imshow(Z[0], cmap='coolwarm', extent=(-self.a,self.a,-self.b,self.b))
        imgplot.set_interpolation('nearest')
    
        plt.hold(True)
        fig.colorbar(imgplot)
        imgplot.set_cmap('coolwarm')

        # Plot v
        fig = plt.figure()
        fig.add_subplot(111)
        fig.set_size_inches(10*self.a,10*self.b)
        
        imgplot = plt.imshow(Z[1,::-1], cmap='coolwarm', extent=(-self.a,self.a,-self.b,self.b))
        imgplot.set_interpolation('nearest')
    
        plt.hold(True)
        fig.colorbar(imgplot)
        imgplot.set_cmap('coolwarm')        


    #################
    # SAVE SOLUTION #
    #################
    def saveuvec(self, path=''):
        """Save velocity vector."""
        
        savetxt(path+'uvec_a='+str(self.a)+'_b='+str(self.b)+'_J='+str(self.J)+'_L='+str(self.L)+'.txt', self.uvec)
        print 'Flow Field Saved!     ' +path+'uvec_a='+str(self.a)+'_b='+str(self.b)+'_J='+str(self.J)+'_L='+str(self.L)+'.txt'


                