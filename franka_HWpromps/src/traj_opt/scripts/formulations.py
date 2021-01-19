import numpy as np
import math
import time, random
import matplotlib.pyplot as plt
#from mathutils.geometry import intersect_point_line

def perpendicular(v):
    """ Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # clip --> limit the values in an array to min and max


def normalize(a):
    return a/np.linalg.norm(a)


def get_orientation(COG, tip):
    v = np.sqrt((COG[0]-tip[0])**2 + (COG[1]-tip[1])**2 + (COG[2]-tip[2])**2) 
    return np.linalg.norm(v)

def get_StemLength(X, X_orient, h_tabletop, theta):
    u = np.array([0, 0, h_tabletop]) # to b changed to: np.array([0, h_tabletop, 0])
    v = X_orient
    # intersection of the 2 directions gives the root point
    # eq of 3d line passing by fruit COG and with direction X_orient
    #y_inter = u[1] - X[1] / v[1]  # since the intersection point lies in the plane of the tabletop 
    #x_inter = (v[0] * y_inter) + X[0]
    #z_inter = (v[2] * y_inter) + X[2]
    deltaZ = u[2] - X[2]
    Lstem = deltaZ / np.cos(theta)
    X_root_vect = v * Lstem
    z_inter = u[2]  # since the intersection point lies in the plane of the tabletop , to b changed to y_inter
    x_inter = X_root_vect[0] + X[0] # from: x_inter - X[0] = vector of unit  X_orient and length z_inter
    y_inter = X_root_vect[1] + X[1]
    inter = np.array([x_inter, y_inter, z_inter]) # root of stem
    #L = np.sqrt((inter[0]-X[0])**2 + (inter[1]-X[1])**2 + (inter[2]-X[2])**2) 
    return Lstem, inter


def plot_2dseg(point, Zangle, length):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''

    # unpack the first point
    x, y = point

    # find the end point
    endy = length * math.sin(math.radians(Zangle)) 
    endx = length * math.cos(math.radians(Zangle))

    # plot the points
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_ylim([0, 10])   # set the bounds to be 10, 10
    ax.set_xlim([0, 10])
    ax.plot([x, endx], [y, endy])

    fig.show()


def plot_3dseg(point, angle, length):
    '''
    point - Tuple (x, y, z)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''

    # unpack the first point
    x, y, z = point[0], point[1], point[2]
    Pangle, Yangle = angle[0], angle[1] # spherical angles: 2 define 2d space, direction angles: 3 define 3d space, here plane is xy

    # find the end point
    endz = length * np.cos(Pangle) + z 
    endx = length * np.sin(Pangle) + x
    endy = length * np.sin(Yangle) + y

    ## plot the points
    #fig = plt.figure()
    #ax = plt.subplot(111)
    #ax.set_ylim([0, 10])   # set the bounds to be 10, 10
    #ax.set_xlim([0, 10])
    #ax.set_zlim([0, 10])
    return [x, endx], [y, endy], [z, endz]

    #fig.show()


def closest_pt_seg(point, seg ):
    #line = ((0.0,0.0,0.0), (1.0,1.0,1.0))
    #point = (0.0,0.2,0.5)

    intersect = intersect_point_line(point, seg[0], seg[1])

    print('point is closest to',intersect[0],'on the line')
    distance1 = (intersect[0] - line[0]).length
    distance2 = (intersect[0] - line[1]).length

    if distance1 < distance2:
        print('The point is closer to the start of the line')
    else:
        print('The point is closer to the end of the line')


def plot_sphere(center, radius):
    #wframe = None
    #tstart = time.time()
    #for num in range(100):
        #oldcol = wframe
        #r = random.randint(1, 20)
    r = radius
    alpha = 1.0 / random.randint(1, 2)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x,y,z, alpha

    #print('Duration: %f' % (100 / (time.time() - tstart)))


class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
          
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Colinear orientation 
        return 0
  
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. (2d intersection lines)
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False
  
# Driver program to test above functions: 
p1 = Point(1, 1) 
q1 = Point(10, 1) 
p2 = Point(1, 2) 
q2 = Point(10, 2) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
  
p1 = Point(10, 0) 
q1 = Point(0, 10) 
p2 = Point(0, 0) 
q2 = Point(10,10) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
  
p1 = Point(-5,-5) 
q1 = Point(0, 0) 
p2 = Point(1, 1) 
q2 = Point(10, 10) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
      
# This code is contributed by Ansh Riyal 

if __name__ == "__main__":    
    a = np.array([1,2, 5])
    print perpendicular(normalize(a))
