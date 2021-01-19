import math
import numpy as np


##################################################################################################################################
# r-Nearest neighbors is a modified version of the k-nearest neighbors. The issue with k-nearest neighbors is the choice of k.   #
# A smaller k, the classifier would be more sensitive to outliers.                                                               #
# If the value of k is large, then the classifier would be including many points from other classes.                             #
# It is from this logic that we get the r near neighbors algorithm.                                                              #
###################################################################################################################################

#  radius neighbors algorithm
def rNN(pcl, p, r = 1):  
    '''  
    This function classifies the point p using  
    r k neareast neighbour algorithm. It assumes only   
    two groups and returns 0 if p belongs to class 0, else  
    1 (belongs to class 1).  

    Parameters -  
            points : training pcl

            p : test data point of form (x, y,z)  
            k : radius of the r nearest neighbors  
    '''

    cluster = []
    for fruit in pcl:
            if math.sqrt((fruit[0]-p[0])**2 + (fruit[1]-p[1])**2 + (fruit[2]-p[2])**2) <= r:
                cluster.append(fruit)
    return cluster


# Driver function  
def main():  

    points = np.array([[1.5, 4, 3.3], [1.8, 3.8, 3.2], [1.65, 5, 3.2], [2.5, 3.8, 3.3], [3.8, 3.8, 3.4], [5.5, 3.5, 3.3], [5.6, 4.5, 3.3], [6, 5.4, 3.2], [6.2, 4.8, 3.4], [6.4, 4.4, 3.3]])  

    # query point p(x, y,z)  
    p = np.array([4.5, 4, 3.5])  

    # Parameter to decide the class of the query point  
    r = 2

    print("The cluster of the query point is: {}".format(rNN(points, p, r)))  



if __name__ == '__main__':
    main()  