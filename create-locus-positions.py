import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import pickle


def bernstein_poly(i, n, t):
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def get_bezier_parameters(X, Y, degree=3):

    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final

def bezier_curve(points, nTimes=50):

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


###############################
standard_room_dim = [7,5.5,2.4]


cols, rows = 1, 1
width, height = 6.0, 7.0

w = width / cols
h = height / rows

result = []
for c in range(cols + 1):
    c1 = c* (h/4 + 0.5)
    for r in range(rows + 1):
        c2= r*w/4
        result.append(((c + h/4+c1), (r + w/4)+c2))

print(result)

locusFinal = []

for ii in range(100):

    for i in range(len(result)):
        points = []
        
        xpoints = result[i]
        xpoints = list(xpoints)
        xpoints.append(xpoints[0])
        xpoints.append(xpoints[1])
        tuple(xpoints)
        
        if i == len(result)-1:
           i =0
        ypoints = result[i+1]
        
        ypoints = list(ypoints)
        ypoints.append(1.75)
        ypoints.append(2.75)
        tuple(ypoints)
        
                   
        for i in range(len(xpoints)):
            k = xpoints[i]
            l = ypoints[i]
            if ii >0:
                k = xpoints[i] + 0.025
                l = ypoints[i] + 0.025
            points.append([xpoints[i],ypoints[i]])
            
        plt.plot(xpoints, ypoints, "ro",label='Original Points')
        # Get the Bezier parameters based on a degree.
        data = get_bezier_parameters(xpoints, ypoints, degree=3)
        x_val = [x[0] for x in data]
        y_val = [x[1] for x in data]
        
        # Plot the control points
        plt.plot(x_val,y_val,'k--o', label='Control Points')
        # Plot the resulting Bezier curve
        xvals, yvals = bezier_curve(data, nTimes=50)
        plt.plot(xvals, yvals, 'b-', label='B Curve')
        plt.legend()
        plt.show()    
        
        locus = []
        
        for x, y in zip(xvals, yvals):
           xx = round(x,2)
           yy = round(y,2)
           locus.append([xx, yy,1.4])
        locusFinal.append(locus)
def flat_list(test_list):
    return [item for sublist in test_list for item in sublist]

#locusFinal = flat_list(locusFinal)

print(locusFinal)

with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/locus_pos_list_20000.pickle', 'wb' ) as f:
    pickle.dump(locusFinal, f) 
