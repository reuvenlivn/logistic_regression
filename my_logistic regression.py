# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:24:35 2019

@author: reuve
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

##create the clusters 
# input: number of points in one cluster, number of clusters, noise_level
def random_clusters(pnt_in_cluster, n_clusters, std):
    center = np.random.rand(2)
#    print("center\t",center)
    angles = 2 * np.pi * np.linspace(0,1,n_clusters+1)[:-1]
    centers = np.c_[np.cos(angles),np.sin(angles)] + center
    noise = np.random.normal(0,std,(n_clusters, pnt_in_cluster, 2))
    points = np.repeat(np.expand_dims(centers,1) ,pnt_in_cluster, axis=1)
    return (points + noise)

# indexes of train and test parts
def train_and_test(X, Y, test_part):
    per_index=int(len(Y)*(1-test_part))
    return (X[:per_index,...],
            X[per_index:,...],
            Y[:per_index],
            Y[per_index:])

#
def sigmoid(x):
    return (1. / (1 + np.exp(-x)))

#
def cost(res, y):  
    #  fix "RuntimeWarning: divide by zero encountered in log" by epsolin
    epsilon = 1e-7
    cost = -(y*np.log(res+epsilon) + (1-y)*np.log(1-res+epsilon)).mean()
    return (cost)

# use sigmoid() to predict the class
# >05 : class 1; 
# <=0.5 class 0
def predict(x, teta):
    return (sigmoid(np.dot(x,teta))>0.5)

#
def gradient_descent_sigmoid(X, Y, start, rate, iterations):
    t=start.copy()  
    for iter in range(iterations):
        res=sigmoid(np.dot(X,t))
        loss=cost(res,Y)
        grad=np.dot((res-Y),X)/len(X)
        t=t-rate*grad
        print('iter {}, loss {}, new t {}'.format(iter, loss, t))
    return (t)

#
def create_circles(points_in_cluster, n_clusters, std):
    center = np.random.rand(2)
    angles = 2*np.pi*np.random.rand(n_clusters, points_in_cluster)

    arrays = []
    for i in range(n_clusters):
        offset = 1+i
        arrays.append(np.random.normal(0, std, points_in_cluster)+offset)
            
    radii = np.array(arrays)
    x_y_coords = np.array([radii*np.cos(angles)+center[0],
                           radii*np.sin(angles)+center[1]])

    # The two swap axes will re-shuffle the dimensions to the order of n_clusters X n_points_in_cluster X Coordinates
    return (np.swapaxes(np.swapaxes(x_y_coords,0,2), 0, 1))


def circle_prediction(X, t):
    circle = (X[:,0]-t[0])**2 + (X[:,1]-t[1])**2 - t[2]**2
    res = sigmoid(circle)>0.5
    return (res)


def gradient_descent_circles_sigmoid(X, Y, start, rate, iterations):
    t=start.copy()  
    for iter in range(iterations):
        res=circle_prediction(X,t)
        loss=cost(res,Y)
        # Derivative of the circle formula for each t
        model_grad_vec = -np.c_[2*(X[:,0]-t[0]), 
                                2*(X[:,1]-t[1]),
                                2*t[2]*np.ones(len(X))]
        # pred - y is the combined derivative of cross entropy loss and logit (or softmax)
        # We multiply it by the chain rule, the dot sums the results among different values of X, len applies average
        grad=np.dot((res-Y),model_grad_vec)/len(X)
        t=t-rate*grad
        print('iter {}, loss {}, new t {}'.format(iter,loss,t))
    return (t)

########
#main 
#1
points_in_cluster   = 300
n_clusters          = 2     # two clusters only
std                 = 0.6 # how much the clusters are separate
std_circle          = 0.4
learning_rate       = 0.2

# create random data 
data = random_clusters(points_in_cluster, n_clusters, std)
data_points=data.reshape(-1,2)

#2
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
nn = points_in_cluster
for i in range(n_clusters):
#    print('idx1 {}, idx2 {} color{}'.format(nn*i,nn*(i+1),colors[i]) )
# plot the random data points
    plt.scatter( 
            data_points[nn*i:nn*(i+1),0],
            data_points[nn*i:nn*(i+1),1],
            color=colors[i], 
            marker='.')
#3
# Y should be the prediction(0,1) 
Y = np.repeat(np.array(range(n_clusters)),points_in_cluster)
X = np.c_[data_points,np.ones(data_points.shape[0])]

test_part = 0.2     # paert of the data to allocate for testing
indexes = np.array(range(points_in_cluster*n_clusters))
np.random.shuffle(indexes)

# set the indexes for train and test.
X_train, X_test, Y_train, Y_test = train_and_test(X[indexes,:],
                                                  Y[indexes],
                                                  test_part)
  
start = np.append(np.random.normal(0,0.1,(2)),0)
teta = gradient_descent_sigmoid(X_train, Y_train, 
                                start, learning_rate, points_in_cluster                           
                                )

#calculate the precision of the train and test data
train_precision=(predict(X_train, teta)==Y_train).mean()
test_precision=(predict(X_test, teta)==Y_test).mean()

print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(teta)

#decision boundary
# y=mx+b
#mean_x = (min(X[:,0]) + max(X[:,0])) * 0.5
plt.axis(xmin=min(X[:,0]),
         xmax=max(X[:,0]),
         ymin=min(X[:,1]),
         ymax=max(X[:,1]),
         )
#x_boundary = np.array([mean_x-0.25, mean_x+0.25])
x_boundary = np.array([min(X[:,0]), max(X[:,0])])
y_boundary = -(teta[0] / teta[1] * x_boundary)/teta[2]
plt.plot(x_boundary, y_boundary)
         
plt.show()

#  10  
circles=create_circles(points_in_cluster, n_clusters, std_circle)

for i in range(circles.shape[0]):
    # plot each set of points
    plt.scatter(circles[i,:,0], circles[i,:,1], 
                color=colors[i], marker='.')
    # define coundaries of plot
    plt.xlim(circles[:,:,0].min(), circles[:,:,0].max())
    plt.ylim(circles[:,:,1].min(), circles[:,:,1].max())


data_points = circles.reshape(-1,2)
X = data_points
#  create labels
Y = np.repeat(np.array(range(n_clusters)), points_in_cluster)
X = np.c_[data_points,np.ones(data_points.shape[0])]
indexes= np.array(range(points_in_cluster*n_clusters))
np.random.shuffle(indexes)
X_train, X_test, Y_train, Y_test = train_and_test(X[indexes,:],
                                                  Y[indexes],
                                                  test_part)

#start =  np.append(np.random.normal(0,0.1,(2)),0)
start = np.random.rand(3)*2
teta = gradient_descent_circles_sigmoid(X_train, Y_train, 
                                        start, learning_rate, 
                                        points_in_cluster)
train_precision = (circle_prediction(X_train, teta)==Y_train).mean()
test_precision  = (circle_prediction(X_test, teta)==Y_test).mean()

print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(teta)

plt.show()
