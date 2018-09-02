# Convex Optimization, is massive. A starting place might be: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
# For a starting place for constraint optimization in general, you could also check out http://www.mit.edu/~dimitrib/Constrained-Opt.pdf

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):  #_init is a constructor, out of standards we pass self first
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    #train
    def fit(self, data):
        # The fit method will be used to train our SVM. This will be the optimization step.
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}
        # Begin building an optimization dictionary as opt_dict, which is going to contain any optimization values.
        # As we step down our w vector, we'll test that vector in our constraint function, finding the largest b, if any,
        # that will satisfy the equation, and then we'll store all of that data in our optimization dictionary. The dictionary
        # will be { ||w|| : [w,b] }. When we're all done optimizing, we'll choose the values of w and b for whichever one in
        # the dictionary has the lowest key value (which is ||w||).

        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data=None

        step_sizes = [self.max_feature_value*0.1, self.max_feature_value*0.01, # starts getting very high cost after this.
                      self.max_feature_value * 0.001]
        # What we're doing here is setting some sizes per step that we want to make.
        # For our first pass, we'll take big steps (10%). Once we find the minimum with these steps,
        # we're going to step down to a 1% step size to continue finding the minimum here. So on.

        # Next, we're going to set some variables that will help us make steps with b (used to make larger steps than we use for w,
        # since we care far more about w precision than b), and keep track of the latest optimal value

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        #STEPPING
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized=True
                    print('Optimized a step')
                else:
                    w=w-step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2

    def predict(self, features):
        # sign(x.w+b)
        #The predict method will predict the value of a new featureset once we've trained the classifier,
        # which is just the sign(x.w+b) once we know what w and b are.
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        # if the classification isn't zero, and we have visualization on, we graph
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        else:
            print('featureset', features, 'is on the decision boundary')
        return classification

    def visualize(self):
        # scattering known featuresets
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x,w,b,v):
            # v=w.x + b
            return (-w[0]*x - b + v)/w[1]
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # w.x + b = 1
        # pos sv hyperplane
        psv1 =  hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 =  hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1,psv2], "k")
        # w.x + b = -1
        # negative sv hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], "k")
        # w.x + b = 0
        # decision
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], "g--")

        plt.show()


data_dict = {-1: np.array([[1, 7],[2, 8],[3, 8], ]), 1: np.array([[5, 1],[6, -1],[7, 3], ])}
svm = Support_Vector_Machine()
svm.fit(data=data_dict)
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()