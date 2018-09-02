from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

def create_dataset(hm, variance, step=2, correlation=False):
    #hm = how many points in dataset
    #This will dictate how much each point can vary from the previous point. The more variance, the less-tight the data will be.
    #This will be how far to step on average per point, defaulting to 2.
    #This will be either False, pos, or neg to indicate that we want no correlation, positive correlation, or negative correlation.
    val = 1
    ys = []
    for i in range (hm):
        y= val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation =='neg':
            val-=step
    xs= [ i for i in range (len(ys))]
    print(ys)
    print(xs)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

style.use('ggplot')

#xs = np.array([1,2,3,4,5], dtype=np.float64)
#ys = np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

# The standard way to check for errors is by using squared errors.
# The distance between the regression line's y values, and the data's y values is the error, then we square that.
# The line's squared error is either a mean or a sum of this, we'll simply sum it.

# The equation is essentially 1 minus the division of the squared error of the regression line and the squared error of the mean y line.
# Thus, the goal is to have the r squared value, otherwise called the coefficient of determination, as close to 1 as possible.

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig) * (ys_line-ys_orig))

def coeff_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    #print(ys_orig)
    #print(y_mean_line)
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 10, 2, correlation='pos')
# Assumption:  Less variance should result in higher r-squared/coefficient of determination, higher variance = lower r squared.
# No correlation should be even lower, and actually quite close to zero, unless we get a crazy random permutation that actually has correlation anyway
m, b = best_fit_slope_and_intercept(xs,ys)
#reg_line = []
#for x in xs:
#    reg_line.append((m*x) + b)
reg_line = [(m*x) + b for x in xs]
r_squared = coeff_of_determination(ys, reg_line)

print(r_squared)

#r_squared is a good parameter if you want to find exact values
#but if you just want to see direction its not reallt important

#GRAPHING
#pred_x = 7
#pred_y = m*pred_x + b
#
plt.scatter(xs, ys, color='#003F72', label='Data')
#plt.scatter(pred_x, pred_y, color='green', label='Predicted Data')
plt.plot(xs, reg_line, label='Regression line')
plt.legend(loc=4)
plt.show()