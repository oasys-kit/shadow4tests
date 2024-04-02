# https://github.com/numpy/numpy/issues/14193

import numpy as np
import time


# here I define a custom function to generate the random vector using Cholesky decompostion.
def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)


# mean = np.random.rand(1881)
# a = np.random.rand(1881, 1881)
#
# cov = a @ a.T / mean.size
#
# np.random.seed(1234)
# t0 = time.time()
# b1 = np.random.multivariate_normal(mean, cov)
# print("old: ", time.time()-t0)
# print(b1)
#
# np.random.seed(1234)
# t0 = time.time()
# b2 = chol_sample(mean, cov)
# print("new: ", time.time()-t0)
# print(b2)

N = 10000
mean = np.random.rand(2) - 0.5
a = np.random.rand(2, 2) - 0.5
cov = a @ a.T / mean.size

X1 = np.zeros(N)
Y1 = np.zeros(N)
X2 = np.zeros(N)
Y2 = np.zeros(N)

t0 = time.time()
np.random.seed(1234)
for i in range(N):
    b1 = np.random.multivariate_normal(mean, cov)
    X1[i] = b1[0]
    Y1[i] = b1[1]
    # print(b1)
print("old: ", time.time()-t0)


t0 = time.time()
np.random.seed(1234)
for i in range(N):
    b2 = chol_sample(mean, cov)
    X2[i] = b2[0]
    Y2[i] = b2[1]
    # print(b2)
print("new: ", time.time()-t0)


# t0 = time.time()
# print("old: ", time.time()-t0)
# print(b1)
#
# np.random.seed(1234)
# b2 = chol_sample(mean, cov)
# print("new: ", time.time()-t0)
# print(b2)
from srxraylib.plot.gol import plot_scatter
plot_scatter(X1, Y1, show=0)
plot_scatter(X2, Y2)

print("meanX: ", X1.mean(), X2.mean(), mean[0])
print("meanY: ", Y1.mean(), Y2.mean(), mean[1])
print("sdX: ", X1.std(), X2.std(), cov[0,0]**(1/2))
print("sdY: ", Y1.std(), Y2.std(), cov[1,1]**(1/2))
print(cov)
