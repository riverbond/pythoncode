
# coding: utf-8

# In[1]:

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from __future__ import division


# In[2]:

def vector_add(v, w):
    return [v_i + w_i 
            for v_i, w_i in zip(v, w)]

def vector_subtract(v, w):
    return [v_i - w_i 
            for v_i, w_i in zip(v, w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    return sum(v_i * w_i
              for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def distance(v, w):
    return magnitude(vector_subtract(v, w))

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in 1st row
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j]       # jth element of each row A_i
           for A_i in A] # for each row of A_i

# Here is a function we can use to generate matrices
def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j)           # given i, create a list
            for j in range(num_cols)] # [entry_fn(i, 0), ...]
            for i in range(num_rows)] # create one list for each i

def is_diagonal(i, j):
    return 1 if i == j else 0

def mean(x):
    return sum(x) / len(x)

def median(v):
    n        = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    
    if n % 2 == 1:
        return sorted_v[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2
    
def quantile(x, p):
    p_index = int(p* len(x))
    return sorted(x)[p_index]

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems()
           if count == max_count]

def data_range(x):
    return max(x) - min(x) 

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    n          = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

def covariance(x, y):
    n = len(x) # should be the case that len(x) equals len(y).
    return dot(de_mean(x), de_mean(y)) / (n - 1) 

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero
    
def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    "returns the probability that a uniform randvar is <= x"
    if x < 0:   return 0 # uniform random is never less than 0
    elif x < 1: return x # e.g. P(X <= 0.4) = 0.4
    else:       return 1 # uniform random is always less than 1
    
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2/ sigma ** 2) / sqrt_two_pi * sigma) 

def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find an approximate inverse using binary search"""
    
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
        
    low_z = -10.0                   # normal_cdf(-10) is approximately zero
    hi_z  = 10.0                    # normal_cdf(10) is approximately zero
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # consider the midpoint
        mid_p = normal_cdf(mid_z)   # and the cdf's value there
        if mid_p < p:
            # midpoint too low? search above it
            low_z = mid_z
        elif mid_p > p:
            # midpoint too high? search below it
            hi_z = mid_z
        else:
            break
    return mid_z

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n)) 

