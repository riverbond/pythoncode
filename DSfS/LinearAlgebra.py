
# coding: utf-8

# ## Linear Algebra

# In[3]:

# adding and subtracting vectors
import math

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


# #### NOTE: Using lists as vectors is great for exposition but TERRIBLE for performance!
# #### In production code use numpy's array class

# ## Matrices

# In[4]:

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


# In[5]:

# Let's make a n x n identity matrix

def is_diagonal(i, j):
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)
# print identity_matrix


# In[ ]:



