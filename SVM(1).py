import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import elice_utils


# Functions linear_func1, linear_func2, and generate_data is used to generate random datapoints. You do not need to change this function
def linear_func1(x):
    l = len(x)
    return (3*x + 100 + 30 * np.random.randn(l))

def linear_func2(x):
    l = len(x)
    return (3*x - 100 + 30 * np.random.randn(l))

def generate_data(n):
    np.random.seed(32840091)

    x1_1 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_1 = linear_func1(x1_1)
    x1_2 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_2 = linear_func2(x1_2)
    y_1 = np.ones(int(0.5 * n))
    y_2 = -1 * np.ones(int(0.5 * n))

    x1 = np.concatenate((x1_1, x1_2))
    x2 = np.concatenate((x2_1, x2_2))
    y = np.concatenate((y_1, y_2))
    X = np.array(list(zip(x1, x2)))

    return (X, y)

def svm(X, y):
    n = len(y)

    # Notations P, q, G, h, A, b are the conventional notations for quadratic programming. 
    # You can find example usage of quadratic programming using cvxopt module in the webpage: courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    P = np.zeros(shape=(n, n));
    for i in range(0, n) :
        for j in range(0, n) :
            P[i][j] = 1 / 2 * y[i] *y[j] * (X[j][0] * X[i][0] + X[j][1] * X[i][1])
        
    P = matrix(P, tc='d')
    q = (-1) * np.ones(shape=(n,1))
    q = matrix(q, tc='d')
    G = (-1) * np.identity(n)
    G = matrix(G, tc='d')
    h = np.zeros(shape=(n,1))
    h = matrix(h, tc='d')
    A = np.zeros(shape=(1,n));
    for i in range(1, n):
        A[0][i] = y[i];
    A = matrix(A, tc='d')
    b = matrix(0.0, tc='d')
    opts={'show_progress' : False}
    sol = solvers.qp(P, q, G, h, A, b, options=opts)
    print(sol['x']);
    alpha = sol['x'];
    
    # Calculate optimal value of W. W should be returned in numpy array format.
    w1 = 0.0
    w2 = 0.0
    for i in range(0, n) :
        w1 += alpha[i] * y[i] * X[i][0]
        w2 += alpha[i] * y[i] * X[i][1]
        
    W = [w1, w2]
    threshold=1e-5
    # Find indices of the support vectors. It should be in python list format.
    sv = alpha > self.threshold
    support_vec_idx = np.arange(len(alp))[sv]
    print(support_vec_idx);
    

    m = support_vec_idx[0]
    # using one artiturarily selected support-vector index, calculate b, which is a scalar value.
    yn = y[m]; xn = X[m]
    b = 1

    return (W, b, support_vec_idx)

def draw_datasetonly(X, y):
    filename = "dataset.png"
    
    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555'); plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')    
    plt.scatter(X[:, 0], X[:, 1], c = ['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()

def draw(X, y, W, b, support_vec_idx):
    filename = "svm.png"

    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555'); plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')    
    plt.scatter(X[:, 0], X[:, 1], c = ['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    ls = np.linspace(-50, 50, 100)
    plt.plot(ls, [-1 * (W[0] * v + b) / W[1] for v in ls], lw=1.5, color='#CE5A57')
    plt.scatter(X[:, 0][support_vec_idx], X[:, 1][support_vec_idx], s=200, c='#CE5A57', edgecolor='none', marker='*')

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()
    
if __name__ == '__main__':
    X, y = generate_data(100)
    W, b, support_vec_idx = svm(X, y)
    draw_datasetonly(X, y)
    # draw(X, y, W, b, support_vec_idx) 