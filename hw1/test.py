import numpy as np
from io import StringIO
from scipy.io import mmread

import time
import statistics

def is_diagonally_dominant(A):
    m, n = np.shape(A)
    for i in range(n):
        sum_non_diag = 0
        for j in range(n):
            if i != j:
                sum_non_diag += abs(A[i,j])
        if sum_non_diag > abs(A[i,i]):
            return False
    return True

def is_symmetric(A):
    m, n = np.shape(A)
    if m != n:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] != A[j,i]:
                return False
    return True

def is_symmetric_positive_definite(A):
    m, n = np.shape(A)

    # check if A is symmetric
    if not is_symmetric(A):
        return False
    
    # check if all eigenvalues are positive
    eigvals = np.linalg.eigvals(A)
    if np.all(eigvals > 0):
        return True
    else:
        return False

def jacobi(A, b, tolerance=1e-6, max_iterations=10000):

    # find the shape of matrix A
    m, n = np.shape(A)

    # check if A is a square matrix
    if m != n:
        raise ValueError('Matrix A must be square')
    
    # check if A is diagonally dominant
    if not is_diagonally_dominant(A):
        print('Warning: Matrix A is not diagonally dominant. May not converge.')

    # initial guess
    x = np.zeros(n)

    # new guess
    x_new = np.zeros(n)

    # initial error (set to large value to ensure loop runs at least once)
    error = np.inf

    # iteration counter
    iteration = 0

    while True:

        for i in range(n):
            summation_term = 0.0
            for j in range(n):
                if i != j:
                    summation_term += A[i, j] * x[j]
            x_new[i] = (b[i] - summation_term) / A[i, i]

        # check for convergence
        error = np.linalg.norm(x - x_new, 2)

        # check for NaN and inf errors and stop if found
        if np.isnan(error) or np.isinf(error):
            raise ValueError(f"Diverged. Error became very high at iteration {iteration}.")
        
        # print('Iteration: ', iteration, 'Error: ', error)
        x = np.copy(x_new)
        iteration += 1

        if iteration == max_iterations:
            raise ValueError("Maximum iterations reached without convergence.")
        
        if error<tolerance:
            # print("Converged!!!")
            break

    return x_new

def gauss_siedel(A, b, tolerance=1e-6, max_iterations=10000):

    # find the shape of matrix A
    m, n = np.shape(A)

    # check if A is a square matrix
    if m != n:
        raise ValueError('Matrix A must be square.')
    
    # check if A is diagonally dominant
    if not is_diagonally_dominant(A):
        #check if A is symmetric positive definite
        if not is_symmetric_positive_definite(A):
            print('Warning: Matrix A is not diagonally dominant or symmetric positive definite. May not converge.')

    # initial guess
    x = np.zeros(n)

    # iteration counter
    iteration = 0

    # loop until error is less than tolerance
    while True:

        x_old = np.copy(x) # to check for convergence

        for i in range(n):
            summation_term = 0
            for j in range(n):
                if i != j:
                    summation_term += A[i,j]*x[j]
            x[i] = (b[i] - summation_term)/A[i,i]

        # check for convergence
        error = np.linalg.norm(x - x_old, 2)

        # check for NaN and inf error and stop if found
        if np.isnan(error) or np.isinf(error):
            raise ValueError(f"Diverged. Error became very high at iteration {iteration}.")
        
        # print('Iteration: ', iteration, 'Error: ', error)
        iteration += 1

        if iteration == max_iterations:
            raise ValueError("Maximum iterations reached without convergence.")
        
        if error<tolerance:
            # print("Converged!!!")
            break
        
    return x

def cholesky(A, b):

   # find the shape of matrix A 
    m, n = np.shape(A)

    # check if A is symmetric positive definite
    if not is_symmetric_positive_definite(A):
        raise ValueError('Matrix A must be symmetric positive definite.')

    # initialize L
    L = np.zeros((n,n))

    # find the Cholesky decomposition
    for i in range(n):
        for j in range(i+1):
            if i == j:
                summation_term = 0
                for k in range(i):
                    summation_term += L[i,k]**2
                L[i,i] = np.sqrt(A[i,i] - summation_term)
            else:
                summation_term = 0
                for k in range(j):
                    summation_term += L[j,k]*L[i,k]
                L[i,j] = (A[i,j] - summation_term)/L[j,j]
    
    # now solving for x
        # Ax = b; LL'x = b
        # Let y = L'x; Ly = b; L'x = y

    # solving Ly = b (using forward substitution)
    y = np.zeros(n)
    for i in range(n):
        summation_term = 0
        for j in range(i):
            summation_term += L[i,j]*y[j]
        y[i] = (b[i] - summation_term)/L[i,i]

    # solving L'x = y (using backward substitution)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        summation_term = 0
        for j in range(i+1, n):
            summation_term += L[j,i]*x[j]
        x[i] = (y[i] - summation_term)/L[i,i]

    return x

mcca = mmread('/home/subarna/ist_597_phyics_informed_machine_learning/hw1/q2.mtx')
A = np.real(mcca.toarray())

# trim the matrix to 100x100
# A = A[:200, :200]
print(is_diagonally_dominant(A))
# A = 0.5*(A + A.T)

# diagonally dominant matrix of size 5x5
# A = np.array([[10, 1, 0, 2, 0],
#               [1, 10, 2, 0, 1],
#               [0, 2, 10, 1, 0],
#               [2, 0, 1, 10, 1],
#               [0, 1, 0, 1, 10]], dtype=float)

# symmetric positive definite matrix of size 5x5
# A = np.array([[10, 1, 0, 2, 0],
#               [1, 10, 2, 0, 1],
#               [0, 2, 10, 1, 0],
#               [2, 0, 1, 10, 1],
#               [0, 1, 0, 1, 10]], dtype=float)

# x_true = np.random.rand(A.shape[0])
# b = A.dot(x_true)

# x = gauss_siedel(A, b)
# print("Solution: ", x)

def statistical_test(A, solver, num_samples=100):
    
    # Generate random true x values
    x_true_list = [np.random.rand(A.shape[0]) for _ in range(num_samples)]
    run_times = []

    for i in range(len(x_true_list)):

        x_true = x_true_list[i]
        print("Sample: ", i)
        b = A.dot(x_true)
        start_time = time.time()
        x = solver(A, b)
        end_time = time.time()
        
        run_time = end_time - start_time
        run_times.append(run_time)
    
    # statistical analysis
    mean_run_time = statistics.mean(run_times)
    std_dev_run_time = statistics.stdev(run_times)
    median_run_time = statistics.median(run_times)

    return mean_run_time, std_dev_run_time, median_run_time

# mean_cholesky, std_dev_cholesky, median_cholesky = statistical_test(A, cholesky)
# mean_jacobi, std_dev_jacobi, median_jacobi = statistical_test(A, jacobi)
# mean_gs, std_dev_gs, median_gs = statistical_test(A, gauss_siedel)

# print("Mean run time for Cholesky: ", mean_cholesky)
# print("Standard deviation of run time for Cholesky: ", std_dev_cholesky)
# print("Median run time for Cholesky: ", median_cholesky)

# print("Mean run time for Jacobi: ", mean_jacobi)
# print("Standard deviation of run time for Jacobi: ", std_dev_jacobi)
# print("Median run time for Jacobi: ", median_jacobi)

# print("Mean run time for Gauss-Siedel: ", mean_gs)
# print("Standard deviation of run time for Gauss-Siedel: ", std_dev_gs)
# print("Median run time for Gauss-Siedel: ", median_gs)



