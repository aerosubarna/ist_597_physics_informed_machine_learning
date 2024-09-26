import numpy as np

def inner_product(a, b):

    # check if lenght of a and b are the same
    if np.size(a) != np.size(b):
        raise ValueError("a and b must have the same length.")
    
    # calculate the inner product
    inner_product = 0
    for i in range(np.size(a)):
        inner_product += a[i] * b[i]

    return inner_product

def normalize(a):

    # calculate the norm
    norm = np.sqrt(inner_product(a, a))

    if norm == 0:
        raise ValueError("The norm of the vector is zero.")

    # normalize
    return a / norm

def projection(a, b):

    # check if lenght of a and b are the same
    if np.size(a) != np.size(b):
        raise ValueError("a and b must have the same length.")

    # calculate the projection
    projection = (inner_product(a, b)/inner_product(b, b)) * b

    return projection

def outer_product(a, b):
    
        # check if lenght of a and b are the same
        if np.size(a) != np.size(b):
            raise ValueError("a and b must have the same length.")
    
        # calculate the outer product
        outer_product = np.zeros((np.size(a), np.size(b)))
        for i in range(np.size(a)):
            for j in range(np.size(b)):
                outer_product[i, j] = a[i] * b[j]
    
        return outer_product

def gram_schmidt(A):

    m = A.shape[0]
    n = A.shape[1]

    # check if the vectors are linearly independent (are a valid basis)
    if np.linalg.matrix_rank(A) != n:
        raise ValueError("The vectors are not linearly independent.")
    
    # apply the Gram-Schmidt process
    Q = A.copy().astype(float)

    for i in range(n):
        for j in range(i):
            Q[:, i] -= projection(Q[:, i], Q[:, j])

    # normalize
    for i in range(n):
        Q[:, i] = normalize(Q[:, i])

    return Q

def modified_gram_schmidt(A):

    m = A.shape[0]
    n = A.shape[1]

    # check if the vectors are linearly independent (are a valid basis)
    if np.linalg.matrix_rank(A) != n:
        raise ValueError("The vectors are not linearly independent.")
    
    # apply the Gram-Schmidt process
    Q = A.copy().astype(float)

    for i in range(n):

        # normalize
        Q[:, i] = normalize(Q[:, i])

        for j in range(i+1, n):
            Q[:, j] -= projection(Q[:, j], Q[:, i])

    return Q

def qr_factorization_gs(A):

    Q = gram_schmidt(A)

    # get R
    # we have, A = QR; so, R = Q^T A)
    R = Q.T @ A

    return np.round(Q, decimals=2), np.round(R, decimals=2)

def qr_factorization_mgs(A):

    Q = modified_gram_schmidt(A)

    # get R
    # we have, A = QR; so, R = Q^T A)
    R = Q.T @ A

    return np.round(Q, decimals=2), np.round(R, decimals=2)

def householder(a):

    # check if all the terms are already zero
    if np.all(a[1:] == 0):
        return np.eye(len(a))
    
    # calculate the norm
    alpha = np.sqrt(inner_product(a, a))

    # the target vector form
    e = np.zeros(len(a))
    e[0] = 1

    v = a + alpha * e
    u = normalize(v)

    H = np.eye(len(a)) - 2 * outer_product(u, u)

    return H

def qr_factorization_householder(A):

    m = A.shape[0]
    n = A.shape[1]

    householder_list = []

    for i in range(n):
        H_prime = householder(A[i:, i])
        H = np.eye(m)
        H[i:, i:] = H_prime
        householder_list.append(H)
    
    # get Q
    Q = np.eye(m)
    for i in range(n):
        Q = Q @ householder_list[i]

    # get R
    # we have, A = QR; so, R = Q^T A)
    R = Q.T @ A

    return np.round(Q, decimals=2), np.round(R, decimals=2)

def bidiagonalize_housholder(A):

    m = A.shape[0]
    n = A.shape[1]

    U = np.eye(m)
    V = np.eye(n)

    for i in range(n):

        # zeroing out the column
        H_prime = householder(A[i:, i])
        H = np.eye(m)
        H[i:, i:] = H_prime

        # apply the transformation
        A = H @ A

        # Update U
        U = U @ H.T  

        if i < n-2:

            # zeroing out the row
            H_prime_ = householder(A[i, i+1:])
            H_ = np.eye(n)
            H_[i+1:, i+1:] = H_prime_

            # apply the transformation
            A = A @ H_

            # Update V
            V = V @ H_

    return U, A, V
    
def svd_golub_reinsch(A, max_iterations = 1000, tol = 1e-10):

    m = A.shape[0]
    n = A.shape[1]

    # check if m >= n
    if m < n:
        raise ValueError("m must be greater than or equal to n.")

    _, B, _ = bidiagonalize_housholder(A)

    # extract the square (n x n) part of B (diagonal and superdiagonal)
    B_square = B[:n, :n]
    B = B_square

    Q_R = np.eye(n)
    Q_L = np.eye(n)

    for _ in range(max_iterations):

        Qi, Ri = qr_factorization_householder(B)

        B = Ri @ Qi

        # Update Q_L and Q_R
        Q_L = Q_L @ Qi  
        Q_R = Qi @ Q_R 

        # Construct diagonal matrix from singular values (diagonal elements of B)
        sigma = np.diag(np.diagonal(B))

        # Convergence check
        if np.allclose(B - np.diag(np.diagonal(B)), 0, atol=tol):
            break

    return Q_R, sigma, Q_L.T

def streaming_svd(A_list, num_singular_values, forget_factor):

    U_list = []  # To store the left singular vectors after each batch
    D_list = []  # To store the singular values after each batch

    # initial data matrix
    A0 = A_list[0] 

    # QR decomposition
    Q, R = qr_factorization_mgs(A0)

    # SVD of R
    U_, D0, _ = svd_golub_reinsch(R)

    # truncate to retain the top K values
    U_ = U_[:, :num_singular_values]
    Q = Q[:, :num_singular_values]  

    # truncated left singular vectors
    U0 = Q @ U_
    D0 = D0[:num_singular_values] 

    U_list.append(U0)
    D_list.append(D0)

    # Iterate over remaining batches
    for i in range(1, len(A_list)):

        Ai = A_list[i]  # New data batch

        
        # compute QR decomposition after concatenation of new data
        prev_info = forget_factor * (U_list[-1] @ D_list[-1])
        Ai_tilde = np.hstack((prev_info, Ai))
        Q, R = qr_factorization_mgs(Ai_tilde)
        
        # compute SVD of R
        Ui_hat, Di_hat, _ = svd_golub_reinsch(R)
        
        # preserve the first K columns of Ui_hat
        Ui_tilde = Ui_hat[:, :num_singular_values]
        
        # obtain the updated left singular vectors
        Ui = Q @ Ui_tilde
        Di = Di_hat[:num_singular_values]
        
        # truncate to retain the top K values
        U_list.append(Ui)
        D_list.append(Di)
    
    return U_list, D_list

# test of streaming_svd

A_list = [np.random.randn(100, 20) for _ in range(10)]
K = 5 
forget_factor = 0.9

U_list, D_list = streaming_svd(A_list, K, forget_factor)


# # test

# A = np.random.rand(3, 3)
# print(A)

# U, sigma, V = golub_reinsch(A)

# print(U)
# print(sigma)
# print(V.T)