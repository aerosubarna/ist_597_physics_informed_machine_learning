import numpy as np
import time

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

def bidiagonalize_householder(A):

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

    U, B, V = bidiagonalize_householder(A)

    # truncate U and B
    if m > n:
        U = U[:, :n]
        B = B[:n, :n]

    Q_R = np.eye(n)
    Q_L = np.eye(n)

    for _ in range(max_iterations):

        Qi, Ri = qr_factorization_mgs(B)

        B = np.matmul(Ri, Qi)

        # Update Q_L and Q_R
        Q_L = np.matmul(Q_L, Qi)
        Q_R = np.matmul(Qi, Q_R)

        # Convergence check
        if np.allclose(B - np.diag(np.diagonal(B)), 0, atol=tol):
            break

        # get U and V
        U = np.matmul(U, Q_L)
        V = np.matmul(V, Q_R.T)

        # get singular values
        singular_values = np.abs(np.diag(B))

        # sort singular values in descending order
        idx = np.argsort(singular_values)[::-1]
        singular_values = singular_values[idx]

        # sort U and V
        U = U[:, idx]
        V = V[:, idx]

    return U, singular_values, V.T

def streaming_svd(A_list, num_modes, forget_factor):

    modes_list = []  # To store the left singular vectors after each batch
    singular_values_list = []  # To store the singular values after each batch

    # initial data matrix
    A0 = A_list[0]

    # QR decomposition
    Q, R = qr_factorization_mgs(A0)

    # SVD of R
    U0, singular_values, _ = svd_golub_reinsch(R)

    # truncated left singular vectors
    truncated_U0 = U0[:, :num_modes]
    modes = np.matmul(Q, truncated_U0)

    # truncated singular values
    singular_values = singular_values[:num_modes]

    # store the results
    modes_list.append(modes)
    singular_values_list.append(singular_values)

    # Iterate over remaining batches
    for i in range(1, len(A_list)):

        # new data batch
        Ai = A_list[i]

        # QR decomposition after concatenation of new data
        prev_mode = modes_list[-1]
        prev_singular_values = singular_values_list[-1]
        prev_info = forget_factor * np.matmul(prev_mode, np.diag(prev_singular_values))
        Ai_concat = np.concatenate((prev_info, Ai), axis=1)

        # QR decomposition
        Q, R = qr_factorization_mgs(Ai_concat)

        # SVD of R
        Ui, singular_values, _ = svd_golub_reinsch(R)

        # truncated left singular vectors
        truncated_Ui = Ui[:, :num_modes]
        modes = np.matmul(Q, truncated_Ui)

        # truncated singular values
        singular_values = singular_values[:num_modes]

        # store the results
        modes_list.append(modes)
        singular_values_list.append(singular_values)

    return modes_list, singular_values_list