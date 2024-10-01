import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tensorflow.keras.datasets import mnist

def get_full_mnist_data():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    return x_train

def get_mnist_data(batch_size):

    # load the MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()

    # preprocess: flatten the images and normalize
    x_train = x_train.reshape(-1, 28 * 28) / 255.0

    # segment the training data into required batch size
    A_list = []
    for i in range(0, len(x_train), batch_size):
        if i + batch_size <= len(x_train):
            A_list.append(x_train[i:i + batch_size])

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

    U = np.eye(m)
    V = np.eye(n)

    for _ in range(max_iterations):

        Q, R = qr_factorization_householder(B)

        B = R @ Q

        # Update Q_L and Q_R
        U = U @ Q  
        V = Q @ V 

        # Construct diagonal matrix from singular values (diagonal elements of B)
        sigma = np.diag(np.diagonal(B))

        # Convergence check
        if np.allclose(B - np.diag(np.diagonal(B)), 0, atol=tol):
            break

        if n > 100:
            break

        n += 1

    return U, sigma, V.T

def streaming_svd(A_list, num_singular_values, forget_factor):

    U_list = []  # To store the left singular vectors after each batch
    D_list = []  # To store the singular values after each batch

    # initial data matrix
    A0 = A_list[0]  

    # QR decomposition
    Q, R = qr_factorization_mgs(A0)

    # SVD of R
    U0, D0, Vt0 = svd_golub_reinsch(R)

    # truncated left singular vectors
    U0 = Q @ U0 

    U_list.append(U0[:, :num_singular_values])
    D_list.append(D0[:num_singular_values])
    
    # Iterate over remaining batches
    for i in range(1, len(A_list)):

        Ai = A_list[i]  # New data batch
        
        # compute QR decomposition after concatenation of new data
        Ai_tilde = np.hstack((forget_factor * U_list[-1] @ np.diag(D_list[-1]), Ai))
        Q, R = qr_factorization_mgs(Ai_tilde)
        
        # compute SVD of R
        Ui_hat, Di_hat, Vt_hat = svd_golub_reinsch(R)
        
        # preserve the first K columns of Ui_hat
        Ui_tilde = Ui_hat[:, :num_singular_values]
        
        # obtain the updated left singular vectors
        Ui = Q @ Ui_tilde
        Di = Di_hat[:num_singular_values]
        
        # truncate to retain the top K values
        U_list.append(Ui)
        D_list.append(Di)
    
    return U_list, D_list

def measure_runtime(batch_sizes, num_trials, num_singular_values, forget_factor):
    
    runtimes = []
    
    for b in batch_sizes:
        trial_runtimes = []
        
        for _ in range(num_trials):
            
            # get the MNIST data with the current batch size
            A_list = get_mnist_data(b)

            # measure the runtime for the current batch size
            start_time = time.time()
            streaming_svd(A_list, num_singular_values, forget_factor)
            end_time = time.time()
            
            trial_runtimes.append(end_time - start_time)
        
        # Average runtime over trials
        avg_runtime = np.mean(trial_runtimes)
        runtimes.append(avg_runtime)
    
    return runtimes

def plot_runtimes_batchsize(batch_sizes, runtimes):

    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes, runtimes, marker='o')
    plt.xlabel('Batch Size (B)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Average Runtime vs. Batch Size for Streaming SVD')
    plt.grid(True)
    plt.show()

def plot_svd(batch_size, num_singular_values, forget_factor):

    # get the MNIST data
    A_list = get_mnist_data(batch_size)

    # Perform streaming SVD
    U_list, _ = streaming_svd(A_list, num_singular_values, forget_factor)

    # Create animation
    fig = plt.figure(figsize=(8, 6))

    # Animation function
    def animate(i):
        plot_vectors(U_list, i)

    # Create the animation object
    ani = FuncAnimation(fig, animate, frames=len(A_list), repeat=False)

    # Save the animation as a GIF or MP4
    # To save as MP4, you can uncomment the MP4 writer
    ani.save('singular_vectors_evolution.gif', writer=PillowWriter(fps=2))

    # Alternatively, to save as MP4 (requires ffmpeg or other video writers)
    # ani.save('singular_vectors_evolution.mp4', writer='ffmpeg', fps=2)

    plt.show()

def plot_vectors(U_list, batch_num):
    plt.clf()
    U = U_list[batch_num]
    
    for i in range(3):  # Only plotting the first 3 singular vectors
        plt.plot(U[:, i], label=f'Singular Vector {i+1}')
    
    plt.title(f'Evolution of First 3 Singular Vectors - Batch {batch_num + 1}')
    plt.xlabel('Vector Components')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.ylim([-1, 1])
    plt.grid(True)

def compare_svd(batch_size, forget_factors, num_singular_values):

    A_full = get_full_mnist_data()

    # perform SVD using numpy
    U_full, D_full, Vt_full = np.linalg.svd(A_full, full_matrices=False)

    singular_vectors = {}

    # iterate through each forget factor

    for forget_factor in forget_factors:

        # get MNIST data in the required batch size
        A_list = get_mnist_data(batch_size)

        # perform streaming SVD
        U_list, _ = streaming_svd(A_list, num_singular_values, forget_factor)
        singular_vectors[forget_factor] = U_list

    # Plotting
    plt.figure(figsize=(15, 10))

    for i, (forget_factor, U) in enumerate(singular_vectors.items()):
        plt.subplot(len(singular_vectors), 1, i + 1)
        plt.title(f'Forget Factor: {forget_factor}')
        for j in range(min(3, U.shape[2])):  # Plotting the first 3 singular vectors
            plt.plot(U[:, -1][:, j], label=f'Singular Vector {j + 1} (Streaming SVD)')
        plt.plot(U_full[:, j], label=f'Singular Vector {j + 1} (np.linalg.svd)', linestyle='--')
        plt.legend()

    plt.tight_layout()
    plt.show()


# plot svd evolution with batch size 50
batch_size = 50
num_singular_values = 5  
forget_factor = 1.0

plot_svd(batch_size, num_singular_values, forget_factor)

# plot runtime vs batch size
# batch_sizes = [10, 20, 50, 100, 200, 500]
# num_singular_values = 5  
# forget_factor = 1.0
# num_trials = 10  

# runtimes = measure_runtime(batch_sizes, num_trials, num_singular_values, forget_factor)
# plot_runtimes_batchsize(batch_sizes, runtimes)

# compare svd generated by streaming svd and numpy svd for different forget factors
# batch_size = 50
# forget_factors = [0.5, 0.9, 0.99, 0.999]
# num_singular_values = 5

# compare_svd(batch_size, forget_factors, num_singular_values)

