import numpy as np
import time
import matplotlib.pyplot as plt

def streaming_svd(A_list, num_singular_values, forget_factor):

    U_list = []  # To store the left singular vectors after each batch
    D_list = []  # To store the singular values after each batch

    # initial data matrix
    A0 = A_list[0]  

    # QR decomposition
    Q, R = np.linalg.qr(A0)

    # SVD of R
    U0, D0, Vt0 = np.linalg.svd(R, full_matrices=False)

    # truncated left singular vectors
    U0 = Q @ U0 

    U_list.append(U0[:, :num_singular_values])
    D_list.append(D0[:num_singular_values])
    
    # Iterate over remaining batches
    for i in range(1, len(A_list)):

        Ai = A_list[i]  # New data batch
        
        # compute QR decomposition after concatenation of new data
        Ai_tilde = np.hstack((forget_factor * U_list[-1] @ np.diag(D_list[-1]), Ai))
        Q, R = np.linalg.qr(Ai_tilde)
        
        # compute SVD of R
        Ui_hat, Di_hat, Vt_hat = np.linalg.svd(R, full_matrices=False)
        
        # preserve the first K columns of Ui_hat
        Ui_tilde = Ui_hat[:, :num_singular_values]
        
        # obtain the updated left singular vectors
        Ui = Q @ Ui_tilde
        Di = Di_hat[:num_singular_values]
        
        # truncate to retain the top K values
        U_list.append(Ui)
        D_list.append(Di)
    
    return U_list, D_list

def measure_runtime(batch_sizes, num_trials, M, num_singular_values, forget_factor):
    
    runtimes = []
    
    for B in batch_sizes:
        trial_runtimes = []
        
        for _ in range(num_trials):

            # Generate synthetic data with the current batch size
            A_list = [np.random.randn(M, B) for _ in range(10)]
            
            # Measure the runtime for the current batch size
            start_time = time.time()
            streaming_svd(A_list, num_singular_values, forget_factor)
            end_time = time.time()
            
            trial_runtimes.append(end_time - start_time)
        
        # Average runtime over trials
        avg_runtime = np.mean(trial_runtimes)
        runtimes.append(avg_runtime)
    
    return runtimes

# Define parameters
batch_sizes = [10, 20, 50, 100, 200, 500]
num_trials = 10 
M = 100  
num_singular_values = 5  
forget_factor = 0.9

# Measure the runtimes
runtimes = measure_runtime(batch_sizes, num_trials, M, num_singular_values, forget_factor)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, runtimes, marker='o')
plt.xlabel('Batch Size (B)')
plt.ylabel('Average Runtime (seconds)')
plt.title('Average Runtime vs. Batch Size for Streaming SVD')
plt.grid(True)
plt.show()