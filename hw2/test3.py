import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

def plot_svd(batch_size, num_batches, M, num_singular_values, forget_factor):

    # Generate synthetic data
    A_list = [np.random.randn(M, batch_size) for _ in range(num_batches)]

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

# Parameters
batch_size = 50 
num_batches = 10 
M = 100 
num_singular_values = 3
forget_factor = 1.0 

# Plot the evolution of singular vectors
plot_svd(batch_size, num_batches, M, num_singular_values, forget_factor)

