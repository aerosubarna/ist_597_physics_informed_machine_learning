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