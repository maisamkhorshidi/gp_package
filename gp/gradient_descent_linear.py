import numpy as np
import matplotlib.pyplot as plt
from .logloss import logloss
from .linearensemble import linearensemble
from .linearensemblelossgrad import linearensemblelossgrad

def gradient_descent_linear(gene_output_str2, gene_outputs_val2, y_binary_tr, y_binary_val, 
                            initial_learning_rate, epsilon, lambda_L1, lambda_L2, 
                            tolgrad, max_iteration):
    """
    Gradient Descent Algorithm to find gene weights.
    """
    linear_ensemble_loss_grad_fun = linearensemblelossgrad(gene_output_str2)
    cls_num = len(gene_output_str2)
    num_genes = [gene_output_str2[kk].shape[1] for kk in range(cls_num)]
    num_weights = sum(num_genes) + cls_num
    num_data_tr = gene_output_str2[0].shape[0]
    num_data_val = gene_outputs_val2[0].shape[0]

    # Softmax of validation data
    linear_ensemble_fun = linearensemble(gene_outputs_val2)

    # Elastic Net Regularization
    l1_derivative = lambda lambda1, w: lambda1 * np.sign(w)
    l2_derivative = lambda lambda2, w: lambda2 * 2 * w

    # Gradient of loss function + l1 and l2 derivatives
    grad_fun = lambda x, y, w, l1, l2: 1.0 / num_data_tr * (linear_ensemble_loss_grad_fun(x, y, w) + l1_derivative(l1, w) + l2_derivative(l2, w))

    # AdaGrad specific initialization
    grad_square_sum = np.zeros(num_weights)  # Sum of squares of gradients

    # Gradient Descent
    w_init = np.random.rand(num_weights)  # Initialize weights randomly
    w_temp = w_init
    w_mem = np.full((max_iteration, num_weights), np.nan)
    grad_mem = np.full((max_iteration, num_weights), np.nan)
    itr_val = []
    loss_val1 = []

    plt.figure(1)
    for itr in range(max_iteration):
        w_mem[itr, :] = w_temp
        grad = grad_fun(gene_output_str2, y_binary_tr, w_temp, lambda_L1, lambda_L2)
        grad_mem[itr, :] = grad

        # Loss
        loss = logloss(linear_ensemble_fun(gene_outputs_val2, w_temp), y_binary_val)

        # Plot
        itr_val.append(itr)
        loss_val1.append(loss)
        plt.plot(itr_val, loss_val1, linewidth=2)
        plt.xlim([0, max_iteration])
        plt.xlabel('Iteration')
        plt.ylabel('Log-Loss')
        plt.draw()
        plt.pause(0.001)

        # Stopping criterion based on the change in the loss of validation set
        if itr > 1:
            crit = loss_val1[-2] - loss_val1[-1]
        else:
            crit = 1

        if np.any(np.isnan(w_temp)) or crit < 0 or loss < 0:
            w_final = w_mem[itr - 1, :]
            grad_final = grad_mem[itr - 1, :]
            loss_val_final = loss_val1[itr - 1]
            break
        elif np.all(np.abs(grad) <= tolgrad) or itr == max_iteration - 1:
            w_final = w_temp
            grad_final = grad
            loss_val_final = loss
        else:
            # AdaGrad Update
            grad_square_sum += grad ** 2
            adjusted_learning_rate = initial_learning_rate / np.sqrt(grad_square_sum + epsilon)
            # Update weights
            w_temp -= adjusted_learning_rate * grad

    w = w_final
    grad = grad_final
    loss_val = loss_val_final

    return w, grad, loss_val
