import numpy as np
def fit(self):
    """
    Train the model using stochastic gradient descent.
    """
    X_train = self.params['xtrain']
    y_train = self.params['yonehot_train']
    X_val = self.params['xval']
    y_val = self.params['yonehot_val']
    batch_size = self.params['batch_size']
    random_seed = self.params['random_seed']
    buffer_size = self.params['buffer_size']
    shuffle = self.params['shuffle']
    epochs = self.params['epochs']
    patience = self.params['patience']
    optimizer_type = self.params['optimizer']
    verbose = self.params['verbose']
    
    # Optimizer's params
    if optimizer_type == 'sgd' or optimizer_type == 'sgdnm':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
    elif optimizer_type == 'adam':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
        m_dw = np.zeros_like(self.weight)
        m_db = np.zeros_like(self.bias)
        v_dw_max = np.zeros_like(self.weight)
        v_db_max = np.zeros_like(self.bias)
    elif optimizer_type == 'rmsprop':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
        m_dw = np.zeros_like(self.weight)
        m_db = np.zeros_like(self.bias)
        mean_dw = np.zeros_like(self.weight)
        mean_db = np.zeros_like(self.bias)
    elif optimizer_type == 'adamw':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
        m_dw = np.zeros_like(self.weight)
        m_db = np.zeros_like(self.bias)
        v_dw_max = np.zeros_like(self.weight)
        v_db_max = np.zeros_like(self.bias)
    elif optimizer_type == 'nadam':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
        m_dw = np.zeros_like(self.weight)
        m_db = np.zeros_like(self.bias)
    elif optimizer_type == 'adagrad':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
    elif optimizer_type == 'adadelta':
        v_dw = np.zeros_like(self.weight)
        v_db = np.zeros_like(self.bias)
    elif optimizer_type == 'ftrl':
        z_dw = np.zeros_like(self.weight)
        z_db = np.zeros_like(self.bias)
        n_dw = np.zeros_like(self.weight)
        n_db = np.zeros_like(self.bias)
    elif optimizer_type == 'adamax':
        m_dw = np.zeros_like(self.weight)
        m_db = np.zeros_like(self.bias)
        u_dw = np.zeros_like(self.weight)
        u_db = np.zeros_like(self.bias)
    # elif optimizer_type == 'lbgfs':
        
    # Initialize the best validation loss for early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_weight = np.copy(self.weight)  # Save best weights
    best_bias = np.copy(self.bias)  # Save best biases

    for epoch in range(epochs):
        # Batch the training data
        X_batches, y_batches, index_batches = self.batcher(X_train, y_train, batch_size=batch_size,\
                        shuffle=shuffle, random_seed=random_seed, buffer_size=buffer_size)
        
        # Training loop
        for X_batch, y_batch in zip(X_batches, y_batches):
            # Forward pass: compute probabilities
            prob = self.predict(X_batch, Training = True)
            
            # Backward pass: compute gradients
            dz = prob - y_batch
            dw = np.dot(dz.T, X_batch) / batch_size + self.regularization_gradient(batch_size)
            db = np.sum(dz, axis=0, keepdims=True).T / batch_size
            
            # Update weights and biases based on optimizer type
            if optimizer_type == 'sgd' or optimizer_type == 'sgdnm':
                v_dw, v_db = self.sgd(dw, db, v_dw, v_db, epoch)
            elif optimizer_type == 'adam':
                v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max = \
                    self.adam(dw, db, v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max, epoch)
            elif optimizer_type == 'rmsprop':
                v_dw, v_db, m_dw, m_db, mean_dw, mean_db = \
                    self.rmsprop(dw, db, v_dw, v_db, m_dw, m_db, mean_dw, mean_db, epoch)
            elif optimizer_type == 'adamw':
                v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max = \
                    self.adamw(dw, db, v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max, epoch)
            elif optimizer_type == 'nadam':
                v_dw, v_db, m_dw, m_db = self.nadam(dw, db, v_dw, v_db, m_dw, m_db, epoch)
            elif optimizer_type == 'adagrad':
                v_dw, v_db = self.adagrad(dw, db, v_dw, v_db, epoch)
            elif optimizer_type == 'adadelta':
                v_dw, v_db = self.adadelta(dw, db, v_dw, v_db, epoch)
            elif optimizer_type == 'ftrl':
                z_dw, z_db, n_dw, n_db = self.ftrl(dw, db, z_dw, z_db, n_dw, n_db, epoch)
            elif optimizer_type == 'adamax':
                m_dw, m_db, u_dw, u_db = self.adamax(dw, db, m_dw, m_db, u_dw, u_db, epoch)
            # elif optimizer_type == 'lbgfs':
                
        prob = self.predict(X_batches, index_batch = index_batches, Training = False)
        # Calculate the loss and add regularization
        loss_value = self.loss(prob, y_train) + self.regularization_loss()
        
        # Validation loss for early stopping (if validation data is provided)
        if X_val is not None and y_val is not None:
            val_prob = self.predict(X_val, batch_size = batch_size, Training = False)
            val_loss = self.loss(val_prob, y_val) + self.regularization_loss()
        else:
            val_loss = None
            
        if val_loss is None and loss_value < best_loss:
            best_loss = loss_value
            best_weight = np.copy(self.weight)  # Save current best weights
            best_bias = np.copy(self.bias)  # Save current best biases
            patience_counter = 0
        elif val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            best_weight = np.copy(self.weight)  # Save current best weights
            best_bias = np.copy(self.bias)  # Save current best biases
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose != 0 and (epoch % verbose) == 0:
                    print(f"Early stopping at epoch {epoch}, Loss: {loss_value:.6f}, Val Loss: {val_loss:.6f}, Patience: {patience_counter}")
                break
        
        if verbose != 0 and (epoch % verbose) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.6f}, Val Loss: {val_loss:.6f}, Patience: {patience_counter}")

    # Always restore the best weights and biases, even if early stopping wasn't triggered
    if verbose != 0:
        print("Restoring the best weights and biases from the best epoch.")
    self.weight = np.copy(best_weight)
    self.bias = np.copy(best_bias)