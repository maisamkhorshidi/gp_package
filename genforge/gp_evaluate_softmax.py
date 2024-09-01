# t1 = time.time()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.initializers import HeNormal, GlorotUniform, RandomNormal
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Disable interactive logging
tf.keras.utils.disable_interactive_logging()
# # Disable progress bars
# class NoProgressBar(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         pas
# t2 = time.time()
# print(f'Load Time: {t2-t1}')
#     def on_train_batch_end(self, batch, logs=None):
#         pass
    
#     def on_test_batch_end(self, batch, logs=None):
#         pass
    
def gp_evaluate_softmax(gp, id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts):
    
    """
    Train a one-layer softmax function using Keras and evaluate the probabilities and losses.
    
    Args:
    gp (object): The GP object containing individuals' gene outputs and training data.
    learning_rate (float): Learning rate for the optimizer.
    optimizer_type (str): The type of optimizer to use ('sgd', 'adam', 'rmsprop').
    initializer (str): The type of weight initializer ('he_normal', 'glorot_uniform', 'random_normal').
    regularization (str): Type of regularization ('l2', 'l1', None).
    regularization_rate (float): Regularization strength.
    batch_size (int): The size of batches for training.
    epochs (int): Number of epochs for training.
    momentum (float): Momentum parameter for SGD optimizer.
    decay (float): Learning rate decay.
    clipnorm (float): Norm for gradient clipping.
    clipvalue (float): Value for gradient clipping.
    
    Returns:
    tuple: A tuple containing the evaluated probabilities and the loss value.
           (probabilities, loss)
    """
    # Extract the gene output and labels from the GP object
    # gene_out_tr = gp.individuals['gene_output']['train'][id_pop][id_ind]
    # gene_out_val = gp.individuals['gene_output']['validation'][id_pop][id_ind]
    # gene_out_ts = gp.individuals['gene_output']['test'][id_pop][id_ind]
    ytr = gp.userdata['ytrain']
    yval = gp.userdata['yval']
    yts = gp.userdata['ytest']
    num_class = gp.config['runcontrol']['num_class']
    
    # softmax parameters
    learning_rate = gp.config['softmax']['learning_rate'][id_pop]
    optimizer_type = gp.config['softmax']['optimizer_type'][id_pop]
    initializer = gp.config['softmax']['initializer'][id_pop]
    regularization = gp.config['softmax']['regularization'][id_pop]
    regularization_rate = gp.config['softmax']['regularization_rate'][id_pop]
    batch_size = gp.config['softmax']['batch_size'][id_pop]
    epochs = gp.config['softmax']['epochs'][id_pop]
    momentum = gp.config['softmax']['momentum'][id_pop]
    decay = gp.config['softmax']['decay'][id_pop]
    clipnorm = gp.config['softmax']['clipnorm'][id_pop]
    clipvalue = gp.config['softmax']['learning_rate'][id_pop]
    
    if gene_out_val is not None:
        patience = gp.config['softmax']['patience'][id_pop]
        
    
    # Choose the weight initializer
    if initializer == 'he_normal':
        init = HeNormal()
    elif initializer == 'random_normal':
        init = RandomNormal()
    elif initializer == 'glorot_uniform':
        init = GlorotUniform()  # Default
    
    # Choose the regularization
    if regularization == 'l2':
        reg = l2(regularization_rate)
    elif regularization == 'l1':
        reg = l1(regularization_rate)
    else:
        reg = None  # No regularization
    # t3 = time.time()
    # print(f'Init Time: {t3-t2}')
    # Define the one-layer model with softmax activation
    model = Sequential([
        Dense(num_class, input_shape=(gene_out_tr.shape[1],), 
              activation='softmax', 
              kernel_initializer=init,
              kernel_regularizer=reg)
    ])
    
    # Choose the optimizer
    if optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, clipnorm=clipnorm, clipvalue=clipvalue)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate, decay=decay, clipnorm=clipnorm, clipvalue=clipvalue)
    elif optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm, clipvalue=clipvalue)  # Default
    
    # Compile the model with the chosen optimizer
    model.compile(optimizer=optimizer,
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    # t4 = time.time()
    # print(f'Compile Time: {t4-t3}')
    # Define EarlyStopping callback if validation data is available
    if gene_out_val is not None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Train the model with early stopping if validation data is avalable
    if gene_out_val is not None:
        model.fit(gene_out_tr, ytr, validation_data=(gene_out_val, yval),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stopping])#, NoProgressBar()])
    else:
        model.fit(gene_out_tr, ytr, epochs=epochs, batch_size=batch_size, verbose=0)#, callbacks=[NoProgressBar()])
    
    # t44 = time.time()
    # print(f'Fit Time: {t44-t4}')
    # Predict probabilities on the training data
    prob_tr = model.predict(gene_out_tr)
    
    # t5 = time.time()
    # print(f'Predict Time: {t5-t44}')
    if gene_out_val is not None:
        prob_val = model.predict(gene_out_val)
    else:
        prob_val = None
    
    if gene_out_ts is not None:
        prob_ts = model.predict(gene_out_ts)
    else:
        prob_ts = None
    
    # Calculate the loss on the training data
    loss_tr, _ = model.evaluate(gene_out_tr, ytr, verbose=0)
    # t6 = time.time()
    # print(f'Evaluate Time: {t6-t5}')
    
    if gene_out_val is not None:
        loss_val, _ = model.evaluate(gene_out_val, yval, verbose=0)
    else:
        loss_val = None
    
    if gene_out_ts is not None:
        loss_ts, _ = model.evaluate(gene_out_ts, yts, verbose=0)
    else:
        loss_ts = None
    
    # calculate predicted labels
    yp_tr = np.argmax(prob_tr, axis=1)
    
    if gene_out_val is not None:
        yp_val = np.argmax(prob_val, axis=1)
    else:
        yp_val = None
    
    if gene_out_ts is not None:
        yp_ts = np.argmax(prob_ts, axis=1)
    else:
        yp_ts = None
    
    # Access the dense layer and extract weights and biases
    dense_layer = model.layers[0]
    weights, biases = dense_layer.get_weights()
    biases = biases.reshape(1, -1)  # Reshape to (1, output_dim)
    weights_with_biases = np.concatenate([weights, biases], axis=0)  # Add biases as the last row
    # print(f'Loss Train: {loss_tr}')
    # print(f'Loss Val: {loss_val}')
    # print(f'Loss Val: {loss_ts}')
    # Assign to results list
    results = [prob_tr, prob_val, prob_ts, loss_tr, loss_val, loss_ts, yp_tr, yp_val, yp_ts, weights_with_biases]
        
        
    return results
        
    # Example usage:
    # Assuming gp is already defined and populated with gene_output['train'], ytrain data, and num_class
    # probabilities, loss = evaluate_softmax(gp)
    # print(f"Probabilities: {probabilities}")
    # print(f"Loss: {loss}")
