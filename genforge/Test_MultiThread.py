import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from SoftmaxModel import SoftmaxModel
import numpy as np
import joblib as jb
gp = jb.load("C:\\Users\\Maisam\\OneDrive - UTS\\Career - New\\Research\\ISI Paper 14 - GP Multi-pop\\Python Package\\aa.pkl")
# Function to process each individual
def process_individual(id_pop, id_ind, optim, id_opt1, best_fit, best_optim, gp):
    gene_out_tr = gp.individuals['gene_output']['train'][id_pop][id_ind]
    gene_out_val = gp.individuals['gene_output']['validation'][id_pop][id_ind]
    gene_out_ts = gp.individuals['gene_output']['test'][id_pop][id_ind]
    params = {
        'xtrain': gene_out_tr,
        'ytrain': gp.userdata['ytrain'],
        'yonehot_train': gp.userdata['ybinarytrain'],
        'xval': gene_out_val,
        'yval': gp.userdata['yval'],
        'yonehot_val': gp.userdata['ybinaryval'],
        'num_class': gp.config['runcontrol']['num_class'],
        'optimizer': optim[id_opt1],  # optimizer_type,
        'optimizer_param': {},  # optimizer_param,
        'initializer': 'glorot_uniform',
        'regularization': gp.config['softmax']['regularization'][id_pop],
        'regularization_rate': gp.config['softmax']['regularization_rate'][id_pop],
        'batch_size': gp.config['softmax']['batch_size'][id_pop],
        'epochs': 1000,
        'patience': gp.config['softmax']['patience'][id_pop],
        'random_seed': None,
        'buffer_size': None,
        'shuffle': False,
        'verbose': 0,
    }

    nn = SoftmaxModel.compiler(**params)
    nn.fit()
    
    # Predictions and loss calculations
    prob_tr = nn.predict(gene_out_tr)
    loss_trn = nn.loss(prob_tr, gp.userdata['ybinarytrain'])
    loss_tro = gp.individuals['fitness']['isolated']['train'][id_ind, id_pop]

    prob_val = nn.predict(gene_out_val)
    loss_valn = nn.loss(prob_val, gp.userdata['ybinaryval'])
    loss_valo = gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop]

    prob_ts = nn.predict(gene_out_ts)
    loss_tsn = nn.loss(prob_ts, gp.userdata['ybinarytest'])
    loss_tso = gp.individuals['fitness']['isolated']['test'][id_ind, id_pop]

    # Update best optimizer and fitness if needed
    if loss_valn < best_fit[id_ind, id_pop]:
        best_fit[id_ind, id_pop] = loss_valn
        best_optim[id_pop][id_ind] = optim[id_opt1]

    result = (id_pop, id_ind, loss_trn, loss_tro, loss_valn, loss_valo, loss_tsn, loss_tso, optim[id_opt1])
    return result


# Main multi-threading logic
def main():
    t1 = time.time()
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    optim = ['sgd', 'sgdnm', 'adam', 'rmsprop', 'adamw', 'nadam', 'adagrad', 'adadelta', 'ftrl', 'adamax']
    id_opt1 = 3  # Selected optimizer index

    best_optim = [[None for _ in range(pop_size)] for _ in range(num_pop)]
    best_fit = np.full((pop_size, num_pop), np.inf)

    # Create a ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor() as executor:
        futures = []
        for id_pop in range(num_pop):
            for id_ind in range(pop_size):
                futures.append(executor.submit(process_individual, id_pop, id_ind, optim, id_opt1, best_fit, best_optim, gp))
        
        # Collect results
        for future in as_completed(futures):
            id_pop, id_ind, loss_trn, loss_tro, loss_valn, loss_valo, loss_tsn, loss_tso, optimizer = future.result()

            print('')
            print(f'Train Fitness, Model: {loss_trn:.6f}, Old: {loss_tro:.6f}, Optim: {optimizer}')
            print(f'Valid Fitness, Model: {loss_valn:.6f}, Old: {loss_valo:.6f}, Optim: {optimizer}')
            print(f'Test  Fitness, Model: {loss_tsn:.6f}, Old: {loss_tso:.6f}, Optim: {optimizer}')

    # Best fitness comparison
    bb = best_fit - gp.individuals['fitness']['isolated']['validation']
    aa = np.minimum(0, bb)
    print(aa)

    t2 = time.time()
    et = t2 - t1
    print(f'Elapsed Time: {et:.4f}')


# Run the main function
if __name__ == "__main__":
    main()
