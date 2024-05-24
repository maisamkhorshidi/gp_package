from .evalfitness_ord import evalfitness_ord
from .evalfitness_par import evalfitness_par

def evalfitness(gp):
    """Evaluate the fitness of individuals stored in the GP structure."""
    if gp.config['runcontrol']['parallel']['enable'] and gp.config['runcontrol']['parallel']['ok']:
        gp = evalfitness_par(gp)
    else:
        gp = evalfitness_ord(gp)
    return gp
