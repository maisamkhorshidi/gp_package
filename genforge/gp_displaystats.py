# from .gpmodelvars import gpmodelvars

def gp_displaystats(gp):
    """Displays run stats periodically.
    
    Args:
        gp (dict): The genetic programming structure.
    """
    # Only display info if required
    if not gp.config['runcontrol']['verbose'] or gp.config['runcontrol']['quiet'] or \
            (gp.state['generation']) % gp.config['runcontrol']['verbose'] != 0:
        return

    gen = gp.state['generation']

    # Multiple independent run count display
    if gp.config['runcontrol']['runs'] > 1 and gen == 0:
        print(f"Run {gp['state']['run']} of {gp['runcontrol']['runs']}")

    print(f"Generation:         {gen}")
    print(f"Best fitness:       {gp.state['best']['fitness']['ensemble']['train'][-1]:.4f}")
    print(f"Mean fitness:       {gp.state['mean_fitness']['ensemble']['train'][-1]:.4f}")
    print(f"Best complexity:    {gp.state['best']['complexity']['ensemble'][-1]:.4f}")
    print(f"Stall Generation:   {gp.state['stallgen']}")
    print(f"Time Elapsed:       {gp.state['TimeElapsed'][-1]:.4f} sec")
    # # Display inputs in "best training" individual, if enabled.
    # if gp['runcontrol']['showBestInputs']:
    #     numx, hitvec = gpmodelvars(gp, 'best')
    #     inputs = ''.join([f"x{num} " for num in numx])
    #     print(f"Inputs in best individual: {inputs}")

    # # Display inputs in "best validation" individual, if enabled.
    # if gp['runcontrol']['showValBestInputs'] and 'xval' in gp['userdata'] and \
    #         gp['userdata']['xval'] and 'valbest' in gp['results']:
    #     numx, hitvec = gpmodelvars(gp, 'valbest')
    #     inputs = ''.join([f"x{num} " for num in numx])
    #     print(f"Inputs in best validation individual: {inputs}")

    print(' ')
