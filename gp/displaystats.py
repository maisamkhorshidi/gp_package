from .gpmodelvars import gpmodelvars

def displaystats(gp):
    """Displays run stats periodically.
    
    Args:
        gp (dict): The genetic programming structure.
    """
    # Only display info if required
    if not gp['runcontrol']['verbose'] or gp['runcontrol']['quiet'] or \
            (gp['state']['count'] - 1) % gp['runcontrol']['verbose'] != 0:
        return

    gen = gp['state']['count'] - 1

    # Multiple independent run count display
    if gp['runcontrol']['runs'] > 1 and gen == 0:
        print(f"Run {gp['state']['run']} of {gp['runcontrol']['runs']}")

    print(f"Generation {gen}")
    print(f"Best fitness:    {gp['results']['best']['fitness']}")
    print(f"Mean fitness:    {gp['state']['meanfitness']}")

    if gp['fitness']['complexityMeasure']:
        print(f"Best complexity: {gp['results']['best']['complexity']}")
    else:
        print(f"Best node count: {gp['results']['best']['nodecount']}")

    # Display inputs in "best training" individual, if enabled.
    if gp['runcontrol']['showBestInputs']:
        numx, hitvec = gpmodelvars(gp, 'best')
        inputs = ''.join([f"x{num} " for num in numx])
        print(f"Inputs in best individual: {inputs}")

    # Display inputs in "best validation" individual, if enabled.
    if gp['runcontrol']['showValBestInputs'] and 'xval' in gp['userdata'] and \
            gp['userdata']['xval'] and 'valbest' in gp['results']:
        numx, hitvec = gpmodelvars(gp, 'valbest')
        inputs = ''.join([f"x{num} " for num in numx])
        print(f"Inputs in best validation individual: {inputs}")

    print(' ')
