import numpy as np
import re

def gpcheck(gp):
    if 'nodes' not in gp or 'functions' not in gp['nodes'] or 'name' not in gp['nodes']['functions']:
        raise ValueError("Function node filenames must be defined as a list in gp['nodes']['functions']['name']")

    for i in range(len(gp['nodes']['functions']['name'])):
        if not callable(gp['nodes']['functions']['name'][i]):
            raise ValueError(f'The function node "{gp["nodes"]["functions"]["name"][i]}" cannot be found or is not callable. Check your config file.')

    gp['nodes']['const']['format'] = f'%0.{gp["nodes"]["const"]["num_dec_places"]}f'

    if 'xtrain' not in gp['userdata'] or 'ytrain' not in gp['userdata']:
        raise ValueError("The fields gp['userdata']['xtrain'] and gp['userdata']['ytrain'] must be populated with data.")

    if gp['runcontrol']['num_pop'] != len(gp['userdata']['xindex']):
        raise ValueError("gp['userdata']['xindex'] must be a list equal to the number of populations.")

    if gp['runcontrol']['num_pop'] > 1:
        required_keys = ['xindex', 'xtrain', 'xtest', 'ytrain', 'ytest']
        for key in required_keys:
            if gp['runcontrol']['num_pop'] != len(gp['userdata'][key]):
                raise ValueError(f'gp["userdata"]["{key}"] must be a list equal to the number of populations.')

    if gp['runcontrol']['num_pop'] > 1 and len(gp['runcontrol']['pop_size']) == 1:
        gp['runcontrol']['pop_size'] = [gp['runcontrol']['pop_size'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['selection']['tournament']['size']) == 1:
        gp['selection']['tournament']['size'] = [gp['selection']['tournament']['size'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['selection']['elite_fraction']) == 1:
        gp['selection']['elite_fraction'] = [gp['selection']['elite_fraction'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['treedef']['max_depth']) == 1:
        gp['treedef']['max_depth'] = [gp['treedef']['max_depth'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['genes']['max_genes']) == 1:
        gp['genes']['max_genes'] = [gp['genes']['max_genes'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['nodes']['inputs']['num_inp']) == 1:
        gp['nodes']['inputs']['num_inp'] = [gp['nodes']['inputs']['num_inp'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['treedef']['max_mutate_depth']) == 1:
        gp['treedef']['max_mutate_depth'] = [gp['treedef']['max_mutate_depth'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['nodes']['const']['p_ERC']) == 1:
        gp['nodes']['const']['p_ERC'] = [gp['nodes']['const']['p_ERC'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['nodes']['const']['p_int']) == 1:
        gp['nodes']['const']['p_int'] = [gp['nodes']['const']['p_int'][0]] * gp['runcontrol']['num_pop']

    if gp['runcontrol']['num_pop'] > 1 and len(gp['treedef']['max_nodes']) == 1:
        gp['treedef']['max_nodes'] = [gp['treedef']['max_nodes'][0]] * gp['runcontrol']['num_pop']

    gp['runcontrol']['num_pop'] = int(np.ceil(gp['runcontrol']['num_pop']))
    gp['runcontrol']['num_gen'] = int(np.ceil(gp['runcontrol']['num_gen']))
    gp['runcontrol']['runs'] = int(np.ceil(gp['runcontrol']['runs']))
    gp['runcontrol']['verbose'] = int(np.ceil(gp['runcontrol']['verbose']))

    if gp['runcontrol']['pop_size'] < 2:
        raise ValueError("Population size must be at least 2. Set gp['runcontrol']['pop_size'] >= 2")

    if gp['runcontrol']['num_gen'] < 1:
        raise ValueError("GPTIPS must run for at least one generation. Set gp['runcontrol']['num_gen'] >= 1")

    if gp['runcontrol']['runs'] < 1:
        raise ValueError("GPTIPS must have at least 1 independent run. Set gp['runcontrol']['runs'] >= 1")

    if gp['runcontrol']['verbose'] < 0:
        raise ValueError("The verbose parameter cannot be negative. Set gp['runcontrol']['verbose'] >= 0")

    if min(gp['treedef']['max_depth']) < 1:
        raise ValueError("gp['treedef']['max_depth'] must be >= 1")

    if min(gp['treedef']['max_mutate_depth']) < 1:
        raise ValueError("gp['treedef']['max_mutate_depth'] must be >= 1")

    if min(gp['treedef']['max_nodes']) < 1:
        raise ValueError("gp['treedef']['max_nodes'] must be >= 1")

    if gp['treedef']['build_method'] < 1 or gp['treedef']['build_method'] > 3:
        raise ValueError("gp['treedef']['build_method'] must be set as one of the following: 1 = full 2 = grow 3 = ramped half and half")

    if min(gp['selection']['elite_fraction']) < 0 or max(gp['selection']['elite_fraction']) > 1:
        raise ValueError("gp['selection']['elite_fraction'] must be >= 0 and <= 1")

    if sum([gp['operators']['mutation']['p_mutate'], gp['operators']['crossover']['p_cross'], gp['operators']['directrepro']['p_direct']]) != 1:
        raise ValueError("gp crossover, mutation and direct reproduction probabilities must add up to 1")

    if len(gp['operators']['mutation']['mutate_par']) != 6 or not isinstance(gp['operators']['mutation']['mutate_par'], (list, tuple)):
        raise ValueError("gp['operators']['mutation']['mutate_par'] must be a list or tuple of length 6")

    if sum(gp['operators']['mutation']['mutate_par']) != 1:
        raise ValueError("The probabilities in gp['operators']['mutation']['mutate_par'] must add up to 1")

    if gp['fitness']['minimisation']:
        gp['fitness']['minimisation'] = True

    if gp['fitness']['complexityMeasure'] not in [0, 1]:
        raise ValueError("gp['fitness']['complexityMeasure'] must be either 0 or 1")

    if min(gp['nodes']['const']['p_ERC']) < 0 or max(gp['nodes']['const']['p_ERC']) > 1:
        raise ValueError("gp['nodes']['const']['p_ERC'] must be >= 0 and <= 1")

    if min(gp['nodes']['const']['p_int']) < 0 or max(gp['nodes']['const']['p_int']) > 1:
        raise ValueError("gp['nodes']['const']['p_int'] must be >= 0 and <= 1")

    if gp['runcontrol']['savefreq'] < 0:
        raise ValueError("gp['runcontrol']['savefreq'] must be >= 0")

    if min(gp['nodes']['inputs']['num_inp']) < 0:
        raise ValueError("gp['nodes']['inputs']['num_inp'] must be >= 0")

    if gp['nodes']['const']['num_dec_places'] <= 0:
        raise ValueError("gp['nodes']['const']['num_dec_places'] must be > 0")

    if len(gp['nodes']['const']['range']) != 2:
        raise ValueError("gp['nodes']['const']['range'] must be a 2 element list or tuple")

    if gp['nodes']['const']['range'][0] > gp['nodes']['const']['range'][1]:
        raise ValueError("Invalid ERC range set in gp['nodes']['const']['range']")

    gp['nodes']['const']['rangesize'] = gp['nodes']['const']['range'][1] - gp['nodes']['const']['range'][0]

    for i in range(len(gp['nodes']['inputs']['num_inp'])):
        if gp['nodes']['inputs']['num_inp'][i] == 0 and gp['nodes']['const']['p_ERC'][i] > 0:
            gp['nodes']['const']['p_ERC'][i] = 1

    if len(gp['nodes']['functions']['name']) != len(gp['nodes']['functions']['active']):
        raise ValueError("There must be the same number of entries in gp['nodes']['functions']['name'] and gp['nodes']['functions']['active']. Check your config file.")

    if len(gp['nodes']['inputs']['names']) > gp['nodes']['inputs']['num_inp']:
        raise ValueError("The supplied input variables 'name' list in gp['nodes']['inputs']['names'] contains more entries than input variables.")

    nameInds = [i for i, name in enumerate(gp['nodes']['inputs']['names']) if name]
    numDefined = len(nameInds)
    numUnique = len(set(gp['nodes']['inputs']['names'][i] for i in nameInds))

    if numDefined != numUnique:
        raise ValueError("Entries of gp['nodes']['inputs']['names'] must be unique.")

    gp['nodes']['inputs']['namesPlain'] = gp['nodes']['inputs']['names']
    gp['nodes']['output']['namePlain'] = gp['nodes']['output']['name']

    for i in nameInds:
        gp['nodes']['inputs']['namesPlain'][i] = re.sub('<.*?>', '', gp['nodes']['inputs']['namesPlain'][i])
        gp['nodes']['inputs']['namesPlain'][i] = re.sub('\s+', '', gp['nodes']['inputs']['namesPlain'][i])

    gp['nodes']['output']['namePlain'] = re.sub('<.*?>', '', gp['nodes']['output']['namePlain'])
    gp['nodes']['output']['namePlain'] = re.sub('\s+', '', gp['nodes']['output']['namePlain'])
