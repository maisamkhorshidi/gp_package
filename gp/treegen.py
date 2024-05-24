import numpy as np
from .extract import extract
import re

def treegen(gp, jj=1, fixed_depth=0):
    """Generate a new encoded GP tree expression."""
    
    max_depth = gp['treedef']['max_depth'][jj] if not fixed_depth else max(fixed_depth, 1)
    build_method = gp['treedef']['build_method'] if not fixed_depth else 1
    p_ERC = gp['nodes']['const']['p_ERC'][jj]
    p_int = gp['nodes']['const']['p_int'][jj]
    num_inp = gp['nodes']['inputs']['num_inp'][jj]
    xindex = gp['userdata']['xindex'][jj]
    range_ = gp['nodes']['const']['range']
    rangesize = gp['nodes']['const']['rangesize']
    fi = gp['nodes']['const']['format']
    ERC_generated = False

    afid_argt0 = gp['nodes']['functions']['afid_argt0']
    afid_areq0 = gp['nodes']['functions']['afid_areq0']
    arity_argt0 = gp['nodes']['functions']['arity_argt0']
    fun_lengthargt0 = gp['nodes']['functions']['fun_lengthargt0']
    fun_lengthareq0 = gp['nodes']['functions']['fun_lengthareq0']

    if build_method == 3:
        max_depth = int(np.ceil(np.random.rand() * gp['treedef']['max_depth'][jj]))
        build_method = int(np.ceil(np.random.rand() * 2))
    
    treestr = '($)'
    
    while True:
        node_token_ind = treestr.find('$')
        if node_token_ind == -1:
            break
        
        node_token = node_token_ind
        treestr = treestr[:node_token] + '@' + treestr[node_token + 1:]
        
        left_seg = treestr[:node_token]
        num_open = left_seg.count('(')
        num_closed = left_seg.count(')')
        depth = num_open - num_closed
        
        if depth == 1:
            node_type = 1 if max_depth != 1 else 2
        elif build_method == 1 and depth < max_depth:
            node_type = 1
        elif depth == max_depth:
            node_type = 2
        else:
            node_type = int(np.ceil(np.random.rand() * 2))
        
        if node_type == 1:
            fun_choice = int(np.ceil(np.random.rand() * fun_lengthargt0))
            fun_name = afid_argt0[fun_choice - 1]
            num_fun_args = arity_argt0[fun_choice - 1]
            
            fun_rep_str = f"{fun_name}($"
            for j in range(1, num_fun_args):
                fun_rep_str += ",$"
            fun_rep_str += ")"
            
            treestr = treestr.replace('@', fun_rep_str, 1)
        
        elif node_type == 2:
            if fun_lengthareq0 and (np.random.rand() >= 0.5 or (num_inp == 0 and p_ERC == 0)):
                fun_choice = int(np.ceil(np.random.rand() * fun_lengthareq0))
                fun_name = afid_areq0[fun_choice - 1]
                fun_rep_str = f"{fun_name}()"
                treestr = treestr.replace('@', fun_rep_str, 1)
            else:
                term = np.random.rand() >= p_ERC
                if term:
                    inp_choice = xindex[int(np.ceil(np.random.rand() * num_inp)) - 1]
                    treestr = treestr.replace('@', f"x{inp_choice}", 1)
                else:
                    treestr = treestr.replace('@', '?', 1)
                    ERC_generated = True

    if ERC_generated:
        const_ind = [m.start() for m in re.finditer(r'\?', treestr)]
        num_constants = len(const_ind)
        const = np.random.rand(num_constants) * rangesize + range_[0]

        if p_int:
            ints = np.random.rand(num_constants) <= p_int
            const[ints] = np.round(const[ints])
        else:
            ints = [False] * num_constants

        for k in range(num_constants):
            if ints[k]:
                arg = f"[{int(const[k])}]"
            else:
                arg = f"[{fi % const[k]}]"
            
            main_tree = extract(const_ind[0], treestr)
            treestr = main_tree.replace('$', arg, 1)
            const_ind = [m.start() for m in re.finditer(r'\?', treestr)]

    treestr = treestr[1:-1]
    return treestr
