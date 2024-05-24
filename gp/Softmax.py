import numpy as np

def softmax(gene_outputs2):
    """
    Softmax function for given gene outputs.
    """
    cls_num = len(gene_outputs2)
    exp_strs = []
    nn = 1
    
    for kk in range(cls_num):
        num_genes = gene_outputs2[kk].shape[1]
        str_temp = "np.exp("
        for gg in range(num_genes):
            str_temp += f'+w[{nn - 1}]*x[{kk}][:, {gg}]'
            nn += 1
        str_temp += f'+w[{nn - 1}])'
        nn += 1
        exp_strs.append(str_temp)
    
    sum_exp_str = '+'.join(exp_strs)
    
    softmax_str = 'lambda x, w: np.column_stack(['
    for kk in range(cls_num):
        if kk < cls_num - 1:
            softmax_str += f'({exp_strs[kk]})/({sum_exp_str}), '
        else:
            softmax_str += f'({exp_strs[kk]})/({sum_exp_str})'
    softmax_str += '])'
    
    softmax_fun = eval(softmax_str)
    return softmax_fun
