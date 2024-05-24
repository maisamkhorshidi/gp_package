import numpy as np

def softmaxlossgrad(gene_outputs2):
    cls_num = len(gene_outputs2)
    num_genes = [gene_outputs2[kk].shape[1] for kk in range(cls_num)]
    
    # Softmax String
    exp_strs = []
    nn = 0
    for kk in range(cls_num):
        num_gene = gene_outputs2[kk].shape[1]
        str_temp = 'np.exp('
        for gg in range(num_gene):
            str_temp += f'+w[{nn}]*x[{kk}][:, {gg}]'
            nn += 1
        str_temp += f'+w[{nn}])'
        nn += 1
        exp_strs.append(str_temp)
    
    sum_exp_str = '+'.join(exp_strs)
    
    # Derivative of weights that appear in both numerator and denominator, and only in denominator
    str_st1 = ['(-1+(', '(']
    str_dv1 = [')./', ')./']
    str_en1 = [')', '']
    
    derivel_str = [[], []]
    nn = 0
    for i in range(2):
        for kk in range(cls_num):
            for gg in range(num_genes[kk]):
                derivel_str[i].append(f'x[{kk}][:, {gg}]*{str_st1[i]}{exp_strs[kk]}{str_dv1[i]}{sum_exp_str}{str_en1[i]}')
                nn += 1
            derivel_str[i].append(f'{str_st1[i]}{exp_strs[kk]}{str_dv1[i]}{sum_exp_str}{str_en1[i]}')
            nn += 1

    str_st1 = '['
    str_st2 = ['np.tile(y[:, ', 'np.tile(np.abs(1-y[:, ']
    str_md1 = [']), (1, ', ']), (1, ']
    str_md2 = ' , '
    str_en1 = ')'
    str_en2 = ']'

    ybin_str = []
    for i in range(2):
        str_temp1 = str_st1
        for kk in range(cls_num):
            if kk < cls_num - 1:
                str_temp1 += f'{str_st2[i]}{kk}{str_md1[i]}{num_genes[kk]+1}{str_en1}{str_md2}'
            elif kk == cls_num - 1:
                str_temp1 += f'{str_st2[i]}{kk}{str_md1[i]}{num_genes[kk]+1}{str_en1}{str_en2}'
        ybin_str.append(str_temp1)
    
    softmax_grad_str = f'lambda x, y, w: np.sum(({ybin_str[0]}*np.array({derivel_str[0]}))+({ybin_str[1]}*np.array({derivel_str[1]})), axis=0)'
    
    softmaxlossgradfun = eval(softmax_grad_str)
    return softmaxlossgradfun
