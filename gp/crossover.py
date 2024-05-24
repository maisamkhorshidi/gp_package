from .utils import picknode, extract

def crossover(mum, dad, gp):
    """Sub-tree crossover of encoded tree expressions to produce 2 new ones."""
    # Select random crossover nodes in mum and dad expressions
    m_position = picknode(mum, 0, gp)
    d_position = picknode(dad, 0, gp)

    # Extract main and subtree expressions
    m_main, m_sub = extract(m_position, mum)
    d_main, d_sub = extract(d_position, dad)

    # Combine to form 2 new GP trees
    daughter = m_main.replace('$', d_sub)
    son = d_main.replace('$', m_sub)

    return son, daughter
