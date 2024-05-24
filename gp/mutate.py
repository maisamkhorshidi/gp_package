import numpy as np
from .treegen import treegen
from .utils import picknode, extract

def mutate(expr, gp, jj=1):
    """Mutate an encoded symbolic tree expression."""
    rnum = np.random.rand()

    while True:  # Loop until a mutation occurs
        # Ordinary subtree mutation
        if rnum <= gp.config['operators']['mutation']['cumsum_mutate_par'][0]:
            # First get the string position of the node
            position = picknode(expr, 0, gp)
            # Remove the logical subtree for the selected node
            # and return the main tree with '$' replacing the extracted subtree
            maintree = extract(position, expr)
            # Generate a new tree to replace the old
            # Pick a depth between 1 and gp.treedef.max_mutate_depth;
            depth = np.random.randint(1, gp.config['treedef']['max_mutate_depth'][jj] + 1)
            newtree = treegen(gp, jj, depth)
            # Replace the '$' with the new tree
            expr_mut = maintree.replace('$', newtree)
            break

        # Substitute terminals
        elif rnum <= gp.config['operators']['mutation']['cumsum_mutate_par'][1]:
            position = picknode(expr, 4, gp)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = extract(position, expr)  # Extract the constant
                terminal = f"x{np.random.choice(gp.config['userdata']['xindex'][jj])}"
                expr_mut = maintree.replace('$', terminal)
                break

        # Constant Gaussian perturbation
        elif rnum <= gp.config['operators']['mutation']['cumsum_mutate_par'][2]:
            position = picknode(expr, 3, gp)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree, subtree = extract(position, expr)
                # Process constant
                oldconst = float(subtree[1:-1])
                newconst = oldconst + (np.random.randn() * gp.config['operators']['mutation']['gaussian']['std_dev'])
                # Use appropriate string formatting for new constant
                if newconst == int(newconst):
                    newtree = f"[{int(newconst)}]"
                else:
                    newtree = f"[{newconst:.6f}]"  # Adjust format as needed
                # Replace '$' with the new tree
                expr_mut = maintree.replace('$', newtree)
                break

        # Make constant zero
        elif rnum <= gp.config['operators']['mutation']['cumsum_mutate_par'][3]:
            position = picknode(expr, 3, gp)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = extract(position, expr)  # Extract the constant
                expr_mut = maintree.replace('$', '[0]')  # Replace it with zero
                break

        # Randomize constant
        elif rnum <= gp.config['operators']['mutation']['cumsum_mutate_par'][4]:
            position = picknode(expr, 3, gp)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = extract(position, expr)  # Extract the constant
                # Generate a constant in the range
                const = np.random.rand() * (gp.config['nodes']['const']['range'][1] - gp.config['nodes']['const']['range'][0]) + gp.config['nodes']['const']['range'][0]
                # Convert the constant to square bracketed string form
                if const == int(const):
                    arg = f"[{int(const)}]"
                else:
                    arg = f"[{const:.6f}]"  # Adjust format as needed
                expr_mut = maintree.replace('$', arg)  # Insert new constant
                break

        # Set a constant to 1
        elif rnum <= gp.config['operators']['mutation']['cumsum_mutate_par'][5]:
            position = picknode(expr, 3, gp)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = extract(position, expr)  # Extract the constant
                expr_mut = maintree.replace('$', '[1]')  # Replace it with one
                break

    return expr_mut
