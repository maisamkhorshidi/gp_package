def extract(position, expr):
    """Extract a subtree from an encoded GP tree expression."""
    
    left_brackets = 0
    right_brackets = 0
    
    for i in range(position, len(expr)):
        if expr[i] == '(':
            left_brackets += 1
        elif expr[i] == ')':
            right_brackets += 1
        
        if left_brackets == right_brackets:
            end_position = i
            break
    else:
        end_position = len(expr)
    
    subtree = expr[position:end_position + 1]
    main_tree = expr[:position] + '$' + expr[end_position + 1:]
    
    return main_tree, subtree
