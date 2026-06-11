def get_leaf_summaries(model) -> list[str]:
    """
    Recursively extracts the 'summary' values of all nodes 
    that have no children.
    """
    # Base case: If there are no children, this is a leaf node.
    if not model.children:
        return [model.summary]
    
    # Recursive case: Gather summaries from all children.
    summaries = []
    for child in model.children:
        summaries.extend(get_leaf_summaries(child))
        
    return summaries

#Show all the nodes in the tree
def print_tree(node, level=0):
    if node is None:
        return
    print("  " * level + f"Node {node.node}: {node.label} - {node.summary}")
    for child in node.children:
        print_tree(child, level + 1)    