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


def get_leaf_labels(model) -> list[str]:
    """Recursively extract the ``label`` of every leaf node (no children)."""
    if not model.children:
        return [model.label]

    labels: list[str] = []
    for child in model.children:
        labels.extend(get_leaf_labels(child))

    return labels


def build_leaf_ancestry(model, ancestors=None) -> dict:
    """Map each leaf ``label`` to the list of ancestor labels (root first).

    The returned mapping is ``{leaf_label: [root_label, ..., parent_label]}`` and
    is used to derive risk_factor/risk_channel from a leaf sub-scenario.
    """
    ancestors = ancestors or []
    if not model.children:
        return {model.label: list(ancestors)}

    mapping: dict = {}
    child_ancestors = [*ancestors, model.label]
    for child in model.children:
        mapping.update(build_leaf_ancestry(child, child_ancestors))

    return mapping

#Show all the nodes in the tree
def print_tree(node, level=0):
    if node is None:
        return
    print("  " * level + f"Node {node.node}: {node.label} - {node.summary}")
    for child in node.children:
        print_tree(child, level + 1)    