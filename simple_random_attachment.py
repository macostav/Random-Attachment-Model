import random
import networkx as nx
import matplotlib.pyplot as plt

"""
Simple random attachment model. The target graph has 4 vertices, 3 edges, and we make 3 copies. The graph looks
something like this:

1 — 2
|   |
3   4

All strong bonds except for 3 - 4.
"""

def base_label(node):
    """
    For a given vertex, it will give only the base vertex label.
    For instance, if given '3_1' -> '3'.

    :param node: vertex
    :return: vertex type 
    """
    return node.split("_")[0]

def can_connect(comp1, comp2, edges):
    """
    Check if it is possible to connect comp1 and comp2.

    :param comp1: set of vertices in the first component sampled
    :param comp2: set of vertices in second component sampled
    :param edges: list of edges in original graph
    :return: True if there is at least one edge connecting two components, False otherwise
    """
    # Write components only in terms of vertex types
    base1 = {base_label(x) for x in comp1}
    base2 = {base_label(x) for x in comp2}

    if base1.isdisjoint(base2): # Only try to join if the components are disjoint
        # Look for edges connecting the two components
        for u, v in edges:
            if (str(u) in base1 and str(v) in base2) or (str(v) in base1 and str(u) in base2):
                return True
        return False
    else:
        return False

def draw_components(components, all_nodes, step, pos):
    """
    Visualize components as a graph. Note that in the image, all components
    are in a cluster are connected but in reality this is not the case.

    :param components: list of sets, each set is a component
    :param all_nodes: all the vertices
    :param step: current step of the simulation
    :param pos: layout position
    """
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    # connect nodes inside each component (to visualize clusters)
    for comp in components:
        comp_list = list(comp)
        for i in range(len(comp_list)):
            for j in range(i+1, len(comp_list)):
                G.add_edge(comp_list[i], comp_list[j])

    # assign colors by component
    node_to_comp = {}
    for comp_id, comp in enumerate(components):
        for v in comp:
            node_to_comp[v] = comp_id
    colors = [node_to_comp.get(v, -1) for v in G.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos=pos, with_labels=True, node_color=colors,
            node_size=600, cmap=plt.cm.tab10, edgecolors='black')
    plt.title(f"Step {step}  (components: {len(components)})")
    plt.savefig(f"step_{step}.png")

if __name__ == "__main__":
    # Initialize original graph
    vertices = [1,2,3,4]
    edges = [(1,3), (1,2), (2,4)]
    n = 3 # Number of copies

    # Create graph components
    all_nodes = [f"{v}_{copy}" for copy in range(n) for v in vertices] # Label each copy
    all_components = [{v} for v in all_nodes]  # singleton components

    # Preserve structure for the visualization
    G_nodes = nx.Graph()
    G_nodes.add_nodes_from(all_nodes)
    pos = nx.spring_layout(G_nodes, seed=42)

    step = 0
    draw_components(all_components, all_nodes, step, pos)
   
    # Start loop with computations
    while len(all_components) > n: # Loop should end when we have all the components joined and have n copies formed
        component_1, component_2 = random.sample(all_components, 2)

        if can_connect(component_1, component_2, edges):
            new_component = component_1.union(component_2) # Merge components

            # Update list of components
            all_components.remove(component_1)
            all_components.remove(component_2)
            all_components.append(new_component)

            
            step += 1
            draw_components(all_components, all_nodes, step, pos)


    # Print the final components nicely
    print("Final components:")
    for comp in all_components:
        print(sorted(comp))
