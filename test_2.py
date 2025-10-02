import random
import matplotlib.pyplot as plt
import networkx as nx

"""Should be fixed implementation. Will check"""


def base_label(node):
    """Extract base vertex from a node like '3_1' -> '3'."""
    return node.split("_")[0]

def can_merge(comp1, comp2, base_edges):
    """Check if two components can merge."""
    base1 = {base_label(x) for x in comp1}
    base2 = {base_label(x) for x in comp2}

    # must be disjoint in base vertices
    if not base1.isdisjoint(base2):
        return False

    # must have at least one base edge crossing
    for u, v in base_edges:
        if (str(u) in base1 and str(v) in base2) or (str(v) in base1 and str(u) in base2):
            return True
    return False

def draw_components(components, all_nodes, step, pos):
    """Draw components as cliques of their current members."""
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
    plt.show()

if __name__ == "__main__":
    # base graph
    vertices = [1, 2, 3, 4]
    base_edges = [(1, 3), (1, 2), (2, 4)]
    n = 3  # number of copies

    # label nodes like '1_0', '1_1', ...
    all_nodes = [f"{v}_{copy}" for copy in range(n) for v in vertices]
    all_components = [{v} for v in all_nodes]  # singleton components

    # consistent layout
    G_nodes = nx.Graph()
    G_nodes.add_nodes_from(all_nodes)
    pos = nx.spring_layout(G_nodes, seed=42)

    step = 0
    draw_components(all_components, all_nodes, step, pos)

    while len(all_components) > n:
        comp1, comp2 = random.sample(all_components, 2)

        if can_merge(comp1, comp2, base_edges):
            new_comp = comp1.union(comp2)
            all_components.remove(comp1)
            all_components.remove(comp2)
            all_components.append(new_comp)

            step += 1
            #print(f"Step {step}: merged {comp1} and {comp2}")
            draw_components(all_components, all_nodes, step, pos)

    print("Final components:")
    for comp in all_components:
        print(sorted(comp))
