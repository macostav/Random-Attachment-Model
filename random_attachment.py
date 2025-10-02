import random
from collections import defaultdict, Counter

"""Modelling a random attachment graph. We start with a simple graph with 4 vertices connected at all but one edge."""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
        return True

def simulate_lift(T_edges, n, mode="edge", seed=None, verbose=False):
    """
    Simulate Amit–Linial lift (random cover) of a tree T.
    
    T_edges: list of (u,v) tree edges, u,v in {0,...,k-1}
    n: number of copies
    mode: "edge" or "trial"
    """
    if seed is not None:
        random.seed(seed)

    # canonicalize tree edges
    T = set((min(u,v), max(u,v)) for (u,v) in T_edges)
    k = 1 + max(max(u,v) for u,v in T)

    # vertex IDs: 0..n*k-1
    def pair_to_id(copy, node):
        return copy * k + node
    def id_to_pair(idx):
        return (idx // k, idx % k)

    N = n * k
    uf = UnionFind(N)

    # adjacency on tree
    neighbors = {i:set() for i in range(k)}
    for u,v in T:
        neighbors[u].add(v)
        neighbors[v].add(u)

    added_edges = []
    clock_time = 0 if mode == "trial" else None
    successes = 0
    target_successes = n * len(T)

    while successes < target_successes:
        if mode == "edge":
            valid = []
            for u,v in T:
                for a in range(n):
                    for b in range(n):
                        ida, idb = pair_to_id(a,u), pair_to_id(b,v)
                        if ida >= idb: 
                            continue
                        if uf.find(ida) != uf.find(idb):
                            valid.append((ida,idb,u,v))
            if not valid:
                break
            ida,idb,u,v = random.choice(valid)
            uf.union(ida,idb)
            added_edges.append((ida,u,idb,v))
            successes += 1

        elif mode == "trial":
            # rebuild component -> vertices map
            comp_vertices = defaultdict(list)
            for vid in range(N):
                comp_vertices[uf.find(vid)].append(vid)
            comps = list(comp_vertices.keys())

            trials = 0
            while True:
                trials += 1
                A,B = random.sample(comps,2)
                possible_pairs = []
                for va in comp_vertices[A]:
                    _, node_a = id_to_pair(va)
                    for vb in comp_vertices[B]:
                        _, node_b = id_to_pair(vb)
                        if (min(node_a,node_b), max(node_a,node_b)) in T:
                            if va < vb:
                                possible_pairs.append((va,vb,node_a,node_b))
                            else:
                                possible_pairs.append((vb,va,node_b,node_a))
                if possible_pairs:
                    ida,idb,u,v = random.choice(possible_pairs)
                    uf.union(ida,idb)
                    added_edges.append((ida,u,idb,v))
                    successes += 1
                    clock_time += trials
                    break
        else:
            raise ValueError("mode must be 'edge' or 'trial'")

        if verbose and successes % 50 == 0:
            print(f"Added {successes}/{target_successes} edges")

    # summarize results
    edge_counts = Counter()
    for ida,u,idb,v in added_edges:
        edge_counts[(min(u,v),max(u,v))] += 1

    return {
        "added_edges": added_edges,
        "edge_counts": edge_counts,
        "clock_time": clock_time,
        "final_components": len(set(uf.find(i) for i in range(N)))
    }

# ------------------- Example usage -------------------
if __name__ == "__main__":
    # Tree = path of length 3 on 4 vertices: 0-1-2-3
    T_edges = [(0,1),(1,2),(2,3)]
    n = 10

    res = simulate_lift(T_edges, n, mode="edge", seed=42, verbose=True)
    print("\nEdge mode results:")
    print("  total edges added:", len(res["added_edges"]))
    print("  edge counts:", dict(res["edge_counts"]))
    print("  number of components:", res["final_components"])

    res2 = simulate_lift(T_edges, n, mode="trial", seed=123)
    print("\nTrial mode results:")
    print("  total edges added:", len(res2["added_edges"]))
    print("  clock time (total trials):", res2["clock_time"])
    print("  number of components:", res2["final_components"])
