import networkx as nx
from networkx.utils import py_random_state


@py_random_state(3)
def watts_strogatz_flexible_graph(n, k, p, seed=None):
    """Returns a Watts–Strogatz flexible small-world graph defined by
    https://arxiv.org/abs/2007.06559

    Parameters
    ----------
    n : int
        The number of nodes
    k : float or int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    watts_strogatz_graph()

    References
    ---------
    .. [1] You, Jiaxuan et al. “Graph Structure of Neural Networks.” (2020).
    """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.Graph()
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, int(k//2) + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))

    # picks e mod n nodes and connects each node to
    # one closest neighboring node
    potential_nodes = list(range(n))
    seed.shuffle(potential_nodes)
    total_add_edges = int(n*k//2) % n
    add_edge_num = 0
    for j in potential_nodes:
        if add_edge_num == total_add_edges:
            break
        i = (j + int(k//2) + 1) % n
        if G.has_edge(j, i):
            continue  # when n = 4, k = 3, additional edges may crash
        else:
            G.add_edge(j, i)
            add_edge_num += 1

    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, int(k//2) + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G
