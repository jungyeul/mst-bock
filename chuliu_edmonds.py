# Adapted from Tim's code here: https://github.com/tdozat/Parser-v3/blob/master/scripts/chuliu_edmonds.py

import numpy as np

def tarjan(tree):
    """Finds the cycles in a dependency graph

    The input should be a numpy array of integers,
    where in the standard use case,
    tree[i] is the head of node i.

    tree[0] == 0 to represent the root

    so for example, for the English sentence "This is a test",
    the input is

    [0 4 4 4 0]

    "Arthritis makes my hip hurt"

    [0 2 0 4 2 2]

    The return is a list of cycles, where in cycle has True if the
    node at that index is participating in the cycle.
    So, for example, the previous examples both return empty lists,
    whereas an input of
      np.array([0, 3, 1, 2])
    has an output of
      [np.array([False,  True,  True,  True])]
    """
    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []
    #-------------------------------------------------------------
    def maybe_pop_cycle(i):
        if lowlinks[i] == indices[i]:
            # There's a cycle!
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)

    def initialize_strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True

    def strong_connect(i):
        # this ridiculous atrocity is because somehow people keep
        # coming up with graphs which overflow python's call stack
        # so instead we make our own call stack and turn the recursion
        # into a loop
        # see for example
        #   https://github.com/stanfordnlp/stanza/issues/962
        #   https://github.com/spraakbanken/sparv-pipeline/issues/166
        # in an ideal world this block of code would look like this
        #    initialize_strong_connect(i)
        #    dependents = iter(np.where(np.equal(tree, i))[0])
        #    for j in dependents:
        #        if indices[j] == -1:
        #            strong_connect(j)
        #            lowlinks[i] = min(lowlinks[i], lowlinks[j])
        #        elif onstack[j]:
        #            lowlinks[i] = min(lowlinks[i], indices[j])
        #
        #     maybe_pop_cycle(i)
        call_stack = [(i, None, None)]
        while len(call_stack) > 0:
            i, dependents_iterator, j = call_stack.pop()
            if dependents_iterator is None: # first time getting here for this i
                initialize_strong_connect(i)
                dependents_iterator = iter(np.where(np.equal(tree, i))[0])
            else: # been here before.  j was the dependent we were just considering
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            for j in dependents_iterator:
                if indices[j] == -1:
                    # have to remember where we were...
                    # put the current iterator & its state on the "call stack"
                    # we will come back to it later
                    call_stack.append((i, dependents_iterator, j))
                    # also, this is what we do next...
                    call_stack.append((j, None, None))
                    # this will break this iterator for now
                    # the next time through, we will continue progressing this iterator
                    break
                elif onstack[j]:
                    lowlinks[i] = min(lowlinks[i], indices[j])
            else:
                # this is an intended use of for/else
                # please stop filing git issues on obscure language features
                # we finished iterating without a break
                # and can finally resolve any possible cycles
                maybe_pop_cycle(i)
            # at this point, there are two cases:
            #
            # we iterated all the way through an iterator (the else in the for/else)
            # and have resolved any possible cycles.  can then proceed to the previous
            # iterator we were considering (or finish, if there are no others)
            # OR
            # we have hit a break in the iteration over the dependents
            # for a node
            # and we need to dig deeper into the graph and resolve the dependent's dependents
            # before we can continue the previous node
            #
            # either way, we check to see if there are unfinished subtrees
            # when that is finally done, we can return

    #-------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

def process_cycle(tree, cycle, scores):
    """
    Build a subproblem with one cycle broken
    """
    # indices of cycle in original tree; (c) in t
    cycle_locs = np.where(cycle)[0]
    # heads of cycle in original tree; (c) in t
    cycle_subtree = tree[cycle]
    # scores of cycle in original tree; (c) in R
    cycle_scores = scores[cycle, cycle_subtree]
    # total score of cycle; () in R
    cycle_score = cycle_scores.sum()

    # locations of noncycle; (t) in [0,1]
    noncycle = np.logical_not(cycle)
    # indices of noncycle in original tree; (n) in t
    noncycle_locs = np.where(noncycle)[0]
    #print(cycle_locs, noncycle_locs)

    # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
    metanode_head_scores = scores[cycle][:,noncycle] - cycle_scores[:,None] + cycle_score
    # scores of cycle's potential dependents; (n x c) in R
    metanode_dep_scores = scores[noncycle][:,cycle]
    # best noncycle head for each cycle dependent; (n) in c
    metanode_heads = np.argmax(metanode_head_scores, axis=0)
    # best cycle head for each noncycle dependent; (n) in c
    metanode_deps = np.argmax(metanode_dep_scores, axis=1)

    # scores of noncycle graph; (n x n) in R
    subscores = scores[noncycle][:,noncycle]
    # pad to contracted graph; (n+1 x n+1) in R
    subscores = np.pad(subscores, ( (0,1) , (0,1) ), 'constant')
    # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
    subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
    # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
    subscores[:-1,-1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]
    return subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps


def expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps):
    """
    Given a partially solved tree with a cycle and a solved subproblem
    for the cycle, build a larger solution without the cycle
    """
    # head of the cycle; () in n
    #print(contracted_tree)
    cycle_head = contracted_tree[-1]
    # fixed tree: (n) in n+1
    contracted_tree = contracted_tree[:-1]
    # initialize new tree; (t) in 0
    new_tree = -np.ones_like(tree)
    #print(0, new_tree)
    # fixed tree with no heads coming from the cycle: (n) in [0,1]
    contracted_subtree = contracted_tree < len(contracted_tree)
    # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
    new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
    #print(1, new_tree)
    # fixed tree with heads coming from the cycle: (n) in [0,1]
    contracted_subtree = np.logical_not(contracted_subtree)
    # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
    new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
    #print(2, new_tree)
    # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
    new_tree[cycle_locs] = tree[cycle_locs]
    #print(3, new_tree)
    # root of the cycle; (n)[() in n] in c = () in c
    cycle_root = metanode_heads[cycle_head]
    # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
    new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
    #print(4, new_tree)
    return new_tree

def prepare_scores(scores):
    """
    Alter the scores matrix to avoid self loops and handle the root
    """
    # prevent self-loops, set up the root location
    np.fill_diagonal(scores, -float('inf')) # prevent self-loops
    scores[0] = -float('inf')
    scores[0,0] = 0

def chuliu_edmonds(scores):
    subtree_stack = []

    prepare_scores(scores)
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)

    #print(scores)
    #print(cycles)

    # recursive implementation:
    #if cycles:
    #    # t = len(tree); c = len(cycle); n = len(noncycle)
    #    # cycles.pop(): locations of cycle; (t) in [0,1]
    #    subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps = process_cycle(tree, cycles.pop(), scores)
    #    # MST with contraction; (n+1) in n+1
    #    contracted_tree = chuliu_edmonds(subscores)
    #    tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)
    # unfortunately, while the recursion is simpler to understand, it can get too deep for python's stack limit
    # so instead we make our own recursion, with blackjack and (you know how it goes)

    while cycles:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # cycles.pop(): locations of cycle; (t) in [0,1]
        subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps = process_cycle(tree, cycles.pop(), scores)
        subtree_stack.append((tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps))

        scores = subscores
        prepare_scores(scores)
        tree = np.argmax(scores, axis=1)
        cycles = tarjan(tree)

    while len(subtree_stack) > 0:
        contracted_tree = tree
        (tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps) = subtree_stack.pop()
        tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)

    return tree

#===============================================================
def chuliu_edmonds_one_root(scores):
    """"""

    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0]+1
    if len(roots_to_try) == 1:
        return tree

    #-------------------------------------------------------------
    def set_root(scores, root):
        root_score = scores[root,0]
        scores = np.array(scores)
        scores[1:,0] = -float('inf')
        scores[root] = -float('inf')
        scores[root,0] = 0
        return scores, root_score
    #-------------------------------------------------------------

    best_score, best_tree = -np.inf, None # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = (tree_probs).sum()+(root_score) if (tree_probs > -np.inf).all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree

#===============================================================
def bock_algorithm(cost_matrix, root):
    n = cost_matrix.shape[0]
    M = np.inf  # robust infinity

    U1 = np.zeros(n, dtype=float)
    I_star = np.full(n, -1, dtype=int)      # parent of column/node j
    bars = np.full((n, 2), -1, dtype=int)   # (I_bar[j], J_bar[j])
    span = np.arange(n, dtype=int)
    SS = n
    K = -1

    while K < n - 1:
        K += 1
        if K == root:
            continue

        while True:
            DU = M
            H = span[K]
            I1 = J1 = -1

            # L4: search complement rows for each column J with span[J]==H
            for J in range(K + 1):
                if span[J] != H:
                    continue
                # rows outside H & finite costs
                rows = np.where((span != H) & np.isfinite(cost_matrix[:, J]))[0]
                if rows.size == 0:
                    continue
                diffs = cost_matrix[rows, J] - U1[J]
                m = np.argmin(diffs)
                if diffs[m] < DU:
                    DU = diffs[m]
                    I1, J1 = rows[m], J

            if not np.isfinite(DU):
                return None, None  # infeasible

            # L5: raise duals for columns with span==H up to K
            colmask = (np.arange(K + 1) >= 0) & (span[:K + 1] == H)
            idx = np.nonzero(colmask)[0]
            U1[idx] += DU

            # L6-L8
            J = I1
            from_L8 = False
            while True:
                # L7: if node J is inside H, open new critical submatrix
                if span[J] == H:
                    SS += 1
                    m2 = (span[:K + 1] == H) | (span[:K + 1] < 0)
                    span[:K + 1][m2] = SS
                    # go back to L4
                    break

                # L8: follow star if exists
                if I_star[J] >= 0:
                    if span[J] >= 0:
                        H1 = span[J]
                        m3 = (span[:K + 1] == H1)
                        span[:K + 1][m3] = -span[:K + 1][m3]
                    if bars[J, 0] == -1:
                        bars[J] = (I1, J1)
                    J = I_star[J]
                else:
                    from_L8 = True
                    break

            if from_L8:
                # proceed to L9
                break

        # L9
        span[:K + 1] = np.abs(span[:K + 1])

        # L10-L11: transfer path swaps (bars <-> stars)
        I2 = J2 = -1
        while True:
            m4 = np.all(bars[:K + 1] == (I1, J1), axis=1)
            bars[:K + 1][m4] = (I2, J2)

            I, J = bars[J1]
            bars[J1] = (I2, J2)
            I2 = I_star[J1]
            I_star[J1] = I1
            if I2 == -1:
                break
            J2 = J1
            I1, J1 = I, J

    # L99: sum cost
    Z = 0.0
    for J in range(n):
        if J == root:
            continue
        if I_star[J] < 0 or not np.isfinite(cost_matrix[I_star[J], J]):
            return None, None  # guard
        Z += cost_matrix[I_star[J], J]

    I_star[root] = root  # parent of root
    return Z, I_star

#===============================================================
def bock_one_root(cost_matrix, root):
    """
    - Runs bock_algorithm(cost_matrix, root) once.
    - Finds nodes whose parent is the root in that solution.
    - If multiple, re-run while forcing exactly one root->j edge at a time,
      picking the minimum-cost solution among those tried.
    
    Args:
        cost_matrix: np.ndarray of shape (n, n), dtype float.
                     Entry [i, j] is the cost of arc i -> j (np.inf to forbid).
        root: int, index of designated root node.
        bock_algorithm: callable(cost_matrix, root) -> (Z, par)
            Should return total cost Z (float) and parent array par (len n),
            where par[j] = parent of j, and typically par[root] = -1 or root.

    Returns:
        (best_Z, best_par)
        best_Z: float (min total cost), or None if infeasible.
        best_par: np.ndarray of parents, or None if infeasible.

    Notes:
        - cost_matrix must be MIN-COST (not max-score). If you have scores S,
          pass C = -S (and np.inf for forbidden).
        - We only iterate over nodes that were root-children in the first run,
          mirroring chuliu_edmonds_one_root's behavior.
    """
    C0 = np.asarray(cost_matrix, dtype=float)
    n = C0.shape[0]
    assert C0.shape == (n, n)

    # First pass: get an initial arborescence
    Z0, par0 = bock_algorithm(C0, root)
    if Z0 is None or par0 is None:
        return None, None

    # Identify nodes whose parent is the root in the initial solution
    # Exclude the root itself
    roots_to_try = [j for j in range(n) if j != root and par0[j] == root]

    # If unique root child, we’re done
    if len(roots_to_try) <= 1:
        return Z0, par0

    best_Z, best_par = None, None

    for j_keep in roots_to_try:
        # Copy and enforce: allow ONLY root->j_keep; forbid root->k for k != j_keep
        C = C0.copy()
        mask = np.ones(n, dtype=bool)
        mask[root] = False
        mask[j_keep] = False
        # forbid all root->k except j_keep
        C[root, mask] = np.inf

        Z, par = bock_algorithm(C, root)
        if Z is not None and (best_Z is None or Z < best_Z):
            best_Z, best_par = Z, par

    # Fallback: if all constrained runs failed, return the initial solution
    # (mirrors typical CLE wrapper behavior that keeps a valid tree if found)
    if best_Z is None:
        return Z0, par0
    return best_Z, best_par
