# MST by Bock's algorithm

- Yuxi Wang, Jungyeul Park* (August 2026, v2). 
**A gentle tutorial on Bock's algorithm for minimum directed spanning trees with a structured reformulation** (https://arxiv.org/abs/2603.27530). 

- Yuxi Wang, Jungyeul Park* (October 2026). 
**Reformulation and Equivalence of Bock's Algorithm for Exact Dependency Parsing**. (https://arxiv.org/abs/2611.00939).

- Yuxi Wang, Jungyeul Park* (November 2026). 
**Efficient Implementation and K-Best Enumeration with Bock’s Arborescence Algorithm**

- Frederick C. Bock (1971).
**An algorithm to construct a minimum directed spanning tree in a directed network**.
*Developments in Operations Research; Proceedings of the Third Annual Israel Conference on Operations Research*, pages 29--44. Gordon and Breach, New York. (A scanned copy: https://tinyurl.com/594b96ca).


In `stanza/stanza/models/depparse/trainer.py`:
```
19c19
< from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
---
> from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root, bock_one_root
150c150
<         head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
---
>         head_seqs = [bock_one_root(-adj[:l, :l].T, 0)[1][1:] for adj, l in zip(preds[0], sentlens)]
```

We also provide an updated `stanza/stanza/models/common/chuliu_edmonds.py` containing the Algol-faithful `bock_algorithm()`, the structured reformulation `bock_algorithm_readable()`, and the one-root decoding wrapper `bock_one_root()`.

