# mst-bock
Revisiting MST-Based Dependency Parsing through Bock's Algorithm

In `stanza/stanza/models/depparse/trainer.py`:
```
19c19
< from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
---
> from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root, bock_with_single_root
150c150
<         head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
---
>         head_seqs = [bock_with_single_root(-adj[:l, :l].T, 0)[1][1:] for adj, l in zip(preds[0], sentlens)]
```

We also provide an updated `stanza/stanza/models/common/chuliu_edmonds.py` that defines both `bock_with_single_root()` and the core `bock_algorithm()` functions.
