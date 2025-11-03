# MST by Bock's algorithm

Yuxi Wang, Jungyeul Park* (November 2025). 
Revisiting MST-Based Dependency Parsing through Bock's Primal-Dual Algorithm.
Submitted to *Computational Linguistics -- MIT Press* as Squibs and Discussions. *Corresponding author.

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

We also provide an updated `stanza/stanza/models/common/chuliu_edmonds.py` that defines both `bock_one_root()` and the core `bock_algorithm()` functions.


To assess the practical runtime differences between the two minimum spanning tree decoding strategies, we measured wall-clock training times for Bock’s primal–dual algorithm and the Chu–Liu–Edmonds algorithm. All experiments were performed under identical hardware and software conditions: PyTorch 2.7.0 with Python 3.12 on Ubuntu 22.04, compiled against CUDA 12.8. The system was equipped with a single NVIDIA RTX 5090 GPU (32GB memory), 25 vCPUs on an Intel(R) Xeon(R) Platinum 8470Q processor, and 120GB of RAM.

For Bock’s algorithm, five independent runs yielded durations between 2463 and 2470 seconds (≈41 minutes each). The five-run average was 2467 seconds, with a standard deviation of 2.94 seconds, reflecting remarkable stability across trials.

For Chu–Liu–Edmonds, the corresponding five runs ranged from 2426 to 2440 seconds (≈40.5 minutes each). The five-run average was 2434 seconds, with a standard deviation of 5.89 seconds, again demonstrating consistent performance with only minimal variability.

Overall, the results indicate that on this hardware configuration both algorithms achieve nearly identical training efficiency. Chu–Liu–Edmonds shows a modest average advantage of roughly 33 seconds per run (≈1.3%), whereas Bock’s algorithm displays slightly lower variance across runs.


## example
See the original example in [{bock-1971-an-algorithm}](https://drive.google.com/file/d/1W72sXq2xKzBZ_MDLECj_OMInLrxjtVe-/view?usp=share_link), which illustrates a 10-node graph containing two independent 2-cycles, for a clearer understanding of Bock’s algorithm  [[Example](10-node-example.md)].


