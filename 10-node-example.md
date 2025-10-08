# 10-node example with two cycles

This appendix reproduces the original example of
@bock-1971-an-algorithm, a 10-node graph containing two independent
2-cycles. Although the final arborescence coincides with that obtained
by Chu-Liu-Edmonds, Bock's mechanism of cycle resolution is distinct: it
proceeds entirely within the cost-matrix and primal-dual framework,
without explicit contraction.

## Cost matrix

Consider the cost matrix $C$, where rows are source nodes, columns are
target nodes, and node 1 is the root. Entries on the diagonal and in
column 1 are $\infty$ (no self-loops, no edges entering the root):

$$C =
\begin{array}{c|cccccccccc}
   & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\ \hline
1 & \infty & 52 & 88 & 7  & 2  & 9  & 9  & 29 & 69 & 79 \\
2 & \infty & \infty & 2  & 13 & 1  & 9  & 9  & 64 & 31 & 93 \\
3 & \infty & 82 & \infty & 7  & 1  & 9  & 9  & 27 & 83 & 49 \\
4 & \infty & 3  & 59 & \infty & 0  & 9  & 9  & 74 & 16 & 42 \\
5 & \infty & 55 & 96 & 32 & \infty & 9  & 9  & 75 & 65 & 87 \\
6 & \infty & 89 & 96 & 30 & 67 & \infty & 5  & 52 & 42 & 86 \\
7 & \infty & 47 & 64 & 72 & 56 & 8  & \infty & 51 & 52 & 61 \\
8 & \infty & 30 & 33 & 43 & 95 & 28 & 25 & \infty & 3  & 47 \\
9 & \infty & 55 & 64 & 43 & 69 & 42 & 81 & 6  & \infty & 7  \\
10& \infty & 61 & 97 & 63 & 25 & 26 & 71 & 72 & 43 & \infty
\end{array}$$


#### Step 1: Initial starring

For each non-root column, the cheapest incoming edge is starred:
$$\begin{aligned}
&(4,2),\ (2,3),\ (1,4),\ (4,5),\\
&(7,6),\ (6,7),\ (9,8),\ (8,9),\ (9,10).
\end{aligned}$$ The corresponding costs set the initial column duals:
$$\begin{aligned}
& U_1[2]=3,\ U_1[3]=2,\ U_1[4]=7,\ U_1[5]=0,\ U_1[6]=8,\  \\
& U_1[7]=5,\ U_1[8]=6,\ U_1[9]=3,\ U_1[10]=7.    
\end{aligned}$$ These stars induce two cycles:
$$6 \rightleftarrows 7,\qquad 8 \rightleftarrows 9.$$ At this point,
Chu-Liu-Edmonds would contract these cycles, whereas Bock proceeds by
dual adjustment.

#### Step 2: First cycle and reduced costs

Let the first cycle be $S=\{6,7\}$. Reduced costs are defined by
$$\hat c_{ij} = c_{ij} - U_1[j].$$ For the complement row $i=1$ into
cycle columns $j\in S$: $$\hat c_{1,6} = 9 - 8 = 1,\qquad
\hat c_{1,7} = 9 - 5 = 4.$$ The smallest reduced cost is
$\mathrm{DU}=1$. Raise the duals of the cycle columns by this amount:
$$U_1[6]\gets 9,\quad U_1[7]\gets 6.$$ Recompute reduced costs:
$$\hat c_{1,6}=9-9=0,\qquad
\hat c_{1,7}=9-6=3.$$ The first zero appears at $(1,6)$, which becomes
the *barred* (tight) element.

#### Step 3: Transfer path for cycle $\{6,7\}$

Construct the alternating bar--star path:
$$\overline{(1,6)} \;\to\; \star(7,6).$$ Swapping along this path
replaces $\star(7,6)$ by $\star(1,6)$, breaking the cycle. Now
$6 \leftarrow 1$ and $7 \leftarrow 6$.

#### Step 4: Second cycle and reduced costs

Next, consider the cycle $S=\{8,9\}$. Current duals: $U_1[8]=6$,
$U_1[9]=3$. Reduced costs from row 1:
$$\hat c_{1,8} = 29 - 6 = 23,\qquad
\hat c_{1,9} = 69 - 3 = 66.$$ The minimum is $\mathrm{DU}=23$. Raise the
duals: $$U_1[8]\gets 29,\quad U_1[9]\gets 26.$$ Recompute:
$$\hat c_{1,8}=29-29=0,\qquad
\hat c_{1,9}=69-26=43.$$ The zero at $(1,8)$ becomes the barred element.

#### Step 5: Transfer path for cycle $\{8,9\}$

Form the alternating bar--star path:
$$\overline{(1,8)} \;\to\; \star(9,8).$$ Swap to obtain $\star(1,8)$,
replacing $\star(9,8)$, and thus breaking the cycle. Now
$8 \leftarrow 1$ and $9 \leftarrow 8$.

#### Step 6: Final arborescence

After the two swaps, each non-root column has exactly one starred edge:
$$\begin{aligned}
&(4,2),\ (2,3),\ (1,4),\ (4,5),\\
&(1,6),\ (6,7),\ (1,8),\ (8,9),\ (9,10).
\end{aligned}$$ The structure is now acyclic. Expressed as parent
relations: $$1\to4,\quad 4\to2,\quad 2\to3,\quad 4\to5,\quad
1\to6,\quad 6\to7,\quad 1\to8,\quad 8\to9,\quad 9\to10.$$

#### Step 7: Total cost

The total cost of the final arborescence is: $$\begin{aligned}
& C[4,2]+C[2,3]+C[1,4]+C[4,5]+C[1,6]+C[6,7]+C[1,8]+C[8,9]+C[9,10] \\
&= 3+2+7+0+9+5+29+3+7 = 65.
\end{aligned}$$

#### Summary

Both cycles $\{6,7\}$ and $\{8,9\}$ were resolved by a single dual
adjustment and bar--star transfer, without contraction. The final
structure is an exact arborescence identical to the one produced by
Chu-Liu-Edmonds, but achieved through Bock's primal-dual framework
operating entirely within the cost matrix.

## Discussion

The resulting arborescence matches Chu-Liu-Edmonds, but the route taken
is different. Chu-Liu achieves acyclicity by contracting the cycle into
a single node and solving recursively. Bock achieves it by dual
adjustment and a transfer path in the cost matrix. This distinction is
crucial for understanding why Bock's method aligns with primal-dual
optimization, while Chu-Liu remains a greedy graph-theoretic algorithm.

#### Comparison with Chu-Liu-Edmonds

Although both begin by selecting the minimum incoming edge for each
non-root node, the mechanism of cycle resolution differs:

$$\footnotesize{
\begin{array}{p{0.44\linewidth} p{0.44\linewidth}} \toprule
\textbf{Bock} & \textbf{Chu-Liu-Edmonds} \\ \midrule

Form the \emph{critical submatrix} of cycle nodes. 
& 
\emph{Contract} the cycle into a supernode. \\

Increase the dual variables of the cycle columns until an outside edge becomes 
\emph{tight} (reduced cost zero). 
& 
Recompute the edges entering the supernode, adjust their costs, 
and recurse on the contracted graph. \\

Insert the tight edge as a barred element and resolve the cycle by a 
\emph{transfer path} (alternating starred and barred edges). 
& 
Expand the supernode, redistributing the incoming edges to the original nodes. \\

Cycle resolution is achieved through primal-dual updates and local basis swaps. 
& 
Cycle resolution is achieved through recursive contraction and expansion of supernodes. \\  \bottomrule
\end{array}
}$$ In summary, Chu-Liu-Edmonds resolves cycles by greedy contraction
and recursion, whereas Bock resolves them through primal-dual
adjustments and transfer paths.
