# SISRRA_tensor_contraction

ICOSSAR 2021 submission codebase:

ABSTRACT: Quantifying network reliability is a hard problem, proven to be #P-complete
[1]. For real-world network planning and decision making, approximations for the network
reliability problem are necessary. This study shows that tensor network contraction (TNC)
methods can quickly estimate an upper bound of All Terminal Reliability, Rel_ATR(G), by
solving a superset of the network reliability problem: the edge cover problem, EC(G). In
addition, these tensor contraction methods can exactly solve S-T reliability for the class of
directed acyclic networks, Rel_S-T (G).
The computational complexity of TNC methods is parameterized by treewidth, significantly
benefitting from recent advancements in approximate tree decomposition algorithms
[2]. This parameterization does not rely on the reliability of the graph, which means these tensor
contraction methods can determine reliability faster than Monte Carlo methods on highly
reliable networks, while also providing exact answers or guaranteed upper bound estimates.
These tensor contraction methods are applied to grid graphs, random cubic graphs, and a selection
of 58 power transmission networks [3], demonstrating computational efficiency and
effective approximation using EC(G).

[1] Leslie G Valiant. The complexity of enumeration
and reliability problems. SIAM Journal on Computing, 8(3):410–421, 1979.

[2] Holger Dell, Christian Komusiewicz, Nimrod
Talmon, and Mathias Weller. The pace
2017 parameterized algorithms and computational
experiments challenge: The second
iteration. In 12th International Symposium
on Parameterized and Exact Computation
(IPEC 2017). Schloss Dagstuhl-
Leibniz-Zentrum fuer Informatik, 2018.

[3] Jian Li, Leonardo Dueñas-Osorio,
Changkun Chen, Benjamin Berryhill,
and Alireza Yazdani. Characterizing the
topological and controllability features of us
power transmission networks. Physica A:
Statistical Mechanics and its Applications,
453:84–98, 2016.
