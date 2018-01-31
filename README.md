
Permutation Invariant Layers for Neural Networks
====================================================

Suppose our data follows the model

$$
    y = f(x)
$$

and suppose that the value of $f$ does not depend on the permutation of rows of $x \in R^{m x n}$.
If we try to model $f$ with standard neural network this is a problem. This project proposes one approach for
solving it.
Suppose $x = (x_1, ... , x_m)$, where $x_i \in R^n$. We consider all possible k-tuples of $x_i$ and on each tuple we apply the same
function (which is specified as a dense neural network). Then we pool the result over the tuples.
