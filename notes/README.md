# Introduction to Higher Order Singular Value Decomposition

In multilinear algebra, an abstract tensor
$\mathcal{A} \in \mathbb{C}^{I_1,...,I_M}$ is an $M$ dimensional
array, where $M$ is the number of modes and the order of the tensor.

The standard m-mode flattening of $\mathcal{A}$ is defined as the
matrix $\mathcal{A}_{(m)}$, whose left index is the $m-$th mode of
$\mathcal{A}$ and columns are indexed by the remaining modes of
$\mathcal{A}\). Let \(\mathbb{U}_m \in \mathbb{C}^{I_m \times I_m}$ be
a unitary matrix containing the basis of the left singular vectors of
$\mathcal{A}_{(m)}\), such that the columns of \(\mathbb{U}_m$ are the
left singular vectors of $\mathcal{A}_{(m)}$.

By properties of multi-linear algebra, we have 
$$\begin{equation}
    \mathcal{A} &= \mathcal{A} \times (\rm{id}_{I_1},\rm{id}_{I_2},...,\rm{id}_{I_M}) \\
                &= \mathcal{A} \times (U_1U_1^\dagger, U_2U_2^\dagger,...,U_MU_M^\dagger) \\
                &= \mathcal{A} \times (U_1, U_2,...,U_M) \times (U_1^\dagger, U_2^\dagger,...,U_M^\dagger) \\
                &= \mathcal{S} \times (U_1, U_2,...,U_M)
\end{equation}$$


where $\mathcal{S} = \mathcal{A} \times (U_1^\dagger, U_2^\dagger,...,U_M^\dagger)$
is the core tensor of $\mathcal{A}$.

This defines the Higher Order Singular Value Decomposition (HOSVD) of a
tensor $\mathcal{A}$, which is a generalization of the matrix SVD to
higher order tensors. The above construction shows that every tensor has
a HOSVD.

# LuSEE-Night Simulation Products

## Single Waterfall

The fundamental data product of simulations for LuSEE-Night are the
waterfalls, which are 2D arrays of complex visibilities, indexed by the
time and frequency. The waterfalls are generated for a single antenna
combination, and the simulation outputs are indexed by the time,
frequency, and antenna combination.

Fig below shows a waterfall of a single antenna
combination, where the x-axis is the time, the y-axis is the frequency,
and the colorbar is the amplitude of the real visibility in this case.

![waterfall.pdf]

## Multiple Antenna Combinations

## Turntable Rotations
