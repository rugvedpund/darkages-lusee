Map making a form of dimensionality reduction
CP decomposition is a technique for dimensionality reduction
Can I generate sky maps from TOD?

## The Map-Making Problem

- traditional techniques assume operations are performed on a frequency by frequency basis
- so, no effect of chromaticity, freq-space correlations

given a map $m(\nu)$ at different frequencies $\nu$,
we can get time-ordered data $d(t,\nu)$ simulated for some assumptions about the beams, tground, beam-coupling, etc. parametrized as $\hat{\theta}$

Assuming linear, 
$$
\begin{align}
d(t,\nu; \hat{\theta}) = A(t;\hat{\theta}) \ m(\nu) 
\end{align}
$$

most importantly, $\hat{\theta}$ contains information like the turntable rotations, ground temperature, antenna couplings, etc.

label the antenna combination with $a$, and the overall turntable rotation with $\phi$,

$$
\begin{align}
d(t,\nu,a,\phi) = A(t,a,\phi)\ m(\nu) 
\end{align}
$$

given that the map can be decomposed into separate physical components 
$$
\begin{align}
m(\nu) = m_{\mathrm{fg}}(\nu) + m_{\mathrm{CMB}}(\nu) + m_{\mathrm{21cm}}(\nu)
\end{align}
$$

thus, the simulated data is also separable into three components.

## Higher Order Singular Value Decomposition

