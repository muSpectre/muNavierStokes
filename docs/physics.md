# Physics

## Governing equations

For an incompressible fluid of constant density with kinematic viscosity
\(\nu\),

\[
\frac{\partial u}{\partial t} + (u\cdot\nabla)u = -\nabla p + \nu\,\nabla^2 u ,
\qquad \nabla\cdot u = 0 .
\]

The code integrates the **rotational form** of the advection term. Using the
identity \((u\cdot\nabla)u = (\nabla\times u)\times u + \nabla(\tfrac12|u|^2)\)
and folding the gradient term into a modified pressure
\(P = p + \tfrac12|u|^2\),

\[
\frac{\partial u}{\partial t} = u\times\omega - \nabla P + \nu\,\nabla^2 u ,
\qquad \omega = \nabla\times u \quad\text{(vorticity).}
\]

## Fourier representation

The velocity field is expanded in Fourier modes on a triply-periodic box. With
\(\hat u(k)\) the Fourier coefficient of \(u\) and wavevector \(k\), derivatives
become algebraic:

- gradient \(\to\) multiplication by \(i k\)
- Laplacian \(\to\) multiplication by \(-|k|^2\)
- vorticity \(\to\) \(\hat\omega = i k \times \hat u\)

Incompressibility \(\nabla\cdot u = 0\) becomes \(k\cdot\hat u = 0\): the
velocity is transverse to its wavevector. The pressure is eliminated by
**projecting** the right-hand side onto the divergence-free subspace
(Leray/Helmholtz projection),

\[
P_\perp f = f - \frac{k\,(k\cdot f)}{|k|^2}.
\]

Applying the projection to the momentum equation removes \(\nabla P\) and yields
the evolution equation actually integrated:

\[
\frac{\partial \hat u}{\partial t}
   = P_\perp\!\big(\widehat{u\times\omega}\big) - \nu\,|k|^2\,\hat u .
\]

The nonlinear term \(\widehat{u\times\omega}\) is computed
**pseudo-spectrally**: transform \(\hat u\) and \(\hat\omega\) to real space,
form the cross product point-wise, and transform back.

## Dealiasing (the 2/3 rule)

The point-wise product of two band-limited fields produces wavenumbers up to
twice the maximum resolved wavenumber. On a finite grid these fold back
("alias") onto the resolved modes and corrupt the solution. The **2/3 rule**
removes this error: zero all modes with \(|k_i| \ge \tfrac23 k_\text{max}\)
*before* forming the product, so the aliased content lands only in the
(discarded) upper third of the spectrum.

The solver applies the dealiasing mask in two places:

1. **On the inputs** \(\hat u\) and \(\hat\omega\), before the inverse
   transforms. This is what keeps the resolved modes free of aliasing error. It
   is essential whenever the field carries energy in the cut band — e.g. the
   random `turbulence` initial condition.
2. **On the nonlinear output** \(\widehat{u\times\omega}\), after the forward
   transform. This prevents spurious aliased energy from being injected into the
   cut-band modes, which then merely decay viscously and stay decoupled from the
   resolved dynamics.

The cut-off wavenumber is \(k_c = 2\pi/(3\,\Delta x)\), i.e.
\(\tfrac23 k_\text{max}\) with \(k_\text{max} = \pi/\Delta x\) the Nyquist
wavenumber.
