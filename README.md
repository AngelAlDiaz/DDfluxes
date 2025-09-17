# DDfluxes
The current directory "DDfluxes" contains a series of .py files (i.e. in Python language), one for each of the finite-volume Drift-Diffusion (DD) numerical flux formulas derived, mentioned and/or employed in the manuscript "Accurate computation of numerical Drift-Diffusion fluxes in Arbitrarily Degenerate Gases". 

These functions receive a minimal set of arguments consisting of dynamical variables evaluated at the nodes connected by the bond along which the flux is estimated. For example, in most cases these arguments are the arithmetic average of the reduced Fermi energy along an arbitrary bond connecting two adjacent nodes in the Finite-Volume mesh, the increment of the reduced Fermi energy (or alternative variables) along the bond, and the increment of the quasi-Fermi potential along the bond. They return the corresponding flux across the bond(s), divided by the mu*N0*VT/h, where h is the length of the bond, mu the mobility, N0 the effective valley Density of States (DoS) and VT the thermal voltage. On the other hand, the derivatives of the fluxes w.r.t. local quasi-Fermi potential (discrete Jacobian) are divided by the same quantity, times VT. Therefore, the outputs are non-dimensionalized. This is done so since most semiconductor device simulators may evaluate the mobility by physical models implemented in independent modules. In our opinion, this way of implementing the formulas makes it compatible with arbitrary Finite-Volume implementations that solve the Continuity Equations (be it 1D, 2D, 3D, for electrons, holes, ions, excitons, etc). The idea is that the solver feeds these general routines that return the fluxes, which are readily used to build the Newton matrix equation during each self-consistent field step.

Each routine is accompanied by a brief minimal description contextualizing the corresponding numerical model (discretized flux formula) as well as the received parameters. The routines are named starting with "calcDDFlux_" followed by a descriptive acronym of the corresponding flux formula. In the following, a guide is provided:

-  ddSG.py\calcDDFlux_SG : classical Scharfetter-Gummel. Eq. (16) in manuscript.

-  ddDE.py\calcDDFlux_DE : modified "Diffusion-Enhanced" Scharfetter-Gummel. See Eq. (7) in  10.1007/s11082-014-0050-9.

-  ddDoSxf.py\calcDDFlux_DoSxf : proposed general formula for arbitrary Density of States (DoS). Eqs. (28)--(31) in manuscript.

-  dd3DBlakemore.py\calcDDFlux_3DBlakemore : proposed formula for Blakemore approximation. Eq. (47) in manuscript.

-  dd2DFD.py\calcDDFlux_2DM : proposed formula for 2D fermion gas with parabolic dispersion. Eq. (51) in manuscript.

-  ddGr.py\calcDDFlux_Gr : proposed formula for Graphene monolayer. Eq. (53) in manuscript.

-  dd2DBE.py\calcDDFlux_2DBE : proposed formula for 2D boson gas with parabolic dispersion. Eq. (57) in manuscript.

