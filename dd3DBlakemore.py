# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 13:29:22 2025

@author: Angel Alonso Diaz-Burgos
"""
import numpy as np;

def fluxKernel(x, D, statistics=1., thresholdLinExpsn=1e-5):
    """
    Computes the value(s) of the integral kernel of the flux.

    Parameters
    ----------
    x : float
        First argument of the kernel.
    D : float
        Second argument of the kernel.
    statistics : int, optional
        Kernel for Fermi-Dirac (1) or Bose-Einstein (-1). Default is 1 (Fermi-Dirac).
    thresholdLinExpsn : float, optional
        Threshold for the second argument of the kernel, below which the
        first non-vanishing term of the Maclaurin expansion of the regularized
        kernel is used. The default is 1E-5.

    Returns
    -------
        Value of the integral kernel.

    Notes
    -----
    Note that the implemented expression results from elemental manipulation
    from the proposed expression in the manuscript, only to avoid precision
    losses by using numpys log1p method.
    """
    kernel = np.zeros_like(x, float);        
    f = 1.0/(np.exp(x + 0.5*D) + statistics);
    
    maskLinearize = (np.abs(D) < thresholdLinExpsn*np.abs(x)); # 1s where Kernel --> f(x)
    kernel[maskLinearize > 0] = f[maskLinearize > 0];
    
    kernelArg = np.zeros_like(kernel);  
    kernelArg[maskLinearize == 0] = ( statistics*(np.exp(D) - 1.0)*f )[maskLinearize == 0];     
    kernel[maskLinearize == 0] = statistics*np.log1p(kernelArg[maskLinearize == 0]) / D[maskLinearize == 0];
    
    return kernel;
#end def

def calcDDFlux_3DBlakemore(ek_, el_, Dphikl_, ck_, returnJacobian=False, thresholdLinExpsn=1e-5, **kwargs):
    """
    Computes the carrier flux particularized to Blakemore's approximation for 
    the carrier density of a 3D fermion gas with parabolic valley.

    Parameters
    ----------
    ek_ : float
        Reduced Fermi energy at starting node k. Normalized by kBT. 
    el_ : float
        Reduced Fermi energy at ending node l. Normalized by kBT.
    Dphikl_ : float
        Increment of quasi-Fermi potential, divided by thermal voltage kBT/q.
    ck_ :  float
        Carrier density at starting node k. Normalized by N0. 
    returnJacobian : bool
        Whether to return the derivative of the flux w.r.t. the Fermi potential
        at both nodes.    
    thresholdLinExpsn : float, optional
        Threshold for the reduced Fermi energy difference below which the
        linear contribution is used to compute the flux. The default is 1e-5.
    **kwargs : 
        * dcdeta_k : float
            Derivative of carrier density w.r.t. the reduced Fermi energies,
            divided by N0/VT, at node k.
        * cl : float
            Carrier density, divided by N0, at node l.
        
    Returns
    -------
        Numerical carrier flux along bond from k to l.

    """
    Dekl_ = el_ - ek_;
    ekl_ = 0.5*(ek_ + el_);
    # (Deta)^{-1} * Integral[deta c(eta)]
    integral = fluxKernel(-ekl_ - np.log(0.27), Dekl_)/0.27;
    j = integral*Dphikl_;
    
    if returnJacobian:
        dcdeta_k, cl = kwargs['dcdeta_k'], kwargs['cl'];

        djdphil, djdphik = np.zeros_like(Dphikl_), np.zeros_like(Dphikl_);
        maskRegularize = (np.abs(Dekl_) < thresholdLinExpsn); # 1s where denominator Deta is close to 0 

        DphiDe = np.zeros_like(Dphikl_);
        DphiDe[maskRegularize == 0] = Dphikl_[maskRegularize == 0]/Dekl_[maskRegularize == 0];
        commonTerm = np.zeros_like(Dphikl_);
        commonTerm[maskRegularize == 0] = (-1)*(1.0 + DphiDe[maskRegularize == 0])*integral[maskRegularize == 0]; 
        commonTerm[maskRegularize > 0] = 0.5*(-1)*(Dekl_ + Dphikl_)[maskRegularize > 0]*dcdeta_k[maskRegularize > 0];
        
        djdphik[maskRegularize == 0] = commonTerm[maskRegularize == 0] + ck_[maskRegularize == 0]*DphiDe[maskRegularize == 0];
        djdphik[maskRegularize > 0] = commonTerm[maskRegularize > 0] - ck_[maskRegularize > 0];
            
        djdphil[maskRegularize == 0] = (-1)*commonTerm[maskRegularize == 0] - cl[maskRegularize == 0]*DphiDe[maskRegularize == 0];
        djdphil[maskRegularize > 0] = commonTerm[maskRegularize > 0] + cl[maskRegularize > 0];
        
        return j.copy(), djdphik.copy(), djdphil.copy();
    else:
        return j.copy();
    #end if
#end def