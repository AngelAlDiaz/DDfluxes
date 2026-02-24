# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 13:29:22 2025

@author: Angel Alonso Diaz-Burgos
"""
import sys;
import numpy as np;
from scipy.integrate import trapezoid;

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
    
def calcIntegralDoSKernel(E, DoS, eta, Deta, statistics=1., thresholdLinExpsn=1e-5):
    """
    Computes the integral of the DoS times the non-linear kernel, required to
    evaluate flux and its derivative.

    Parameters
    ----------
    E : 1D array of floats
        Energies referred to the valley extreme (minimum particle energy
        of the ensemble). Divided by thermal energy kBT.
    DoS : 1D array of floats
        Corresponding density of states. Divided by N0/kBT, where N0 has
        dimensions of particle density, so as to make the expression
        dimensionless.
    eta : float
        Reduced Fermi energy. Normalized by kBT.
    Deta : float
        Increment of reduced Fermi energy, divided by kBT.
    statistics : int, optional
        Kernel for Fermi-Dirac (1) or Bose-Einstein (-1). Default is 1 (Fermi-Dirac).
    thresholdLinExpsn : float, optional
        Threshold for the reduced Fermi energy difference below which the
        linear contribution is used to compute the flux. The default is 1e-5.

    Returns
    -------
        Normalized integral of DoS times the integration Kernel.
    """
    if not ((E.shape == DoS.shape) and (len(E.shape) == 1)):
        print('DDfluxes.calcIntegralDoSKernel Error: arguments <<E>> and\
            <<DoS>> have the same shape and be 1D arrays.');
        sys.exit(1);
    #end if
    result = np.ones_like(eta);
    etaIsScalar, DetaIsScalar = np.isscalar(eta), np.isscalar(Deta);
    if etaIsScalar and not DetaIsScalar: etaarr, Detaarr = eta*np.ones_like(Deta), Deta.copy();
    elif not etaIsScalar and DetaIsScalar: etaarr, Detaarr = eta.copy(), Deta*np.ones_like(eta);
    elif not etaIsScalar and not DetaIsScalar:
        etaarr, Detaarr = np.array(eta), np.array(Deta);
        if (etaarr.shape != Detaarr.shape):
            print('DDfluxes.calcIntegralDoSKernel Error: arguments <<eta>> and\
                <<Deta>> must either have the same shape or at least one of them be scalar');
            sys.exit(1);
        #end if
    else:
        etaarr, Detaarr = 1.0*eta, 1.0*Deta;
    #end if
    shapeOutput = [lenAxis for lenAxis in etaarr.shape]; # Shape as list, not tuple
    if np.isscalar(eta) and np.isscalar(Deta):
        etaAux, DetaAux = etaarr, Detaarr;
    else:
        etaAux, DetaAux = np.array(etaarr.flat), np.array(Deta.flat);
        EAux = np.tile(E[np.newaxis,:], (len(etaAux), 1));
        DoSAux = np.tile(DoS[np.newaxis,:], (len(etaAux), 1));
        etaAux = np.tile(etaAux[:,np.newaxis], (1, len(E)));
        DetaAux = np.tile(DetaAux[:,np.newaxis], (1, len(E)));
    #end if
    result = trapezoid(DoSAux*fluxKernel(EAux-etaAux, DetaAux, thresholdLinExpsn, statistics), E, axis=-1);
    return np.reshape(result, shapeOutput);
#end def


def calcDDFlux_DoSxf(ek_, el_, Dphikl_, E, DoS, statistics=1., returnJacobian=False, thresholdLinExpsn=1e-5, **kwargs):
    """
    Computes the carrier flux using a 2nd order accurate (w.r.t. grid step)
    generalized S-G method. Supported for F-D and B-E.

    Parameters
    ----------
    ek_ : float
        Reduced Fermi energy at starting node k. Normalized by kBT. 
    el_ : float
        Reduced Fermi energy at ending node l. Normalized by kBT.
    Dphikl_ : float
        Increment of quasi-Fermi potential, divided by thermal voltage kBT/q.
    E : 1D array of floats
        Energies referred to the valley extreme (minimum particle energy
        of the ensemble). Divided by thermal energy kBT.
    DoS : 1D array of floats
        Corresponding density of states. Divided by N0/kBT, where N0 has
        dimensions of particle density.
    statistics : int, optional
        Fermi-Dirac (1) or Bose-Einstein (-1) statistics.
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
        * ck : float
            Carrier density, divided by N0, at node k.
        * cl : float
            Carrier density, divided by N0, at node l.
    Returns
    -------
        Numerical carrier flux along bond from k to l.

    """
    Dekl_ = el_ - ek_;
    ekl_ = 0.5*(ek_ + el_);
    # (Deta)^{-1} * Integral[deta c(eta)]
    integral = calcIntegralDoSKernel(E, DoS, ekl_, Dekl_, thresholdLinExpsn, statistics);
    j = integral*Dphikl_;

    if returnJacobian:
        dcdeta_k, cl = kwargs['dcdeta_k'], kwargs['cl'];
        # Derivatives w.r.t. quasi-Fermi potential phik and phil
        djdphil, djdphik = np.zeros_like(Dphikl_), np.zeros_like(Dphikl_);
        
        DphiDe = np.zeros_like(Dphikl_);
        DphiDe[maskRegularize == 0] = Dphikl_[maskRegularize == 0]/Dekl_[maskRegularize == 0];
        commonTerm = np.zeros_like(Dphikl_);
        commonTerm[maskRegularize == 0] = (-1)*(1.0 + DphiDe[maskRegularize == 0])*integral[maskRegularize == 0]; 
        commonTerm[maskRegularize > 0] = 0.5*(-1)*(Dekl_ + Dphikl_)[maskRegularize > 0]*dcdeta_k[maskRegularize > 0];
        
        djdphik[maskRegularize == 0] = commonTerm[maskRegularize == 0] + ck_[maskRegularize == 0]*DphiDe[maskRegularize == 0];
        djdphik[maskRegularize > 0] = commonTerm[maskRegularize > 0] - ck_[maskRegularize > 0];
            
        djdphil[maskRegularize == 0] = (-1)*commonTerm[maskRegularize == 0] - cl[maskRegularize == 0]*DphiDe[maskRegularize == 0];
        djdphil[maskRegularize > 0] = commonTerm[maskRegularize > 0] + cl[maskRegularize > 0];

        # Derivatives w.r.t. electrostatic potential Vk and Vl
        z = kwargs['chargeNumber'];
        djdVk, djdVl = np.zeros_like(Dphikl_), np.zeros_like(Dphikl_);

        djdVk[maskRegularize == 0] = -z*( integral[maskRegularize == 0] - ck_[maskRegularize == 0]] )*DphiDe;
        djdVk[maskRegularize > 0] = -commonTerm[maskRegularize > 0];

        djdVl[maskRegularize == 0] = z*( integral[maskRegularize == 0] - cl_[maskRegularize == 0]] )*DphiDe;
        djdVl[maskRegularize > 0] = djdVk[maskRegularize > 0];
        
        return j.copy(), djdphik.copy(), djdphil.copy(), djdVk.copy(), djdVl.copy();
    else:
        return j;
    #end if    
#end def
    

