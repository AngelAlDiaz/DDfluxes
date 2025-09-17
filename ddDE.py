# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 13:29:22 2025

@author: Angel Alonso Diaz-Burgos
"""
import numpy as np;

def BernouilliGenFunction(x):
    """
    Computes the Bernouilli's generating function appearing in the classical
    Scharfetter-Gummel method.

    Parameters
    ----------
    x : numpy.array(float)
        Dimensionless argument of the function.

    Returns
    -------
    numpy.array(float)
        Array with the values of the Bernoulli's function.
    """
    result = np.ones_like(x, float);
    auxMask = (np.abs(x) > 1e-10);
    result[auxMask > 0] = x[auxMask > 0]/(np.exp(x[auxMask > 0]) - 1.0);
    return result;
#end def

def calcDDFlux_DE(ck_, cl_, zDVkl_, avgDE_, returnJacobian=False, thresholdLinExpsn=1e-5, **kwargs):
    """
    Computes the carrier flux according to the Diffusion-Enhanced scheme.

    Parameters
    ----------
    ck_ : float
        Carrier density at starting node k. Normalized by N0. 
    cl_ : float
        Carrier density at ending node l. Normalized by N0.
    zDVkl_ : float
        Increment of electrostatic potential, multiplied by charge number z and
        divided by thermal voltage kBT/q.
    avgDE_ : float
        Averaged Diffusion enhancement factor. Must be provided after
        regularization, if required.
    returnJacobian : bool
        Whether to return the derivative of the flux w.r.t. the Fermi potential
        at both nodes.    
    thersholdLinExpsn : float
        Threshold to regularize singular expressions in case denominator goes
        to zero. Default is 1E-5.
    **kwargs : 
        * dcdeta_k : float
            Derivative of carrier density w.r.t. the reduced Fermi energies,
            divided by N0/VT, at node k.
        * dcdeta_l : float
            Derivative of carrier density w.r.t. the reduced Fermi energies,
            divided by N0/VT, at node l.
        
    Returns
    -------
        Numerical carrier flux along bond from k to l.
    """
    zDVkl_ /= avgDE_;    
    B_zDV, BzDV = BernouilliGenFunction(-zDVkl_), BernouilliGenFunction(zDVkl_);
    j = -avgDE_*(B_zDV*cl_ - BzDV*ck_);
    
    if returnJacobian:
        dcdeta_k, dcdeta_l = kwargs['dcdeta_k'], kwargs['dcdeta_l'];
        dlnc = np.log(cl_/ck_);
        
        DEk, DEl = ck_/dcdeta_k, cl_/dcdeta_l; # Diff. enhancement at nodes k and l
        # Derivative of the averaged diff. enhancement w.r.t. eta
        davgDEdeta_k, davgDEdeta_l = (-1.0+avgDE_/DEk)/dlnc, (1.0-avgDE_/DEl)/dlnc;
        davgDEdeta_k[np.abs(dlnc) < thresholdLinExpsn] = 0.0;
        davgDEdeta_l[np.abs(dlnc) < thresholdLinExpsn] = 0.0;
        # Derivatives of Bernoulli functions w.r.t. their argument, i.e. B'(x) := dB(x)/dx
        expp = np.exp(zDVkl_);
        exp_ = 1.0/expp;
        dBzDV, dB_zDV = -0.5*np.ones_like(zDVkl_), -0.5*np.ones_like(zDVkl_); # Limit of Derivative when x->0 is -0.5
        maskRegularize = (np.abs(zDVkl_) < thresholdLinExpsn);
        dBzDV[maskRegularize == 0] = (expp[maskRegularize == 0]-1-zDVkl_[maskRegularize == 0]*expp[maskRegularize == 0])/(expp[maskRegularize == 0]-1)**2;
        dB_zDV[maskRegularize == 0] = (exp_[maskRegularize == 0]-1+zDVkl_[maskRegularize == 0]*exp_[maskRegularize == 0])/(exp_[maskRegularize == 0]-1)**2;
        # Quantities that are used twice (to prevent from computing them twice)
        aux = (-1)*avgDE_*(cl*dB_zDV + ck_*dBzDV)*(zDVkl_/avgDE_);
        jk = j/avgDE_;
        
        djdphik = jk*davgDEdeta_k + avgDE_*dcdeta_k*BzDV + aux*davgDEdeta_k;
        djdphik *= -1;
        
        djdphil = jk*davgDEdeta_l - avgDE_*dcdeta_l*B_zDV + aux*davgDEdeta_l;
        djdphil *= -1;
        
        return j.copy(), djdphik.copy(), djdphil.copy();
    else:
        return j.copy();
    #end if
#end def

