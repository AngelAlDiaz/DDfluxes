# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 13:29:22 2025

@author: Angel Alonso Diaz-Burgos
"""
import numpy as np;

def BernouilliGenFunction(x, returnDerivative=False):
    """
    Computes the Bernouilli's generating function appearing in the classical
    Scharfetter-Gummel method.

    Parameters
    ----------
    x : numpy.array(float)
        Dimensionless argument of the function.
    returnDerivative : bool, OPTIONAL
        Whether to return the derivative.
    Returns
    -------
    numpy.array(float)
        Array with the values of the Bernoulli's function.
    """
    result = np.ones_like(x, float);
    safeIndices = np.nonzero(np.abs(x) > 1e-10)[0];
    result[safeIndices] = x[safeIndices]/(np.exp(x[safeIndices]) - 1.0);
    if returnDerivative:
        deriv = -0.5*np.ones_like(x, float);
        deriv[safeIndices] = result[safeIndices]*(1.0/x[safeIndices] - np.exp(x[safeIndices]));
        return result, deriv;
    else: return result;
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
    if returnJacobian: B_zDV, dB_zDV, BzDV, dBzDV = BernouilliGenFunction(-zDVkl_, returnDerivative=True), BernouilliGenFunction(zDVkl_, returnDerivative=True);
    else: B_zDV, BzDV = BernouilliGenFunction(-zDVkl_), BernouilliGenFunction(zDVkl_);
    j = -avgDE_*(B_zDV*cl_ - BzDV*ck_);
    
    if returnJacobian:
        dcdeta_k, dcdeta_l = kwargs['dcdeta_k'], kwargs['dcdeta_l'];
        dlnc = np.log(cl_/ck_);
        
        # Derivative of the averaged diff. enhancement w.r.t. eta (requires 2nd derivative of particle density, 'd2cdeta2')
        DEk, DEl = ck_/dcdeta_k, cl_/dcdeta_l; # Diff. enhancment at nodes k and l
        maskRegularize = (np.abs(dlnc) < thresholdLinExpsn);
        davgDEdeta_k, davgDEdeta_l = np.zeros_like(avgDE_, float), np.zeros_like(avgDE_, float);
        safeIndices = np.nonzero( maskRegularize == 0 )[0];
        regularize = np.nonzero( maskRegularize )[0];
        davgDEdeta_k[safeIndices] = (-1.0 + avgDE_[safeIndices]/DEk[safeIndices])/dlnc[safeIndices];
        davgDEdeta_l[safeIndices] = (1.0 - avgDE_[safeIndices]/DEl[safeIndices])/dlnc[safeIndices];
        davgDEdeta_k[regularize] = 0.5 * (1.0 - DEk[regularize]**2 * kwargs['d2cdeta2'][regularize] / ck_[regularize]);
        davgDEdeta_l[regularize] = -davgDEdeta_k[regularize];

        # Quantities that are used twice (to prevent from computing them twice)
        aux = (-1)*(cl_*dB_zDV + ck_*dBzDV)*(zDVkl_/avgDE);
        jDE = j/avgDE_;

        # Derivatives w.r.t. quasi-Fermi potential phik and phil
        djdphik = jDE*davgDEdeta_k + avgDE_*dcdeta_k*BzDV + aux*davgDEdeta_k;
        djdphik *= -1;
        
        djdphil = jDE*davgDEdeta_l - avgDE_*dcdeta_l*B_zDV + aux*davgDEdeta_l;
        djdphil *= -1;

        # Derivatives w.r.t. electrostatic potential Vk and Vl
        z = kwargs['chargeNumber'];
        common = (dB_zDV*cl_ + dBzDV*ck_);
        djdVk = z*( djdphik - common );
        djdVl = z*( djdphil + common );
        
        return j.copy(), djdphik.copy(), djdphil.copy(), djdVk.copy(), djdVl.copy();
    else:
        return j.copy();
    #end if
#end def


