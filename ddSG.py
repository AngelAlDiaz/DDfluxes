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

def calcDDFlux_SG(ek_, el_, zDVkl_, returnJacobian=False):
    """
    Computes the carrier flux according to the classical Scharfetter-Gummel
    method, valid for Maxwell-Boltzmann statistics.

    Parameters
    ----------
    ek_ : float
        Reduced Fermi energy at starting node k. Normalized by kBT. 
    el_ : float
        Reduced Fermi energy at ending node l. Normalized by kBT.
    zDVkl_ : float
        Increment of electrostatic potential, multiplied by charge number z and
        divided by thermal voltage kBT/q.
    returnJacobian : bool
        Whether to return the derivative of the flux w.r.t. the Fermi potential
        at both nodes.
    
    Returns
    -------
        Numerical carrier flux along bond from k to l.
    """        
    expk, expl = np.exp(ek_), np.exp(el_);
    if returnJacobian: B_zDV, dB_zDV, BzDV, dBzDV = BernouilliGenFunction(-zDVkl_, returnDerivative=True), BernouilliGenFunction(zDVkl_, returnDerivative=True);
    else: B_zDV, BzDV = BernouilliGenFunction(-zDVkl_), BernouilliGenFunction(zDVkl_);
    j = -(B_zDV*expl - BzDV*expk);
        
    if returnJacobian:
        DcDphik, DcDphil = (-1)*expk, (-1)*expl;
        # Derivatives w.r.t. quasi-Fermi potential 
        djdphik = BzDV*DcDphik;
        djdphil = (-1)*B_zDV*DcDphil;
        # Derivatives w.r.t. electrostatic potential
        z = kwargs['chargeNumber'];
        djdVk = -z*( dB_zDV*expl + (dBzDV + BzDV)*expk );
        djdVl = z*( (dB_zDV + B_zDV)*expl + dBzDV*expk );

        return j.copy(), djdphik.copy(), djdphil.copy(), djdVk.copy(), djdVl.copy();
    else:
        return j.copy();
    #end if
#end def

