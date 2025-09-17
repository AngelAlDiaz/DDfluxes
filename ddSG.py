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
    B_zDV, BzDV = BernouilliGenFunction(-zDVkl_), BernouilliGenFunction(zDVkl_);
    j = -(B_zDV*expl - BzDV*expk);
        
    if returnJacobian:
        DcDphik, DcDphil = (-1)*expk, (-1)*expl;
        djdphik = BzDV*DcDphik;
        djdphil = (-1)*B_zDV*DcDphil;
        return j.copy(), djdphik.copy(), djdphil.copy();
    else:
        return j.copy();
    #end if
#end def
