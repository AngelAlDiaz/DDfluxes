# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 13:29:22 2025

@author: Angel Alonso Diaz-Burgos
"""
import numpy as np;

def calcDiffFlux_2DBE(ek_, el_, etaTh, N, returnJacobian=False, **kwargs):
    """
    Computes the diffusion flux using the generalized S-G integration on an
    approximated equation of state for a 2D boson gas with parabolic valley.

    Parameters
    ----------
    ek_ : float
        Reduced Fermi energy at starting node k. Normalized by kBT. 
    el_ : float
        Reduced Fermi energy at ending node l. Normalized by kBT.
    etaTh : float
        Parameter related to the approximation of the equation of state.
    N : float
        Parameter related to the approximation of the equation of state.
    returnJacobian : bool
        Whether to return the derivative of the flux w.r.t. the Fermi potential
        at both nodes.    
    **kwargs : 
        * ck : float
            Carrier density, divided by N0, at node k.
        * cl : float
            Carrier density, divided by N0, at node l.

    Returns
    -------
        Numerical carrier flux along bond from k to l.

    """
    sgnk, sgnl = np.sign(ek_-etaTh), np.sign(el_-etaTh);
    aux1 = np.nonzero((sgnk < 0)*(sgnl < 0))[0];
    aux2 = np.nonzero((sgnk < 0)*(sgnl > 0))[0];
    aux3 = np.nonzero((sgnk > 0)*(sgnl < 0))[0];
    aux4 = np.nonzero((sgnk >= 0)*(sgnl >= 0))[0];
    
    n = np.arange(1, N+1);
    expnk = np.exp(n[np.newaxis, :]*ek_[:, np.newaxis]);
    expnl = np.exp(n[np.newaxis, :]*el_[:, np.newaxis]);
    lnTermk = ek_*(1.0+np.log(-ek_));
    lnTerml = el_*(1.0+np.log(-el_));
    A, B = np.sum( np.exp(n*etaTh)/n**2 ), etaTh*(1.0 + np.log(-etaTh));
    j = np.ones_like(ek_);
    j[aux1] = np.sum( (expnl[aux1]-expnk[aux1])/n[np.newaxis,:]**2, axis=-1 );
    j[aux2] = A - np.sum( expnk[aux2]/n[np.newaxis,:]**2, axis=-1 ) - (lnTerml[aux2] - B);
    j[aux3] = np.sum( expnl[aux3]/n[np.newaxis,:]**2, axis=-1 ) - A - (B - lnTermk[aux3]);
    j[aux4] =  -(lnTerml[aux4] - lnTermk[aux4]);
    j *= (-1);
    if returnJacobian:
        ck, cl = kwargs['ck'], kwargs['cl'];

        djdphik = (-1)*ck;
        
        djdphil = cl;
        
        return j.copy(), djdphik.copy(), djdphil.copy();
    else:
        return j.copy();
    #end if
#end def