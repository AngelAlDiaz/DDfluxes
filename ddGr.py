# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 13:29:22 2025

@author: Angel Alonso Diaz-Burgos
"""
import numpy as np;

def calcDDFlux_Gr(ek_, el_, Dphikl_, ck_, returnJacobian=False, thresholdLinExpsn=1e-5, **kwargs):
    """
    Computes the carrier flux using the generalized S-G integration on an
    approximated F-D distribution function, assuming linear DoS.

    Parameters
    ----------
    ek_ : float
        Reduced Fermi energy at starting node k. Normalized by kBT. 
    el_ : float
        Reduced Fermi energy at ending node l. Normalized by kBT.
    Dphikl_ : float
        Increment of quasi-Fermi potential, divided by thermal voltage kBT/q.
    ck_ : float
        Carrier density at starting node k. Normalized by N0. 
    returnJacobian : bool
        Whether to return the derivative of the flux w.r.t. the Fermi potential
        at both nodes.    
    **kwargs : 
        * dcdeta_k : float
            Derivative of carrier density w.r.t. the reduced Fermi energies,
            divided by N0/VT, at node k.
        * cl : float
            Carrier density, divided by N0, at node l.

    Returns
    -------
        Numerical carrier flux along bond from k to l.

    Notes
    -----
    Denote this function by j. Let ek, el and Dphikl be the quantities as defined
    above for electrons in the CB. Then, the electron flux is simply jn = j(ek, el, Dphikl, nk, ...),
    while the hole (from VB) flux is jp = j(-ek, -el, -Dphikl, pk, ...)
    """
    Dekl_ = el_ - ek_;
    sgnk, sgnl = np.sign(ek_), np.sign(el_);
    maskRegularize = (np.abs(Dekl_) < thresholdLinExpsn);
    aux1 = np.nonzero((sgnk < 0)*(sgnl < 0))[0];
    aux2 = np.nonzero((sgnk < 0)*(sgnl > 0))[0];
    aux3 = np.nonzero((sgnk > 0)*(sgnl < 0))[0];
    aux4 = np.nonzero((sgnk >= 0)*(sgnl >= 0))[0];
    A, B, C = 4.0/27.0, 14.0/9.0, 1.0/6.0;
    integral = np.ones_like(Dekl_);
    integral[aux1] = np.exp(el_[aux1]) - np.exp(ek_[aux1]) - A*np.exp(1.5*el_[aux1]) + A*np.exp(1.5*ek_[aux1]);
    integral[aux2] = B*el_[aux2] + C*el_[aux2]**3 - np.exp(ek_[aux2]) + np.exp(-el_[aux2]) + A*np.exp(1.5*ek_[aux2]) - A*np.exp(-1.5*el_[aux2]);
    integral[aux3] = -B*ek_[aux3] - C*ek_[aux3]**3 + np.exp(el_[aux3]) - np.exp(-ek_[aux3]) - A*np.exp(1.5*el_[aux3]) + A*np.exp(-1.5*ek_[aux3]);
    integral[aux4] = B*(el_[aux4]-ek_[aux4]) + C*(el_[aux4]**3-ek_[aux4]**3) - np.exp(-ek_[aux4]) + np.exp(-el_[aux4]) - A*np.exp(-1.5*el_[aux4]) + A*np.exp(-1.5*ek_[aux4]);
    integral[maskRegularize > 0] = ck_[maskRegularize > 0];
    integral[maskRegularize == 0] *= 1.0/Dekl_[maskRegularize == 0];
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
        return j.copy();
    #end if

#end def
