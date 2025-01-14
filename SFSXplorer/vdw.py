#!/usr/bin/env python3
#
################################################################################
# SFSXplorer                                                                   #
# Scoring Function Space eXplorer                                              #
################################################################################
#
# Class to calculate intermolecular van der Waals potential based on 
# atomic coordinates in the PDBQT format. It calculates the potential energy 
# based on Assisted Model Building with Energy Refinement (AMBER) force field 
# (Cornell et al., 1995) using the energy terms derived from the AutoDock4 
# (Morris et al., 1998; 2009). The traditional 12/6 potential energy is 
# modified to adapt to the data used to train the scoring function. We vary the
# exponents (m and n parameters) to scan the Scoring Function Space 
# (Ross et al., 2013; Heck et al., 2017; Bitencourt-Ferreira & de Azevedo Jr., 
# 2019; Veríssimo et al., 2022) to find the van der Waals potential adequate 
# to a targeted protein system.
#
# References:
# Bitencourt-Ferreira G, de Azevedo WF Jr. Exploring the Scoring Function Space. 
# Methods Mol Biol. 2019; 2053: 275-281.
#
# Cornell WD, Cieplak P, Bayly CI, Gould IR, Merz KM Jr, Ferguson DM, Spellmeyer
#  DC, Fox T, Caldwell JW, Kollman PA (1995). A Second Generation Force Field 
# for the Simulation of Proteins, Nucleic Acids, and Organic Molecules. 
# J Am Chem Soc. 1995; 117 (19): 5179–5197.
#
# Heck GS, Pintro VO, Pereira RR, de Ávila MB, Levin NMB, de Azevedo WF. 
# Supervised Machine Learning Methods Applied to Predict Ligand-Binding 
# Affinity. Curr Med Chem. 2017;24(23):2459–2470.
#
# Morris GM, Goodsell D, Halliday R, Huey R, Hart W, Belew R, Olson A. 
# Automated docking using a Lamarckian genetic algorithm and an empirical 
# binding free energy function. J Comput Chem. 1998; 19:1639–1662.
#
# Morris GM, Huey R, Lindstrom W, Sanner MF, Belew RK, Goodsell DS, Olson AJ. 
# AutoDock4 and AutoDockTools4: Automated docking with 
# selective receptor flexibility. J Comput Chem. 2009; 30(16): 2785–2791.
#
# Ross GA, Morris GM, Biggin PC. One Size Does Not Fit All: The Limits of 
# Structure-Based Models in Drug Discovery. J Chem Theory Comput. 2013; 9(9): 
# 4266–4274.  
#
# Veríssimo GC, Serafim MSM, Kronenberger T, Ferreira RS, Honorio KM, 
# Maltarollo VG. Designing drugs when there is low data availability: 
# one-shot learning and other approaches to face the issues of a 
# long-term concern. Expert Opin Drug Discov. 2022; 17(9): 929–947. 
# 
################################################################################
# Dr. Walter F. de Azevedo, Jr.                                                #
# https://azevedolab.net/                                                      #
# January 12, 2023                                                             #
################################################################################
#
# Import section
import numpy as np


#VECTORIZED VERSION FOR MATRIXES OF VDW_PARAMS and R
"""Method to calculate pairwise potential energy based on 
matrixes of r, reqm_i, epsilon_i, reqm_j and epsilon_j values"""

def vdw_potentialv(reqm_i, epsilon_i, reqm_j, epsilon_j, r, n, m):
    reqm = 0.5 * (reqm_i + reqm_j)
    epsilon = np.sqrt(epsilon_i * epsilon_j)
    cm = np.zeros_like(r)
    cn = np.zeros_like(r)
    v = np.zeros_like(r)
    mask = r <= 12
    cm[mask] = (n / (n - m)) * epsilon[mask] * (reqm[mask] ** m)
    cn[mask] = (m / (n - m)) * epsilon[mask] * (reqm[mask] ** n)
    v[mask] = cn[mask] / (r[mask] ** n) - cm[mask] / (r[mask] ** m)
    return np.sum(v)*0.1662

def vdw_potentialvv(reqm_i, epsilon_i, reqm_j, epsilon_j, r, n, m,ehb_i,ehb_j):
    hbchk = np.zeros_like(ehb_i)
    reqm = np.zeros_like(r)
    epsilon = np.zeros_like(r)
    mask = (r <= 12)
    hbchk[mask] = ehb_i[mask] * ehb_j[mask]
    reqm[mask] = 0.5 * (reqm_i[mask] + reqm_j[mask])
    epsilon[mask] = np.sqrt(epsilon_i[mask] * epsilon_j[mask])
    cm = np.zeros_like(r)
    cn = np.zeros_like(r)
    v = np.zeros_like(r)
    mask2 = (hbchk >= 0) & (r <= 12)
    cm[mask2] = (n / (n - m)) * epsilon[mask2] * (reqm[mask2] ** m)
    cn[mask2] = (m / (n - m)) * epsilon[mask2] * (reqm[mask2] ** n)
    v[mask2] = cn[mask2] / (r[mask2] ** n) - cm[mask2] / (r[mask2] ** m)
    return np.sum(v) *0.1662

def vdw_potentialvvs(reqm_i, epsilon_i, reqm_j, epsilon_j, r, n, m,ehb_i,ehb_j,smooth,cap):
    hbchk = np.zeros_like(ehb_i)
    reqm = np.zeros_like(r)
    epsilon = np.zeros_like(r)
    mask = (r <= 12)
    hbchk[mask] = ehb_i[mask] * ehb_j[mask]
    reqm[mask] = 0.5 * (reqm_i[mask] + reqm_j[mask])
    epsilon[mask] = np.sqrt(epsilon_i[mask] * epsilon_j[mask])
    cm = np.zeros_like(r)
    cn = np.zeros_like(r)
    v = np.zeros_like(r)
    mask2 = (hbchk >= 0) & (r <= 12)
    cm[mask2] = (n / (n - m)) * epsilon[mask2] * (reqm[mask2] ** m)
    cn[mask2] = (m / (n - m)) * epsilon[mask2] * (reqm[mask2] ** n)
    #v[mask2] = cn[mask2] / (r[mask2] ** n) - cm[mask2] / (r[mask2] ** m)
    #Smooth both on attractive and repulsive sides
    #smooth_r = np.where(r > reqm + 0.5, r - 0.5, np.where(r < reqm - 0.5, r + 0.5, r))
    #Smooth only on repulsive side, flat bottom
    smooth_r = np.where(r < (reqm - smooth), r + smooth, np.where(r < reqm, reqm, r))
    #Smooth also attractive side
    smooth_r2 = np.where(r > (reqm + smooth), r - smooth, np.where(r > reqm, reqm, smooth_r))
    # Calculate v using the updated smoothed r values
    v[mask2] = cn[mask2] / (smooth_r2[mask2] ** n) - cm[mask2] / (smooth_r2[mask2] ** m)
    v = np.clip(v, a_min=None, a_max=cap)
    return np.sum(v)*0.1662

# Define PairwisePot() class
class PairwisePot(object):
    """Class to calculate pairwise potential energy for van der Waals 
        interactions based on the assisted Model Building with Energy Refinement
        (AMBER) force field (Cornell et al., 1995)"""
        
    # Define potential() method (it is better to follow n=12,m=6) 
    def potential(self,reqm_i,epsilon_i,reqm_j,epsilon_j,r,n,m):
       
        reqm = 0.5 * (reqm_i + reqm_j)
        epsilon = (epsilon_i * epsilon_j) ** 0.5
        if n != m:
            cm = (n / (n - m)) * epsilon * (reqm ** m)
            cn = (m / (n - m)) * epsilon * (reqm ** n)
            v = cn / (r ** n) - cm / (r ** m)
        else:
            cm = None
            cn = None
            v = None
        return cn, cm, v
    

