#!/usr/bin/env python3
#
################################################################################
# SFSXplorer                                                                   #
# Scoring Function Space eXplorer                                              #
################################################################################
#
# Class to calculate intermolecular hydrogen-bond potential based on 
# atomic coordinates in the PDBQT format. It calculates the potential energy 
# based on Assisted Model Building with Energy Refinement (AMBER) force field 
# (Cornell et al., 1995) using the energy terms derived from the AutoDock4 
# (Morris et al., 1998; 2009). The traditional 12/10 potential energy is 
# modified to adapt to the data used to train the scoring function. We vary the
# exponents (m and n parameters) to scan the Scoring Function Space 
# (Ross et al., 2013; Heck et al., 2017; Bitencourt-Ferreira & de Azevedo Jr., 
# 2019; Veríssimo et al., 2022) to find the hydrogen-bond potential adequate 
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
# Define class PairwisePotHB()
import numpy as np


class PairwisePotHB(object):
    """Class to calculate pairwise potential energy for hydrogen bonds based on 
        the assisted Model Building with Energy Refinement (AMBER) force 
        field (Cornell et al., 1995)"""
    
    # Define potential() method
    # It is better to follow n=12,m=10 
    def potential(self,reqm_i,epsilon_i,reqm_j,epsilon_j,r,n,m):

        # To obtain the Rij value for H-bonding atoms
        if reqm_i > reqm_j:
            reqm = reqm_i
        elif reqm_i < reqm_j:
            reqm = reqm_j
        else:
            reqm = reqm_i
        
        #  To obtain the epsilon value for H-bonding atoms
        if epsilon_i > epsilon_j:
            epsilon = epsilon_i
        elif epsilon_i < epsilon_j:
            epsilon = epsilon_j
        else:
            epsilon = epsilon_i
        
        # Calculate cm and cn parameters if n != m
        if n != m:
            cm = (n/(n-m))*epsilon*reqm**m 
            cn = (m/(n-m))*epsilon*reqm**n
        else:
            return None,None,None
                
        # Calculate v(r)     
        v = cn/r**n - cm/r**m
        
        # Return results
        return cn,cm,v
    

def hb_potentialv(reqm_i, epsilon_i, reqm_j, epsilon_j, r, n, m):
    # Select max Rij for each pair of atoms
    reqm = np.maximum(reqm_i, reqm_j) # ideally should replace this with sum reqm_i and reqm_j and divide by 2
    # Select max epsilon for each pair of atoms
    epsilon = np.maximum(epsilon_i, epsilon_j)
    cm = np.zeros_like(r)
    cn = np.zeros_like(r)
    v = np.zeros_like(r)
    # Calculate cm and cn for all atom pairs in array
    mask = r <= 8
    cm[mask] = (n / (n - m)) * epsilon[mask] * (reqm[mask] ** m)
    cn[mask] = (m / (n - m)) * epsilon[mask] * (reqm[mask] ** n)
    v[mask] = cn[mask] / (r[mask] ** n) - cm[mask] / (r[mask] ** m)
    # Calculate sum of v over all atom pairs
    return np.sum(v) * 0.1209

def hb_potentialvv(reqm_i, epsilon_i, reqm_j, epsilon_j, r, n, m):
    # Use sum of HBradii and multiply epsilons
    reqsum = np.zeros_like(reqm_i)
    epsmult = np.zeros_like(epsilon_i)
    mask = (r <= 8)
    reqsum[mask] = np.maximum(reqm_i[mask], reqm_j[mask])  # Max because HD HBradii is 0
    epsmult[mask] = -1 * epsilon_i[mask] * epsilon_j[mask] #change sign, because eps for HD is -1
    cm = np.zeros_like(r)
    cn = np.zeros_like(r)
    v = np.zeros_like(r)
    #Only calculate for dist<=8A and hbondable atom pairs (HD + acceptor)
    mask2= epsmult > 0
    # Calculate cm and cn for all atom pairs in array
    cm[mask2] = (n / (n - m)) * epsmult[mask2] * (reqsum[mask2] ** m)
    cn[mask2] = (m / (n - m)) * epsmult[mask2] * (reqsum[mask2] ** n)
    v[mask2] = cn[mask2] / (r[mask2] ** n) - cm[mask2] / (r[mask2] ** m)
    # Calculate sum of v over all atom pairs
    #mask3=v<0
    #return np.sum(v[mask3])
    return np.sum(v) * 0.1209

def hb_potentialvvs(reqm_i, epsilon_i, reqm_j, epsilon_j, r, n, m,smooth,cap):
    #Add smoothing
    reqsum = np.zeros_like(reqm_i)
    epsmult = np.zeros_like(epsilon_i)
    mask = (r <= 8)
    reqsum[mask] = np.maximum(reqm_i[mask], reqm_j[mask])  # Max because HD HBradii is 0
    epsmult[mask] = -1 * epsilon_i[mask] * epsilon_j[mask] #change sign, because eps for HD is -1
    cm = np.zeros_like(r)
    cn = np.zeros_like(r)
    v = np.zeros_like(r)
    #Only calculate for dist<=8A and hbondable atom pairs (HD + acceptor)
    mask2= epsmult > 0
    # Calculate cm and cn for all atom pairs in array
    cm[mask2] = (n / (n - m)) * epsmult[mask2] * (reqsum[mask2] ** m)
    cn[mask2] = (m / (n - m)) * epsmult[mask2] * (reqsum[mask2] ** n)
    #v[mask2] = cn[mask2] / (r[mask2] ** n) - cm[mask2] / (r[mask2] ** m)
    # Apply smoothing of 0.5, recalculate values in r
    #smooth_r = np.where(r > reqsum + 0.5, r - 0.5, np.where(r < reqsum - 0.5, r + 0.5, r))
    #Smooth only on repulsive side, flat bottom
    smooth_r = np.where(r < (reqsum - smooth), r + smooth, np.where(r < reqsum, reqsum, r))
    # Smooth attractive side too
    smooth_r2 = np.where(r > (reqsum + smooth), r - smooth, np.where(r > reqsum, reqsum, smooth_r))
    # Calculate v using the updated smoothed r values
    v[mask2] = cn[mask2] / (smooth_r2[mask2] ** n) - cm[mask2] / (smooth_r2[mask2] ** m)
    # Calculate sum of v over all atom pairs
    #mask3=v<0
    #return np.sum(v[mask3])
    v = np.clip(v, a_min=None, a_max=cap)
    return np.sum(v) * 0.1209