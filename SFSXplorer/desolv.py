#!/usr/bin/env python3
#
################################################################################
# SFSXplorer                                                                   #
# Scoring Function Space eXplorer                                              #
################################################################################
#
# Class to calculate intermolecular desolvation potential based on
# protein-ligand atomic coordinates in the PDBQT format. It uses
# pair-wise energetic terms of the AutoDock4 Force Field
# (Morris et al., 1998; 2009). Here we explore the Scoring Function Concept
# (Ross et al., 2013; Heck et al., 2017; Bitencourt-Ferreira & de Azevedo Jr.,
# 2019; Veríssimo et al., 2022) to generate models to predict binding affinity
# for protein systems.
#
#
# References:
# Bitencourt-Ferreira G, de Azevedo WF Jr. Exploring the Scoring Function Space.
# Methods Mol Biol. 2019; 2053: 275–281.
#
# Heck GS, Pintro VO, Pereira RR, de Ávila MB, Levin NMB, de Azevedo WF.
# Supervised Machine Learning Methods Applied to Predict
# Ligand-Binding Affinity. Curr Med Chem. 2017; 24(23): 2459–2470.
#
# Morris GM, Goodsell D, Halliday R, Huey R, Hart W, Belew R, Olson A.
# Automated docking using a Lamarckian genetic algorithm and an
# empirical binding free energy function. J Comput Chem. 1998; 19:1639–1662.
#
# Morris GM, Huey R, Lindstrom W, Sanner MF, Belew RK, Goodsell DS, Olson AJ.
# AutoDock4 and AutoDockTools4: Automated docking with
# selective receptor flexibility. J Comput Chem. 2009 Dec; 30(16): 2785–2591.
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

# Define PairwisePotDesol() class
class PairwisePotDesol(object):
    """Class to calculate pairwise potential energy for desolvation based on
    the Autodock4 force field"""

    # Define potential() method
    def potential(self,vol_i,sol_i,vol_j,sol_j,r,m,n,sigma):
        """Method to calculate pairwise potential energy based on the
        Autodock4 force field"""

        # Calculate v(r)
        v = ((vol_i*sol_i) + (vol_j*sol_j))*np.exp(-r**n/(2*sigma**m))

        # Return result
        return v 


def desol_potentialv(vol_i,vol_j,sol_i,sol_j,r,m,n,sigma):
    """Method to calculate pairwise potential energy based on the
    Autodock4 force field, works on 2d arrays of inputs, returns array
    No charges, based on Autodock 4.2 documentation 
    n and m need  to be 2 to match the AD42 scoring function"""

    # Calculate v(r)
    mask = r <= 20
    v = np.zeros_like(r)
    v[mask] = ((vol_i[mask]*sol_j[mask]) + (vol_j[mask]*sol_i[mask]))*np.exp(-r[mask]**n/(2*sigma**m))
    #v[mask] = (vol_i[mask]  * sol_j[mask] + vol_j[mask] * sol_i[mask]) * np.exp(-0.5 * (r[mask] / sigma)**2)
    #print(v)
    # Return result
    return np.sum(v) 

def desol_potentialvv(q_i2d,q_j2d,vol_i,vol_j,sol_i,sol_j,r,m,n,sigma,use_charge_coef):
    """Method to calculate pairwise potential energy based on the
    Autodock4 force field, works on 2d arrays of inputs, returns array
    Uses charges, based on Autodock Vina 1.2 AD42 scoring function code
    n and m need  to be 2 and use_charge_coef needs to be 0.01097
    to match the AD42 scoring function."""

    # Calculate v(r)
    mask = r <= 20
    v = np.zeros_like(r)
    q_j=np.zeros_like(r)
    q_i=np.zeros_like(r)
    q_j[mask]=abs(q_j2d)[mask]
    q_i[mask]=abs(q_i2d)[mask]
    v[mask] = ((vol_i[mask]*(sol_j[mask]+use_charge_coef*q_j[mask])) + (vol_j[mask]*(sol_i[mask]+use_charge_coef*q_i[mask])))*np.exp(-r[mask]**n/(2*sigma**m))
    #print(v)
    # Return result
    return np.sum(v)