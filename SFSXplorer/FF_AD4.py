#!/usr/bin/env python3
#
################################################################################
# SFSXplorer                                                                   #
# Scoring Function Space eXplorer                                              #
################################################################################
#
# Class to calculate several intermolecular potentials based on
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
from SFSXplorer import vdw as vd
from SFSXplorer import hb as hb
from SFSXplorer import desolv as ds1
from SFSXplorer import elec as e1

# Define InterMol() class
class InterMol(object):
    """Class to calculate intermolecular potential based on AutoDock4 force
    field."""
    
    # Define constructor method
    def __init__(self,ad4_par_file):
        """Constructor method"""
        
        # Set up attributes
        self.ad4_par_file = ad4_par_file    # AutoDock4 parameter file
        self.n_tors = 0                     # Set up zero to n_tors 
                                            # (number of torsions)
                                            # to be read from TORSDOF in 
                                            # lig.pdbqt        
    # Define read_AD4_bound() method
    def read_AD4_bound(self):
        """Method to read AD4.1_bound.data file and return a list"""
        
        # Set up empty list
        self.ad4_list = []
        
        # Try to open self.ad4_par_file
        try:
            fo1 = open(self.ad4_par_file,"r")
        except IOError:
            print("\n I can't find ",self.ad4_par_file," file.")
            return
            
        # Looping through fo1
        for line in fo1:
            if line[0:8] == "atom_par":
                # print(line)
                self.ad4_list.append(line)
        
        # Close file
        fo1.close()
        
        # Return list
        return self.ad4_list
    

    # Define get_atom_par_array()
    def get_atom_par_array(self,par,lig_list,rec_list):
        """Method to retrieve van der Waals parameters for each atom pair"""
        lig_coords = np.zeros((len(lig_list), 3))
        lig_type = np.empty(len(lig_list),dtype=object)
        lig_charge = np.zeros((len(lig_list)))
        rec_coords = np.zeros((len(rec_list), 3))
        rec_type = np.empty(len(rec_list),dtype=object)
        rec_charge = np.zeros((len(rec_list)))
        #Fill arrays with atomic coordinates, types and charges
        for i, line2 in enumerate(lig_list):
            lig_coords[i] = np.array([float(line2[30:38]), float(line2[38:46]), float(line2[46:54])])
            lig_type[i] = line2[77:79]
            lig_charge[i] = float(line2[66:75])
        for i, line2 in enumerate(rec_list):
            rec_coords[i] = np.array([float(line2[30:38]), float(line2[38:46]), float(line2[46:54])])
            rec_type[i] = line2[77:79]
            rec_charge[i] = float(line2[66:75])
        return(lig_coords,lig_type,lig_charge,rec_coords,rec_type,rec_charge)
    
    # Define vectorized_dist() method
    def vectorized_dist(self,lig_coords,rec_coords):
        """Method to calculate Euclidian distance
        in a vectorized manner using broadcasting,
        takes array of x,y,z coords for lig and rec as input"""
        # Calculate Euclidian distance
        dist_pairs = np.sqrt(np.sum((lig_coords[:, np.newaxis, :] - rec_coords[np.newaxis, :, :3])**2, axis=2))
        # Return distance
        return dist_pairs
    
    #Define get_atom_par_array_VDW using dictionary and broadcasting
    def get_atom_par_array_VDW(self, par, lig_type, rec_type):
        """Method to retrieve van der Waals parameters for each atom pair of the complex and store them in arrays, takes a 1D array of lig and rec atom_ypes as input"""
        # Create a dictionary that maps atom types to their corresponding parameters
        par_dict = {}
        for line in par:
            atom_type = line[9:11]
            if atom_type not in par_dict:
                par_dict[atom_type] = {'rvdw': float(line[16:20]), 'evdw': float(line[21:27])}
    
        # Use numpy arrays to store the VDW parameters using broadcasting and list comprehension
        ri=np.array([par_dict[x]['rvdw'] for x in lig_type])
        rj=np.array([par_dict[x]['rvdw'] for x in rec_type])
        ei=np.array([par_dict[x]['evdw'] for x in lig_type])
        ej=np.array([par_dict[x]['evdw'] for x in rec_type])
        rvdw_i= np.transpose(np.broadcast_to(ri,(len(rj),)+ri.shape))
        rvdw_j = np.broadcast_to(rj,(len(ri),)+rj.shape)
        evdw_i= np.transpose(np.broadcast_to(ei,(len(ej),)+ei.shape))
        evdw_j = np.broadcast_to(ej,(len(ei),)+ej.shape)        

        return(rvdw_i,rvdw_j,evdw_i,evdw_j)
    
    #Version using dictionaries:
    def get_atom_par_array_HB(self, par, lig_type, rec_type):
        """Method to retrieve van der Waals parameters for each atom pair of the complex and store them in arrays, takes a 1D array of lig and rec atom_ypes as input"""
        # Create a dictionary that maps atom types to their corresponding parameters
        par_dict = {}
        for line in par:
            atom_type = line[9:11]
            if atom_type not in par_dict:
                par_dict[atom_type] = {'rhb': float(line[46:51]), 'ehb': float(line[51:56])}
        # for i, atom_i in enumerate(lig_type):
        #     for j, atom_j in enumerate(rec_type):
        #         print("new",atom_i,par_dict[atom_i]['rhb'],par_dict[atom_i]['ehb'],atom_j,par_dict[atom_j]['rhb'],par_dict[atom_j]['ehb'])
        # Use numpy arrays to store the hb parameters using broadcasting and list comprehension
        ri=np.array([par_dict[x]['rhb'] for x in lig_type])
        rj=np.array([par_dict[x]['rhb'] for x in rec_type])
        ei=np.array([par_dict[x]['ehb'] for x in lig_type])
        ej=np.array([par_dict[x]['ehb'] for x in rec_type])
        rhb_i= np.transpose(np.broadcast_to(ri,(len(rj),)+ri.shape))
        rhb_j = np.broadcast_to(rj,(len(ri),)+rj.shape)
        ehb_i= np.transpose(np.broadcast_to(ei,(len(ej),)+ei.shape))
        ehb_j = np.broadcast_to(ej,(len(ei),)+ej.shape)        

        return(rhb_i,rhb_j,ehb_i,ehb_j)
    
    def get_atom_par_array_desolv(self, par, lig_type, rec_type):
        """Method to retrieve van der Waals parameters for each atom pair of the complex and store them in arrays, takes a 1D array of lig and rec atom_ypes as input"""
        # Create a dictionary that maps atom types to their corresponding parameters
        par_dict = {}
        for line in par:
            atom_type = line[9:11]
            if atom_type not in par_dict:
                par_dict[atom_type] = {'v': float(line[27:36]), 's': float(line[36:46])}
        # for i, atom_i in enumerate(lig_type):
        #     for j, atom_j in enumerate(rec_type):
        #         print("new",atom_i,par_dict[atom_i]['rhb'],par_dict[atom_i]['ehb'],atom_j,par_dict[atom_j]['rhb'],par_dict[atom_j]['ehb'])
        # Use numpy arrays to store the hb parameters using broadcasting and list comprehension
        vi=np.array([par_dict[x]['v'] for x in lig_type])
        vj=np.array([par_dict[x]['v'] for x in rec_type])
        si=np.array([par_dict[x]['s'] for x in lig_type])
        sj=np.array([par_dict[x]['s'] for x in rec_type])

        v_i= np.transpose(np.broadcast_to(vi,(len(vj),)+vi.shape))
        v_j = np.broadcast_to(vj,(len(vi),)+vj.shape)
        s_i= np.transpose(np.broadcast_to(si,(len(sj),)+si.shape))
        s_j = np.broadcast_to(sj,(len(si),)+sj.shape)        

        return(v_i,v_j,s_i,s_j)

    def get_atom_par_array_elec(self, q_i,q_j):
        """Method to retrieve van der Waals parameters for each atom pair of the complex and store them in arrays, takes a 1D array of lig and rec atom_ypes as input"""
        # Create a dictionary that maps atom types to their corresponding parameters
        q_i_2d= np.transpose(np.broadcast_to(q_i,(len(q_j),)+q_i.shape))
        q_j_2d = np.broadcast_to(q_j,(len(q_i),)+q_j.shape)
        return(q_i_2d,q_j_2d)
    

        
    # Define read_PDBQT() method
    def read_PDBQT(self,file_in):
        """Method to read PDBQT file"""
        
        # Set up empty for atom lines
        atom_list = []
        
        # Try to open PDBQT file
        try:
            fo1 = open(file_in,"r")
        except IOError:
            print("\nI can't find ",file_in," file.")
            return atom_list
            
        # Looping through fo1
        for line in fo1:
            if line[0:6] == "HETATM" or line[0:6] == "ATOM  ":
                atom_list.append(line)
            elif line[0:7] == "TORSDOF":
                self.n_tors = int(line[7:])
                    
        # Close file
        fo1.close()
        
        # Return results
        return atom_list
    
    # Define read_torsion() method
    def read_torsion(self,name_dir):
        """Method to return number of torsion angles (TORSDOF)"""
        
        # Return result
        return self.n_tors