#!/usr/bin/env python3
#
################################################################################
# SFSXplorer                                                                   #
# Scoring Function Space eXplorer                                              #
################################################################################
#
# SFSXplorer reads input files generated by SAnDReS 2.0 (Xavier et al., 2016)
# (pdbqt and bind_#####.csv files) to create targeted scoring functions
# (Seifert, 2009). This program calculates energy terms loosely based on the
# AutoDock 4 Force Field (Morris et al., 1998; 2009) to explore the
# Scoring Function Space concept (Ross et al., 2013; Heck et al., 2017;
# Bitencourt-Ferreira & de Azevedo Jr., 2019; Veríssimo et al., 2022). We may
# vary the exponents of van der Waals and hydrogen-bond potentials and the
# parameters used to determine the electrostatic (Bitencourt-Ferreira & de
# Azevedo Jr., 2021) and desolvation potentials.
#
# References:
# Bitencourt-Ferreira G, de Azevedo WF Jr. Exploring the Scoring Function Space.
# Methods Mol Biol. 2019; 2053: 275–281.
#
# Bitencourt-Ferreira G, de Azevedo Junior WF. Electrostatic Potential Energy in
# Protein-Drug Complexes. Curr Med Chem. 2021; 28(24): 4954–4971.
#
# Heck GS, Pintro VO, Pereira RR, de Ávila MB, Levin NMB, de Azevedo WF.
# Supervised Machine Learning Methods Applied to Predict Ligand-Binding
# Affinity. Curr Med Chem. 2017; 24(23): 2459–2470.
#
# Morris GM, Goodsell D, Halliday R, Huey R, Hart W, Belew R, Olson A.
# Automated docking using a Lamarckian genetic algorithm and an empirical
# binding free energy function. J Comput Chem. 1998; 19: 1639–1662.
#
# Morris GM, Huey R, Lindstrom W, Sanner MF, Belew RK, Goodsell DS, Olson AJ.
# AutoDock4 and AutoDockTools4: Automated docking with
# selective receptor flexibility. J Comput Chem. 2009; 30(16): 2785–2791.
#
# Ross GA, Morris GM, Biggin PC. One Size Does Not Fit All: The Limits of
# Structure-Based Models in Drug Discovery. J Chem Theory Comput. 2013; 9(9):
# 4266–4274.
#
# Seifert MH. Targeted scoring functions for virtual screening. Drug Discov
# Today. 2009; 14(11-12): 562–569.
#
# Xavier MM, Heck GS, Avila MB, Levin NMB, Pintro VO, Carvalho NL, Azevedo WF.
# SAnDReS: a Computational Tool for Statistical Analysis of Docking Results and
# Development of Scoring Functions. Comb Chem High Throughput Screen. 2016;
# 9(10):801–812.
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
import sys
import numpy as np
from SFSXplorer import FF_AD4 as ad4
from SFSXplorer import vdw as vd
from SFSXplorer import hb as hb
from SFSXplorer import elec as e1
from SFSXplorer import desolv as ds1
from multiprocessing import Pool, cpu_count


def process_csv_chunk(hbtype,vdwtype,desoltype,dataset_dir,pot_VDW_n_min,pot_VDW_n_max, pot_VDW_m_min, pot_VDW_m_max,pot_HB_n_min, pot_HB_n_max, pot_HB_m_min, pot_HB_m_max, a_array, e0_array, k_array, l_array, log_array, m_array_desol, n_array_desol, s_array_desol, cc_array_desol, smooth_vdw_array, smooth_hb_array, cap_vdw_array,cap_hb_array, chunk):
    #print("entering 'process_csv_chunk' function")
    lines_out = []
    for line in chunk:
        if line[0].strip() != "PDB" and "#" not in line[0].strip():    
            pot = ad4.InterMol("/home/rquiroga/Downloads/sfs/misc/data/AD4.1_bound.dat")
            pot2 = ad4.InterMol("/home/rquiroga/Downloads/sfs/misc/data/AD4.2_bound.dat")
            #for line in chunk:
            # Same code as before, but without the `self.` prefixes.
            # Assign directory for a specific PDB to name_dir
            #print('line', line)
            #print('line0', line[0])
            name_dir = dataset_dir+str(line[0]).strip() + "/"
            # Invoking read_AD4_bound() method
            par_list = pot.read_AD4_bound()
            par_list2 = pot2.read_AD4_bound()
            # Invoking read_PDBQT() method
            lig_list = pot.read_PDBQT(name_dir+"lig.pdbqt")
            # Invoking read_PDBQT() method
            receptor_list = pot.read_PDBQT(name_dir+"receptor.pdbqt")
            #Retrieve atom type, coordinates and charge and put them in numpy arrays
            lig_coords,lig_type,lig_charge,rec_coords,rec_type,rec_charge = pot.get_atom_par_array(par_list,lig_list,receptor_list)
            #Calculate distance by pairs in a vectorized manner using broadcasting
            #dist_pairs = np.sqrt(np.sum((lig_coords[:, np.newaxis, :] - rec_coords[np.newaxis, :, :3])**2, axis=2))                    
            dist_pairs =pot.vectorized_dist(lig_coords,rec_coords)
            ################################################################
            # Calculate hydrogen-bond potentials
            if hbtype == 'walter':
                string_HB=""
                rhb_i,rhb_j,ehb_i,ehb_j = pot.get_atom_par_array_HB(par_list,lig_type,rec_type)
                n_exp_range = np.arange(pot_HB_n_min, pot_HB_n_max+1)
                m_exp_range = np.arange(pot_HB_m_min, pot_HB_m_max+1)
                for n_exp in n_exp_range:
                    for m_exp in m_exp_range:
                       if n_exp > m_exp: #if n_exp > m_exp: #if n_exp > m_exp:
                            v_HB_n_m = hb.hb_potentialv(rhb_i,ehb_i,rhb_j,ehb_j,dist_pairs,n_exp,m_exp)
                            string_HB+= str(",")+str(v_HB_n_m)
            # Calculate hydrogen-bond potentials vina1.2 style
            if hbtype == 'vina':
                string_HB=""
                rhb_i,rhb_j,ehb_i,ehb_j = pot.get_atom_par_array_HB(par_list2,lig_type,rec_type)
                n_exp_range = np.arange(pot_HB_n_min, pot_HB_n_max+1)
                m_exp_range = np.arange(pot_HB_m_min, pot_HB_m_max+1)
                for n_exp in n_exp_range:
                    for m_exp in m_exp_range:
                        if n_exp ==2*m_exp: #if n_exp > m_exp: #if n_exp > m_exp:
                            v_HB_n_m = hb.hb_potentialvv(rhb_i,ehb_i,rhb_j,ehb_j,dist_pairs,n_exp,m_exp)
                            string_HB+= str(",")+str(v_HB_n_m)   
              # Calculate hydrogen-bond potentials vina1.2 style
            if hbtype == 'smooth':
                string_HB=""
                rhb_i,rhb_j,ehb_i,ehb_j = pot.get_atom_par_array_HB(par_list2,lig_type,rec_type)
                n_exp_range = np.arange(pot_HB_n_min, pot_HB_n_max+1)
                m_exp_range = np.arange(pot_HB_m_min, pot_HB_m_max+1)
                for n_exp in n_exp_range:
                    for m_exp in m_exp_range:
                       if n_exp > m_exp: #if n_exp > m_exp: #if n_exp > m_exp:
                            for smooth in smooth_hb_array:
                                for cap in cap_hb_array:
                                    v_HB_n_m = hb.hb_potentialvvs(rhb_i,ehb_i,rhb_j,ehb_j,dist_pairs,n_exp,m_exp,smooth,cap)
                                    string_HB+= str(",")+str(v_HB_n_m)                        
            ###############################################################
            # Calculate van der Waals potentials
            if vdwtype == 'walter':
                string_VDW=""
                reqm_i,reqm_j,epsilon_i,epsilon_j = pot.get_atom_par_array_VDW(par_list,lig_type,rec_type)
                n_exp_range = np.arange(pot_VDW_n_min, pot_VDW_n_max+1)
                m_exp_range = np.arange(pot_VDW_m_min, pot_VDW_m_max+1)
                for n_exp in n_exp_range:
                    for m_exp in m_exp_range:
                        if n_exp ==2*m_exp: #if n_exp > m_exp: #if n_exp > m_exp:
                            v_VDW_n_m = vd.vdw_potentialv(reqm_i, epsilon_i, reqm_j, epsilon_j, dist_pairs, n_exp, m_exp)
                            string_VDW += str(",")+(str(v_VDW_n_m))
            
            if vdwtype == 'vina':            
                string_VDW=""
                reqm_i,reqm_j,epsilon_i,epsilon_j = pot.get_atom_par_array_VDW(par_list2,lig_type,rec_type)
                n_exp_range = np.arange(pot_VDW_n_min, pot_VDW_n_max+1)
                m_exp_range = np.arange(pot_VDW_m_min, pot_VDW_m_max+1)
                for n_exp in n_exp_range:
                    for m_exp in m_exp_range:
                        if n_exp ==2*m_exp: #if n_exp > m_exp: #if n_exp > m_exp:
                            v_VDW_n_m = vd.vdw_potentialvv(reqm_i, epsilon_i, reqm_j, epsilon_j, dist_pairs, n_exp, m_exp,ehb_i,ehb_j)
                            string_VDW += str(",")+(str(v_VDW_n_m))
            if vdwtype == 'smooth':            
                string_VDW=""
                reqm_i,reqm_j,epsilon_i,epsilon_j = pot.get_atom_par_array_VDW(par_list2,lig_type,rec_type)
                n_exp_range = np.arange(pot_VDW_n_min, pot_VDW_n_max+1)
                m_exp_range = np.arange(pot_VDW_m_min, pot_VDW_m_max+1)
                for n_exp in n_exp_range:
                    for m_exp in m_exp_range:
                        if n_exp ==2*m_exp: #if n_exp > m_exp: #if n_exp > m_exp:
                            for smooth in smooth_vdw_array:
                                for cap in cap_vdw_array:
                                    v_VDW_n_m = vd.vdw_potentialvvs(reqm_i, epsilon_i, reqm_j, epsilon_j, dist_pairs, n_exp, m_exp,ehb_i,ehb_j,smooth,cap)
                                    string_VDW += str(",")+(str(v_VDW_n_m))
            ################################################################

            # For Electrostatic Potential (Logistic Function)
            v_Elec_logistic=""
            v_Elec_tanh=""
            v_Elec_logistic_tanh=""
            q_i2d,q_j2d= pot.get_atom_par_array_elec(lig_charge,rec_charge)
            #Looping through a_array, e0_array, k_array, and l_array
            for a in a_array:
                for e0 in e0_array:
                    for k in k_array:
                        for l in l_array:
                            # Calculate Potential
                            for log_w in log_array:
                                if log_w ==1:
                                    tanh_w=0
                                    v_Elec_pot = e1.elec_potentialv(dist_pairs,q_i2d,q_j2d,l,k,a,e0,log_w,tanh_w)    
                                    v_Elec_logistic +=str(",")+str(v_Elec_pot)
                                if log_w ==0.5:
                                    tanh_w=0.5
                                    v_Elec_pot = e1.elec_potentialv(dist_pairs,q_i2d,q_j2d,l,k,a,e0,log_w,tanh_w)    
                                    v_Elec_logistic_tanh +=str(",")+str(v_Elec_pot)
                                if log_w ==0:
                                    tanh_w=1
                                    v_Elec_pot = e1.elec_potentialv(dist_pairs,q_i2d,q_j2d,l,k,a,e0,log_w,tanh_w)    
                                    v_Elec_tanh +=str(",")+str(v_Elec_pot)
                                
            ################################################################
            # For desolvation potential
            v_Desol=""
            v_i,v_j,s_i,s_j = pot.get_atom_par_array_desolv(par_list,lig_type,rec_type)
            # Looping through m_array_desol, n_array_desol, and s_array_desol
            if desoltype =="vina":            
                for m in m_array_desol:
                    for n in n_array_desol:
                        for sigma in s_array_desol:
                            for cc in cc_array_desol:
                                # Calculate potential
                                v_Desol_pot = ds1.desol_potentialvv(q_i2d,q_j2d,v_i,v_j,s_i,s_j,dist_pairs,m,n,sigma,cc)
                                #v_Desol_pot = ds1.desol_potentialv_charge(q_i2d,q_j2d,v_i,v_j,s_i,s_j,dist_pairs,m,n,sigma)
                                v_Desol += str(",")+str(v_Desol_pot)
                                #print('Desol',v_Desol)        
                            
            
            # Looping through m_array_desol, n_array_desol, and s_array_desol
            if desoltype =="walter":
                for m in m_array_desol:
                    for n in n_array_desol:
                        for sigma in s_array_desol:
                            # Calculate potential
                            v_Desol_pot = ds1.desol_potentialv(v_i,v_j,s_i,s_j,dist_pairs,m,n,sigma)
                            v_Desol += str(",")+str(v_Desol_pot)
                            #print('Desol',v_Desol)      
            ################################################################
            # Set up line_VDW_HB
            line_VDW_HB = string_VDW+string_HB 
            # Set up an empty string
            data_in = ""
            # Looping through the data (from bind_####.csv)
            #print(type(line))
            for count,ele in enumerate(line):
                if count != len(line)-1:
                    data_in += str(line[count]) + ","
                else:     
                    data_in += str(line[count])
            lines_out.append(data_in + line_VDW_HB + v_Elec_logistic + v_Elec_logistic_tanh + v_Elec_tanh + v_Desol + "\n")
    return lines_out

# Define Explorer() class
class Explorer(object):
    """A class to explore the scoring function space"""	

    # Define the constructor method
    def __init__(self,sfs_in):
        """Constructor method"""
        
        # Set up attributes
        self.sfs_in = sfs_in
        # Show message
        print("\nExploring the Scoring Function Space...")
# Define read_input() method
    def read_input(self):
        """Method to read input data"""
        
        # Import section
        import csv
        
        # Define function to handle hash in number field (float or integer)
        def handle_hash(type_in,line_in):
            """Function to handle hash in number field and returns a float or
            an integer"""
            
            # Test type
            if type_in == "float":
                # Handle hash for float output
                try:
                    index_hash = str(line_in).index("#")
                    number_out = float(line_in[:index_hash])
                except:
                    number_out = float(line_in)
            elif type_in == "int":
                # Handle hash for integer output
                try:
                    index_hash = str(line_in).index("#")
                    number_out = int(line_in[:index_hash])
                except:
                    number_out = int(line_in)
            else:
                # Print error message and exit
                sys.exit("\nError! Not defined type of number!")
            
            # Return number
            return number_out
            
        # Try to open sfs.in file
        try:
            fo = open(self.sfs_in,"r")
            csv = csv.reader(fo)
        except IOError:
            msg_out = "\nI can't find "+self.sfs_in+" file!"
            msg_out += "Finishing execution."
            sys.exit(msg_out)
        
        # Looping through input file with commands (e.g., sfs.in)
        self.smooth_vdw_i=0;self.smooth_vdw_f=0;self.smooth_hb_i=0;self.smooth_hb_f=0;self.n_smooth_vdw=0;self.n_smooth_hb=0;self.cap_vdw_i=0;self.cap_vdw_f=0;self.cap_hb_i=0;self.cap_hb_f=0;self.n_cap_vdw=0;self.n_cap_hb=0;self.cc_desol_i=0;self.cc_desol_f=0;self.n_cc_desol=0;self.log_i=0;self.log_f=0;self.n_log=0;
        for line in csv:
            #print(line)
            if (line[0] == "#") | (line[0] == ""):
                continue
            elif line[0].strip() == "dataset_dir":
                self.dataset_dir = str(line[1])
            elif line[0].strip() == "ligands_in":
                self.ligands_in = str(line[1])
            elif line[0].strip() == "scores_out":
                self.scores_out = str(line[1])
            elif line[0].strip() == "binding_type":
                self.binding_type = str(line[1])
            
            # For van der Waals potential
            elif line[0].strip() == "smooth_vdw_i":
                self.smooth_vdw_i =  handle_hash("float",line[1])                    
            elif line[0].strip() == "smooth_vdw_f":
                self.smooth_vdw_f =  handle_hash("float",line[1])  
            elif line[0].strip() == "n_smooth_vdw":
                self.n_smooth_vdw = handle_hash("int",line[1])
            elif line[0].strip() == "cap_vdw_i":
                self.cap_vdw_i =  handle_hash("float",line[1])                    
            elif line[0].strip() == "cap_vdw_f":
                self.cap_vdw_f =  handle_hash("float",line[1])  
            elif line[0].strip() == "n_cap_vdw":
                self.n_cap_vdw = handle_hash("int",line[1])
            elif line[0].strip() == "pot_VDW_m_min":
                self.pot_VDW_m_min = handle_hash("int",line[1])
            elif line[0].strip() == "pot_VDW_m_max":
                self.pot_VDW_m_max = handle_hash("int",line[1])
            elif line[0].strip() == "pot_VDW_n_min":
                self.pot_VDW_n_min = handle_hash("int",line[1])
            elif line[0].strip() == "pot_VDW_n_max":
                self.pot_VDW_n_max = handle_hash("int",line[1])
            
            # For hydrogen-bond potential
            elif line[0].strip() == "smooth_hb_i":
                self.smooth_hb_i =  handle_hash("float",line[1])                    
            elif line[0].strip() == "smooth_hb_f":
                self.smooth_hb_f =  handle_hash("float",line[1])  
            elif line[0].strip() == "n_smooth_hb":
                self.n_smooth_hb = handle_hash("int",line[1])
            elif line[0].strip() == "cap_hb_i":
                self.cap_hb_i =  handle_hash("float",line[1])                    
            elif line[0].strip() == "cap_hb_f":
                self.cap_hb_f =  handle_hash("float",line[1])  
            elif line[0].strip() == "n_cap_hb":
                self.n_cap_hb = handle_hash("int",line[1])
            elif line[0].strip() == "pot_HB_m_min":
                self.pot_HB_m_min = handle_hash("int",line[1])
            elif line[0].strip() == "pot_HB_m_max":
                self.pot_HB_m_max = handle_hash("int",line[1])
            elif line[0].strip() == "pot_HB_n_min":
                self.pot_HB_n_min = handle_hash("int",line[1])
            elif line[0].strip() == "pot_HB_n_max":
                self.pot_HB_n_max = handle_hash("int",line[1])
            
            # For electrostatic potential (set up parameters for arrays)
            elif line[0].strip() == "lambda_i":
                self.lambda_i =  handle_hash("float",line[1])                    
            elif line[0].strip() == "lambda_f":
                self.lambda_f =  handle_hash("float",line[1])  
            elif line[0].strip() == "n_lambda":
                self.n_lambda = handle_hash("int",line[1])
            
            elif line[0].strip() == "k_i":
                self.k_i = handle_hash("float",line[1])
            elif line[0].strip() == "k_f":
                self.k_f = handle_hash("float",line[1])
            elif line[0].strip() == "n_k":
                self.n_k = handle_hash("int",line[1])
            
            elif line[0].strip() == "log_i":
                self.log_i = handle_hash("float",line[1])
            elif line[0].strip() == "log_f":
                self.log_f = handle_hash("float",line[1])
            elif line[0].strip() == "n_log":
                self.n_log = handle_hash("int",line[1])

            elif line[0].strip() == "A_i":
                self.A_i = handle_hash("float",line[1])
            elif line[0].strip() == "A_f":
                self.A_f = handle_hash("float",line[1])
            elif line[0].strip() == "n_A":
                self.n_A = handle_hash("int",line[1])
            
            elif line[0].strip() == "epsilon0_i":
                self.epsilon0_i = handle_hash("float",line[1])
            elif line[0].strip() == "epsilon0_f":
                self.epsilon0_f = handle_hash("float",line[1])
            elif line[0].strip() == "n_epsilon0":
                self.n_epsilon0 = handle_hash("int",line[1])
            
            # For desolvation potential (set up parameters for arrays)
            elif line[0].strip() == "m_desol_i":
                self.m_desol_i = handle_hash("int",line[1])
            elif line[0].strip() == "m_desol_f":
                self.m_desol_f = handle_hash("int",line[1])
            elif line[0].strip() == "n_m_desol":
                self.n_m_desol = handle_hash("int",line[1])
            
            elif line[0].strip() == "n_desol_i":
                self.n_desol_i = handle_hash("int",line[1])
            elif line[0].strip() == "n_desol_f":
                self.n_desol_f = handle_hash("int",line[1])
            elif line[0].strip() == "n_n_desol":
                self.n_n_desol = handle_hash("int",line[1])
                
            elif line[0].strip() == "sigma_desol_i":
                self.sigma_desol_i = handle_hash("float",line[1])
            elif line[0].strip() == "sigma_desol_f":
                self.sigma_desol_f = handle_hash("float",line[1])
            elif line[0].strip() == "n_sigma_desol":
                self.n_sigma_desol = handle_hash("int",line[1])
                
            elif line[0].strip() == "cc_desol_i":
                self.cc_desol_i = handle_hash("float",line[1])
            elif line[0].strip() == "cc_desol_f":
                self.cc_desol_f = handle_hash("float",line[1])
            elif line[0].strip() == "n_cc_desol":
                self.n_cc_desol = handle_hash("int",line[1])
                
        # Close file
        fo.close()
        
    # Define read_data() method
    def read_data(self):
        """Method to read data ligands.in"""
        
        # Import section
        import csv

        # Open csv file
        self.fo0 = open(self.ligands_in,"r")
        self.csv0 = csv.reader(self.fo0)

    def write_energy1(self,hbtype,vdwtype,desoltype):
         # Create arrays of parameters for multicore processing
        a_array = np.linspace(self.A_i, self.A_f, self.n_A)
        e0_array = np.linspace(self.epsilon0_i, self.epsilon0_f, self.n_epsilon0)
        log_array = np.linspace(self.log_i, self.log_f, self.n_log)
        k_array = np.linspace(self.k_i, self.k_f, self.n_k)
        l_array = np.linspace(self.lambda_i, self.lambda_f, self.n_lambda)
        m_array_desol = np.linspace(self.m_desol_i, self.m_desol_f, self.n_m_desol)
        n_array_desol = np.linspace(self.n_desol_i, self.n_desol_f, self.n_n_desol)
        s_array_desol = np.linspace(self.sigma_desol_i, self.sigma_desol_f, self.n_sigma_desol)
        cc_array_desol = np.linspace(self.cc_desol_i, self.cc_desol_f, self.n_cc_desol)
        smooth_hb_array = np.linspace(self.smooth_hb_i, self.smooth_hb_f, self.n_smooth_hb)
        smooth_vdw_array = np.linspace(self.smooth_vdw_i, self.smooth_vdw_f, self.n_smooth_vdw)
        cap_hb_array = np.linspace(self.cap_hb_i, self.cap_hb_f, self.n_cap_hb)
        cap_vdw_array = np.linspace(self.cap_vdw_i, self.cap_vdw_f, self.n_cap_vdw)
        #Create headers
        def get_headers_svdw(n_min, n_max, m_min, m_max, smooth_array, cap_array, prefix):
            n_exp_range = np.arange(n_min, n_max + 1)
            m_exp_range = np.arange(m_min, m_max + 1)
            headers = [f"{prefix}_{n_exp}_{m_exp}_s{s}_c{cap}" for cap in np.round(cap_array,decimals=0) for s in np.round(smooth_array,decimals=2) for n_exp in n_exp_range for m_exp in m_exp_range if n_exp ==2*m_exp] #if n_exp > m_exp] # uncomment and delete "]" to only explore n > m
            return headers
        def get_headers_shb(n_min, n_max, m_min, m_max, smooth_array, cap_array, prefix):
            n_exp_range = np.arange(n_min, n_max + 1)
            m_exp_range = np.arange(m_min, m_max + 1)
            headers = [f"{prefix}_{n_exp}_{m_exp}_s{s}_c{cap}" for cap in np.round(cap_array,decimals=0) for s in np.round(smooth_array,decimals=2) for n_exp in n_exp_range for m_exp in m_exp_range if n_exp > m_exp] #if n_exp > m_exp] # uncomment and delete "]" to only explore n > m
            return headers
        def get_headersvdw(n_min, n_max, m_min, m_max, prefix):
            n_exp_range = np.arange(n_min, n_max + 1)
            m_exp_range = np.arange(m_min, m_max + 1)
            headers = [f"{prefix}_{n_exp}_{m_exp}" for n_exp in n_exp_range for m_exp in m_exp_range if n_exp ==2*m_exp] #if n_exp > m_exp] # uncomment and delete "]" to only explore n > m
            return headers
        def get_headershb(n_min, n_max, m_min, m_max, prefix):
            n_exp_range = np.arange(n_min, n_max + 1)
            m_exp_range = np.arange(m_min, m_max + 1)
            headers = [f"{prefix}_{n_exp}_{m_exp}" for n_exp in n_exp_range for m_exp in m_exp_range if n_exp > m_exp] #if n_exp > m_exp] # uncomment and delete "]" to only explore n > m
            return headers
        if vdwtype == 'smooth':
            vdw_headers = get_headers_svdw(self.pot_VDW_n_min, self.pot_VDW_n_max, self.pot_VDW_m_min, self.pot_VDW_m_max,smooth_vdw_array, cap_vdw_array, 'v_VDW_v')
        if vdwtype == 'vina':
            vdw_headers = get_headersvdw(self.pot_VDW_n_min, self.pot_VDW_n_max, self.pot_VDW_m_min, self.pot_VDW_m_max, 'v_VDW_v')
        if vdwtype == 'walter':
            vdw_headers = get_headersvdw(self.pot_VDW_n_min, self.pot_VDW_n_max, self.pot_VDW_m_min, self.pot_VDW_m_max, 'v_VDW')
        if hbtype == 'smooth':
            hb_headers = get_headers_shb(self.pot_HB_n_min, self.pot_HB_n_max, self.pot_HB_m_min, self.pot_HB_m_max,smooth_hb_array,cap_hb_array, 'v_HB_v')
        if hbtype == 'vina':
            hb_headers = get_headershb(self.pot_HB_n_min, self.pot_HB_n_max, self.pot_HB_m_min, self.pot_HB_m_max, 'v_HB_v')
        if hbtype == 'walter':
            hb_headers = get_headershb(self.pot_HB_n_min, self.pot_HB_n_max, self.pot_HB_m_min, self.pot_HB_m_max, 'v_HB')
        e1=[]
        e2=[]
        e3=[]
        for log_w in log_array:
            if log_w == 1:
                e1 = [f"v_Elec_Log_{a}_{e0}_{k}_{l}"           for a in a_array                for e0 in e0_array                for k in k_array                for l in l_array]
            if log_w==0.5:
                e2 = [f"v_Elec_Log_Tanh_{a}_{e0}_{k}_{l}"           for a in a_array                for e0 in e0_array                for k in k_array                for l in l_array]   
            if log_w==0:
                e3 = [f"v_Elec_Tanh_{a}_{e0}_{k}_{l}"           for a in a_array                for e0 in e0_array                for k in k_array                for l in l_array]                
        # join the list of strings with commas
        elec_headers = [s for lst in [e1, e2, e3] for s in lst if s]     
        #elec_headers = [f"v_Elec_Log_{a}_{e0}_{k}_{l},v_Elec_Tanh_{a}_{e0}_{k}_{l},v_Elec_Log_Tanh_{a}_{e0}_{k}_{l}"           for a in a_array                for e0 in e0_array                for k in k_array                for l in l_array]
        if desoltype == 'vina':
            desol_headers = [f"v_Desol_vina_{m}_{n}_{sigma}_{cc}" for m in m_array_desol         for n in n_array_desol                 for sigma in s_array_desol for cc in cc_array_desol]
        if desoltype == 'walter':
            desol_headers = [f"v_Desol_{m}_{n}_{sigma}" for m in m_array_desol         for n in n_array_desol                 for sigma in s_array_desol]
        headers = ",".join(vdw_headers + hb_headers + elec_headers + desol_headers)

        #with open(self.scores_out, "w") as fo1:
        header_in = ",".join(next(self.csv0))
        header_out = header_in + "," + headers
        out = open(self.scores_out,"w")
        out.write(header_out + "\n")
        num_processes = cpu_count() # uses all available cores
        csv_chunks = [[] for _ in range(num_processes)]
        csv0_list = list(self.csv0)
        for i, line in enumerate(csv0_list):
            #print("id",line[0],"i",i,"index",(i // chunk_size) % num_processes)
            #csv_chunks[(i // chunk_size) % num_processes].append(line)
            csv_chunks[i % num_processes].append(line)
        #define variables globally to be able to pickle and feed to process_csv_chunk
        dataset_dir=self.dataset_dir
        pot_VDW_n_min=self.pot_VDW_n_min
        pot_VDW_n_max=self.pot_VDW_n_max
        pot_VDW_m_min=self.pot_VDW_m_min
        pot_VDW_m_max=self.pot_VDW_m_max
        pot_HB_n_min=self.pot_HB_n_min
        pot_HB_n_max=self.pot_HB_n_max
        pot_HB_m_min=self.pot_HB_m_min
        pot_HB_m_max=self.pot_HB_m_max
        # Process each chunk in a separate process.
        with Pool(num_processes) as pool:
            results = pool.starmap(process_csv_chunk, [(hbtype,vdwtype,desoltype,dataset_dir,pot_VDW_n_min,pot_VDW_n_max, pot_VDW_m_min, pot_VDW_m_max,pot_HB_n_min, pot_HB_n_max, pot_HB_m_min, pot_HB_m_max, a_array, e0_array, k_array, l_array, log_array, m_array_desol, n_array_desol, s_array_desol, cc_array_desol, smooth_vdw_array, smooth_hb_array, cap_vdw_array,cap_hb_array, chunk) for chunk in csv_chunks])

        for i in range(len(csv0_list)):
            self.fo1=out
            chunk_index = i % num_processes
            line_index = i // num_processes
            self.fo1.write(results[chunk_index][line_index])
    # Close files
        self.fo0.close()
        self.fo1.close()
        print("\nDone!")
