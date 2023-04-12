#!/usr/bin/env python3
#
################################################################################
# SFSXplorer                                                                   #
# Scoring Function Space eXplorer                                              #
################################################################################
#
################################################################################
# Dr. Walter F. de Azevedo, Jr.                                                #
# https://azevedolab.net/                                                      #
# January 12, 2023                                                             #
################################################################################
#
# To run SFSXplorer
# python3 sfsxplorer.py sfs.in all > sfs.log &
#
# Import section
import sys
from SFSXplorer import sfs
from SFSXplorer import statistical_analysis as sa

# Define main()
def main():
    print(sys.argv)
    if len(sys.argv) == 0:
        print('*You failed to provide the input file (e.g., sfs.in) and run mode on the command line, using default*')
        #sys.exit(1)  # abort because of error
    if len(sys.argv) == 1:
        print('*You failed to provide the run mode on the command line, using "all" as default*')
        # Get input files from terminal
    #sfs_in = sys.argv[2] if len(sys.argv) >= 2 else '/home/rquiroga/Downloads/sfs/datasets/SFSXplorer_Tutorial_01/sfs_02.in'# Input file (e.g., sfs.in)
    sfs_in = sys.argv[1] if len(sys.argv) >= 2 else './sfs.in'# Input file (e.g., sfs.in)
    mode_in = sys.argv[2] if len(sys.argv) >= 3 else 'Explore'       # All for exploring the scoring function space
                                 # and statistical analysis of results
                                 # Stats for statistical analysis only
                                 # Explore for exploring the scoring function
                                 # space only

    # Define explore() function
    def explore():
        """Function to explore the scoring function space"""

        # Explore the Scoring Function Space
        #
        # Instantiate an object of the Explorer class
        space = sfs.Explorer(sfs_in)

        # Invoke read_input() method
        space.read_input()

        # Invoke read_data() method
        space.read_data()

        # Invoke write_energy() method
        #space.write_energy1(hbtype='x1',vdwtype='x2',desoltype='x3')
        space.write_energy1(hbtype='smooth',vdwtype='smooth',desoltype='vina')
        #space.write_energy1(hbtype='walter',vdwtype='walter',desoltype='walter')
        #space.write_energy1(hbtype='vina',vdwtype='vina',desoltype='vina')

    # Define stats_analysis() function
    def stats_analysis():
        """Function to carry out statistical analysis of the results"""

        # Statistical Analysis
        #
        # Instantiate an object of Stats class
        data1 = sa.Stats(sfs_in)

        # Invoke read_stats_in() method
        data1.read_stats_in()

        # Invoke read_data() method
        data1.read_data()

        # Invoke bundle() method
        data1.bundle()

    # Check mode_in
    if mode_in.upper() == "ALL":
        explore()
        stats_analysis()
    elif mode_in.upper() == "EXPLORE":
        explore()
    elif mode_in.upper() == "STATS":
        stats_analysis()
    else:
        msg_out = "Unidentified mode request!\n"
        msg_out += "Valid modes: All, Stats, Explore\n"
        msg_out += "All for exploring the scoring function space"
        msg_out += "and statistical analysis of results.\n"
        msg_out += "Explore for exploring the scoring function space only.\n"
        msg_out += "Stats for statistical analysis only.\n"
        sys.exit(msg_out)

main()
