'''
This script executes all the code needed to generate the figures and tables in "Tensor Network Contraction For Network Reliability Estimates"

Analysis and computation is performed by calling functions from ICOSSAR_2021_analysis.

The plots are generated by calling functions from generate_figures

Notes
-----
This script takes ~24 hours to run.

When some of these functions are first executed, numba performs jit compilation. This causes the first execution of that function to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

Examples
--------
>>> python script.py
'''

import os
import sys
# ensures local modules are imported properly
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,cwd)
# must use ubuntu 18.04, for p17 to compile and work correctly
# sudo apt-get install make
# sudo apt-get install autotools-dev
# sudo apt-get install automake
# sudo apt-get install libtool
## sudo apt-get install libtool-bin
# sudo apt-get install libboost-all-dev
# sudo apt install g++
# sudo apt install stx-btree-dev
# https://ubuntu.pkgs.org/18.04/ubuntu-universe-amd64/stx-btree-dev_0.9-2build2_all.deb.html
# make --always-make

# GridGraphComputeTime8
# RcubicWidth_short
# misplaced \noalign

import ICOSSAR_2021_analysis
import generate_figures
import cubic_graph_utilities

if __name__ == "__main__":
    ### create figure showing #MonteCarloTrials ~ 1/p
    # ICOSSAR_2021_analysis.MCtrials_vs_e()
    generate_figures.MCtrials_vs_e_Figure()

    ### create figure showing the tensor contraction process
    generate_figures.TensorGraphFigure()

    ### perform analysis on the grid graphs
    # ICOSSAR_2021_analysis.GridRel_vs_n()
    # ICOSSAR_2021_analysis.GridRel_vs_n_MCtrials()
    # ICOSSAR_2021_analysis.GridRel_vs_len()
    # ICOSSAR_2021_analysis.GridRel_vs_len_MCtrials()
    generate_figures.GridGraphComplexityFigure()

    ### generate list of random connected cubic graphs
    # Create list of large cubic graphs for treewidth analysis
    # cubic_graph_utilities.GenConnectedRandomCubic(ID0=0,ID1=10000,Vmin=20,Vmax=400,seed=42)
    # measure the treewidths of these graphs
    filename1="data/ConnectedRandomCubic-0-10000-Vmin-20-Vmax-400-seed-42"
    # ICOSSAR_2021_analysis.CubicGraphTreeWidths(filename1)

    # Create list of smaller cubic graphs for reliability and
    # edge cover analysis
    # cubic_graph_utilities.GenConnectedRandomCubic(ID0=0,ID1=10000,Vmin=20,Vmax=50,seed=64)

    ### perform analysis on the cubic graphs
    filename2="data/ConnectedRandomCubic-0-10000-Vmin-20-Vmax-50-seed-64"
    # get cubic graph reliability, set edge reliability to .99
    # ICOSSAR_2021_analysis.RandomCubicRel(filename2,edge_reliability=.99)
    # ICOSSAR_2021_analysis.RandomCubicRel_MCtrials(filename2)
    generate_figures.RandomCubicWidthFigure_short()
    generate_figures.RandomCubicMonteCarlo()

    ### perform analysis on the power grids
    # ICOSSAR_2021_analysis.PowerGridsREL()
    # ICOSSAR_2021_analysis.PowerGridsREL_MCtrials()
    generate_figures.TreewidthCompare()
    generate_figures.PowerGridMonteCarlo()
    generate_figures.PowerGridTable()
