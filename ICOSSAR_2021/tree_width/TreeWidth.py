'''
This module contains functions to determine the tree decomposition of graphs.

The exact treewidth solver is from here
https://github.com/freetdi/p17

The approximate treewidth solver is from here
https://github.com/kit-algo/flow-cutter-pace17

Functions
---------
**TreeWidthExact(EdgeList,seconds=6,ID='') :**  
    This function computes the minimum width tree decomposition of the graph. If it fails to find this minimum width tree decomposition, it falls back to the approximate solver.
    https://github.com/freetdi/p17

**TreeWidthApprox(EdgeList,seconds=6,ID='') :**  
    This function computes an approximate minimum width tree decomposition. When a termination signal is received, it outputs the best solution it found.

Notes
-----
These tree decomposition codes are only written to work on UNIX systems. To get it to work on windows systems, the command wsl.exe must be used.

This can cause some strange and unstable behavior when doing parallel processes.
'''

import subprocess
import sys
import os
import signal
import time

import numpy

def TreeWidthApprox(EdgeList,seconds=6,ID=''):
    '''
    This function computes an approximate minimum width tree decomposition. When a termination signal is received, it outputs the best solution it found.

    Parameters
    ----------
    EdgeList : list of tuples, int
        Edgelist[i][0] and Edgelist[i][1] are the node labels connected by edge i. Labels should be integers consecutive from 0 to |V|-1.

    seconds : int
        The length of time to run the approximate treewidth solver.

    ID : string
        When doing parallel processing, a unique ID should be fed into this function to prevent different instances from overwriting each other on the same file

    Returns
    -------
    filename : string
        The name of the output file is returned, so the calling function can open the correct file.

    Notes
    -----
    The internal pseudo-rng is not easily accessible, so different runs may produce different tree decompositions.

    subprocess.Popen is used to call this function. It may be possible to hook these functions directly in python in the future, for better compatibility, portability, and stability

    Examples
    --------
    >>> EdgeList=[  (0,1,.5),
                    (0,2,.5),
                    (0,3,.5),
                    (1,2,.5),
                    (1,3,.5),
                    (2,3,.5)]
    >>> seconds = 15
    >>> ID=0
    >>> filename = TreeWidthApprox(EdgeList,seconds=seconds,ID=ID)
    '''
    # get edges
    NN=numpy.max(EdgeList)+1

    ee=[]
    for e in range(0,len(EdgeList)):
        Edge=[EdgeList[e][0],EdgeList[e][1]]
        ee.append(tuple(Edge))

    cwd=os.path.dirname(os.path.realpath(__file__))

    file=open(cwd+'/G'+ID+'.gr','w',newline='\n')
    file.write('p tw '+str(int(NN))+' '+str(len(ee))+'\n')
    for e in ee:
        file.write(str(int(e[0])+1)+' '+str(int(e[1])+1)+'\n')
    file.close()

    # try:
    #     os.remove(os.getcwd()+'/'+'outApprox'+ID)
    # except:
    #     pass
    outfile=open('outApprox'+ID,'w')

    sec=str(int(seconds))

    if sys.platform == "win32":
        p = subprocess.Popen(['wsl.exe','timeout','-k',sec+'s',sec+'s','flow-cutter-pace17/flow_cutter_pace17','G'+ID+'.gr'],stdout=outfile,cwd=cwd)
    elif sys.platform == "linux":
        p = subprocess.Popen(['timeout','-k',sec+'s',sec+'s','flow-cutter-pace17/flow_cutter_pace17','G'+ID+'.gr'],stdout=outfile,cwd=cwd)
    else:
        raise RuntimeError('OS X is not supported')
    p.wait()
    # p.kill()
    outfile.close()
    os.remove(cwd+'/G'+ID+'.gr')

    return 'outApprox'+ID

def TreeWidthExact(EdgeList,seconds=6,ID=''):
    '''
    This function computes an exact minimum width tree decomposition. When a termination signal is received, it does not return an ErrorCode equal to 0, so the approximate treewidth solver is called as a fallback.

    Parameters
    ----------
    EdgeList : list of tuples, int
        Edgelist[i][0] and Edgelist[i][1] are the node labels connected by edge i. Labels should be integers consecutive from 0 to |V|-1.

    seconds : int
        The length of time to run the approximate treewidth solver.

    ID : string
        When doing parallel processing, a unique ID should be fed into this function to prevent different instances from overwriting each other on the same file

    Returns
    -------
    filename : string
        The name of the output file is returned, so the calling function can open the correct file. This also lets the calling function know if the approximate solver fallback was used.

    Notes
    -----
    The internal pseudo-rng is not easily accessible, so different runs may produce different tree decompositions.

    subprocess.Popen is used to call this function. It may be possible to hook these functions directly in python in the future, for better compatibility, portability, and stability

    Examples
    --------
    >>> EdgeList=[  (0,1,.5),
                    (0,2,.5),
                    (0,3,.5),
                    (1,2,.5),
                    (1,3,.5),
                    (2,3,.5)]
    >>> seconds = 15
    >>> ID=0
    >>> filename = TreeWidthExact(EdgeList,seconds=seconds,ID=ID)
    '''

    NN=numpy.max(EdgeList)+1

    ee=[]
    for e in range(0,len(EdgeList)):
        Edge=[EdgeList[e][0],EdgeList[e][1]]
        ee.append(tuple(Edge))

    cwd=os.path.dirname(os.path.realpath(__file__))

    file=open(cwd+'/G'+ID+'.gr','w',newline='\n')
    file.write('p tw '+str(int(NN))+' '+str(len(ee))+'\n')
    for e in ee:
        file.write(str(int(e[0])+1)+' '+str(int(e[1])+1)+'\n')
    file.close()

    # try:
    #     os.remove(os.getcwd()+'/'+'outExact'+ID)
    # except:
    #     pass
    outfile=open(os.getcwd()+'/'+'outExact'+ID,'w')

    sec=str(int(seconds))
    if sys.platform == "win32":
        p = subprocess.Popen(['wsl.exe','timeout','-k',sec+'s',sec+'s','p17/tw-exact','<','G'+ID+'.gr'],stdout=outfile,cwd=cwd)
    elif sys.platform == "linux":
        p = subprocess.Popen(['timeout','-k',sec+'s',sec+'s','p17/tw-exact','<','G'+ID+'.gr'],stdout=outfile,cwd=cwd)
    else:
        raise RuntimeError('OS X is not supported')
    ErrorCode=p.wait()
    outfile.close()
    if ErrorCode==0:
        print('exact')
        os.remove(cwd+'/G'+ID+'.gr')
        return 'outExact'+ID

    key=True
    while key:
        try:
            os.remove(os.getcwd()+'/'+'outExact'+ID)
            key=False
        except:
            pass

    TreeWidthApprox(EdgeList,seconds=seconds,ID=ID)
    print('approx')
    return 'outApprox'+ID
