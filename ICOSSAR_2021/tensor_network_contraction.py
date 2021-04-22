'''
This module contains functions to perform miscellaneous tasks on cubic graphs. Mostly for generating uniform connected random cubic graphs.

Functions
---------
**Main(Nodes,EdgeOrder,pf=.5,verbose=True) :**  
    Main function called by user to calculate the tensor network contraction defined by the list of Nodes (tensors and their indices).

**CoreLoop(T1,T1ix,T2,T2ix,Toutix) :**  
    Performs the tensor contraction.

**CoreLoopScalar(T1,T1ix,T2,T2ix) :**  
    Performs the final tensor contraction down to a scalar.

Notes
-----
When the tensor network contraction is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

While the tensor network contraction for Source-Terminal reliability is discussed in the ICOSSAR 2021 paper and has an exact solution for directed acyclic networks, it is not implemented in this module.
Proof-of-concept code has been written to verify the S-T tensor construction.
'''

import sys
import os

import time

import numpy
from numba import jit

# each node defined by a tuple
# (Tensor, edges)
# Nodes is list of these tuples
# EdgeOrder is list of tuples (e1,e2)

# Sizeix=10
# BitArray=numpy.zeros(Sizeix,dtype=numpy.int8)
# for k in range(0,2**Sizeix):
#     for b in range(0,Sizeix):
#         BitArray[b]=bool((1<<b)&k)
#     print(BitArray)
#     input()
# sys.exit()

@jit(nopython=True,cache=True)
def CoreLoop(T1,T1ix,T2,T2ix,Toutix):
    '''
    This function performs the tensor product to contract, merge, two tensors.
    Numba complied for extra speed, much better at for loops then plain python or numpy.

    Parameters
    ----------
    T1 : 1D numpy array of 2^(T1ix.shape), float
        Multi dimensional tensor flattened to a vector.

    T1ix : 1D numpy array of size n, int
        Tensor indices for T1. T1 is addressed by:
        f(T1ix[0])*2^0+f(T1ix[1])*2^1+f(T1ix[2])*2^2+...+f(T1ix[n-1])*2^(n-1)
        where f is some function that returns the value of index k.

    T2 : 1D numpy array of 2^(T2ix.shape), float
        Multi dimensional tensor flattened to a vector.

    T2ix : 1D numpy array of size n, int
        Tensor indices for T2. T2 is addressed by:
        f(T2ix[0])*2^0+f(T2ix[1])*2^1+f(T2ix[2])*2^2+...+f(T2ix[n-1])*2^(n-1)
        where f is some function that returns the value of index k.

    Toutix : 1D numpy array of size n, int
        Tensor indices for Tout

    Returns
    -------
    Tout : 1D numpy array of 2^(Toutix.shape), float
        The result of the tensor product

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Examples
    --------
    >>> T1=[1,2,3,4,5,6,7,8]
    >>> T1ix=[0,1,2]
    >>> T2=[1,2,4,8,16,32,64,128]
    >>> T2ix=[2,3,4]
    >>> Toutix=[0,1,3,4]
    >>> Tout=CoreLoop(T1,T1ix,T2,T2ix,Toutix)
    '''
    # determine size of bit array to iterate through
    Sizeix=max((numpy.max(T1ix),numpy.max(T2ix)))+1

    # get number of indices
    T1s=T1ix.shape[0]
    T2s=T2ix.shape[0]
    Touts=Toutix.shape[0]

    # allocate output tensor
    Tout=numpy.zeros(2**Touts)

    # allocate bit array
    # is really an array for determining if index j is 1,
    # how much should the tensor vector index shift?
    BitArray=numpy.zeros((3,Sizeix),dtype=numpy.int64)
    for zz in range(0,T1s):
        BitArray[0,T1ix[zz]]=(1<<zz)
    for zz in range(0,T2s):
        BitArray[1,T2ix[zz]]=(1<<zz)
    for zz in range(0,Touts):
        BitArray[2,Toutix[zz]]=(1<<zz)

    # loop over all permutations of the tensor indices
    # k is used as a bit mask
    for k in range(0,2**Sizeix):
        T1p=0
        T2p=0
        Toutp=0
        # scan through the bit array,
        # applying the proper masking and index shifting
        for b in range(0,Sizeix):
            TT=bool((1<<b)&k)
            T1p=T1p+BitArray[0,b]*TT
            T2p=T2p+BitArray[1,b]*TT
            Toutp=Toutp+BitArray[2,b]*TT

        # perform the multiplication and addition to output tensor
        Tout[Toutp]=Tout[Toutp]+T1[T1p]*T2[T2p]


    return Tout

@jit(nopython=True,cache=True)
def CoreLoopScalar(T1,T1ix,T2,T2ix):
    '''
    This function performs the tensor product to contract, merge, two tensors.
    Numba complied for extra speed, much better at for loops then plain python or numpy. This function is used when the output is a scalar, no tensor indices needed.

    Parameters
    ----------
    T1 : 1D numpy array of 2^(T1ix.shape), float
        Multi dimensional tensor flattened to a vector.

    T1ix : 1D numpy array of size n, int
        Tensor indices for T1. T1 is addressed by:
        f(T1ix[0])*2^0+f(T1ix[1])*2^1+f(T1ix[2])*2^2+...+f(T1ix[n-1])*2^(n-1)
        where f is some function that returns the value of index k.

    T2 : 1D numpy array of 2^(T2ix.shape), float
        Multi dimensional tensor flattened to a vector.

    T2ix : 1D numpy array of size n, int
        Tensor indices for T2. T2 is addressed by:
        f(T2ix[0])*2^0+f(T2ix[1])*2^1+f(T2ix[2])*2^2+...+f(T2ix[n-1])*2^(n-1)
        where f is some function that returns the value of index k.

    Returns
    -------
    Tout : float
        The result of the tensor product

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Examples
    --------
    >>> T1=[1,2,3,4,5,6,7,8]
    >>> T1ix=[0,1,2]
    >>> T2=[1,2,4,8,16,32,64,128]
    >>> T2ix=[0,1,2]
    >>> Tout=CoreLoopScalar(T1,T1ix,T2,T2ix)
    '''
    # determine size of bit array to iterate through
    Sizeix=max((numpy.max(T1ix),numpy.max(T2ix)))+1

    # get number of indices
    T1s=T1ix.shape[0]
    T2s=T2ix.shape[0]

    # allocate output scalar
    Tout=0

    # allocate bit array
    # is really an array for determining if index j is 1,
    # how much should the tensor vector index shift?
    BitArray=numpy.zeros((2,Sizeix),dtype=numpy.int64)
    for zz in range(0,T1s):
        BitArray[0,T1ix[zz]]=(1<<zz)
    for zz in range(0,T2s):
        BitArray[1,T2ix[zz]]=(1<<zz)

    # loop over all permutations of the tensor indices
    # k is used as a bit mask
    for k in range(0,2**Sizeix):
        T1p=0
        T2p=0
        # scan through the bit array,
        # applying the proper masking and index shifting
        for b in range(0,Sizeix):
            TT=bool((1<<b)&k)
            T1p=T1p+BitArray[0,b]*TT
            T2p=T2p+BitArray[1,b]*TT

        # perform the multiplication and addition to output scalar
        Tout=Tout+T1[T1p]*T2[T2p]

    return Tout

@jit(nopython=True,cache=True)
def EdgeWeight(T1,pf,Eix):
    '''
    This function contracts tensor T1 with a variable constraint tensor.

    Parameters
    ----------
    T1 : 1D numpy array of 2^(T1ix.shape), float
        Multi dimensional tensor flattened to a vector.

    pf : float
        probability of the edge failing, edge unreliability

    Eix : int
        Tensor index being contracted with the variable constraint tensor.

    Returns
    -------
    Tout : float
        The result of the tensor product with the variable constraint tensor.

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Examples
    --------
    >>> T1=[1,2,3,4,5,6,7,8]
    >>> pf=0.75
    >>> Eix=2
    >>> Tout=EdgeWeight(T1,pf,Eix)
    '''

    # allocate output tensor
    Tout=numpy.zeros(T1.shape[0])

    # loop over all permutations of the tensor indices
    for k in range(0,T1.shape[0]):

        TT=bool((1<<Eix)&k)
        # when index Eix is 1, multiply by chance of edge not failing,
        # edge reliability
        # when index Eix is 0, multiply by chance of edge failing.
        Tout[k]=T1[k]*numpy.abs(TT-pf)

    return Tout


def Main(Nodes,EdgeOrder,pf=.5,verbose=True):
    '''
    This main function performs the tensor network contraction.
    This outer loop is done in python, for ease of string, dictionary, list, and other mutable manipulation.

    Parameters
    ----------
    Nodes : list of tuples, (1D numpy array, (tuple of indices))
        Each node is a tensor in the tensor network. Nodes[i][0] defines the ith tensor, and Nodes[i][1] defines the indices of this tensor. Each index aligns with an edge, a properly specified tensor network will have just two occurrences of an index.

    EdgeOrder : list of tuples, int
        A list of the edges in the order they should be contracted.

    pf : float
        A uniform edge probability. If len(EdgeOrder[i])=3, then the edge specifies its probability of existing, its reliability, and will override this global value.

    verbose : bool
        boolean to control how much text is displayed to the console

    Returns
    -------
    Tout : float
        The result of the tensor network contraction

    CalculationEffort : int
        The number of floating point operations used to perform the entire calculation.

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Currently the code assumes the edge cover problem, and adds in the variable constraint tensors as needed. This code could be modified to be more general purpose.

    Examples
    --------
    >>> T=[]
    >>> for k in range(2,5):
    >>>     TT=numpy.ones(2**k)
    >>>     TT[0]=0
    >>>     T.append(TT)

    >>> Nodes=[]
    >>> Nodes.append((T[0],(1,3)))
    >>> Nodes.append((T[1],(1,4,2)))
    >>> Nodes.append((T[0],(2,5)))
    >>> Nodes.append((T[1],(3,6,8)))
    >>> Nodes.append((T[2],(4,6,7,9)))
    >>> Nodes.append((T[1],(5,7,10)))
    >>> Nodes.append((T[0],(8,11)))
    >>> Nodes.append((T[1],(11,9,12)))
    >>> Nodes.append((T[0],(12,10)))
    >>> EdgeOrder=list(range(1,13))

    >>> Tout,CalculationEffort = Main(Nodes,EdgeOrder,pf=.9)
    '''
    # initialize variables
    Tout=0
    CalculationEffort=0
    EdgeToNode=dict()
    NodeToEdge=dict()
    NodeToEdgeix=len(Nodes)

    # given an edge, determine the nodes it is connected too
    for loc,n in enumerate(Nodes):

        for e in n[1]:
            if e not in EdgeToNode:
                EdgeToNode[e]=[]
            EdgeToNode[e].append(loc)

    # loop over each edge
    for k in range(0,len(EdgeOrder)):
        ee=EdgeOrder[k]
        if verbose:
            print(k,ee)

        Cnodes=[]
        # determine the tensors connected to the edge
        Cnodes=list(EdgeToNode[ee])

        # if no connected tensors, that means the edge was already contracted
        # in a parallel process
        if Cnodes==[]:
            if verbose:
                print('Edge Already Contracted')
            continue

        # if the two connected tensors are the same, the edge is a loop
        if Cnodes[0]==Cnodes[1]:
            NN=Cnodes[1]
            Cnodes[0]=Nodes[NN]
            Cnodes[1]=Nodes[NN]

            Nodes[NN]=[]

        else:
            # load tensor into local memory
            NN=Cnodes[1]
            Cnodes[1]=Nodes[NN]
            Nodes[NN]=[]

            NN=Cnodes[0]
            Cnodes[0]=Nodes[NN]
            Nodes[NN]=[]

        # determine the indices that will be contracted
        Cedges=list(set(Cnodes[0][1]).intersection(set(Cnodes[1][1])))

        # pf=.01
        if len(ee)==3:
            # override global pf if the edge specifies the edge reliability
            pf=1-ee[2]
        # for every index about to be contracted
        # make sure to merge in the degree 2 variable constraint tensors for
        # the edge cover problem
        for e in Cedges:
            if Cnodes[0][0].shape[0]<Cnodes[1][0].shape[0]:
                Eix=Cnodes[0][1].index(e)
                Cnodes[0]=(EdgeWeight(Cnodes[0][0],pf,Eix),Cnodes[0][1])
            else:
                Eix=Cnodes[1][1].index(e)
                Cnodes[1]=(EdgeWeight(Cnodes[1][0],pf,Eix),Cnodes[1][1])

        # get list of all participating indicies in the contraction
        Tix=[]
        Tix.extend(list(set(Cnodes[0][1]).union(set(Cnodes[1][1]))))

        # reindex edges from 0 to n
        NodeTix=dict()
        ixTnode=dict()
        zz=0
        for ix in Tix:
            NodeTix[ix]=zz
            ixTnode[zz]=ix
            zz=zz+1

        T1ix=[]
        for ix in Cnodes[0][1]:
            T1ix.append(NodeTix[ix])
        T1ix=numpy.array(T1ix)

        T2ix=[]
        for ix in Cnodes[1][1]:
            T2ix.append(NodeTix[ix])
        T2ix=numpy.array(T2ix)

        # find indices participating but not being contracted.
        Toutix=list(set(T1ix).symmetric_difference(set(T2ix)))

        # remove the tensors from being considered in the future
        MergeNodes=list(EdgeToNode[ee])
        for e in list(set(Cnodes[0][1]).symmetric_difference(set(Cnodes[1][1]))):
            for m in MergeNodes:
                try:
                    EdgeToNode[e].remove(m)
                except ValueError:
                    pass
        for e in list(set(Cnodes[0][1]).intersection(set(Cnodes[1][1]))):
            EdgeToNode[e]=[]

        # calculate the calculation effort
        CalculationEffort=CalculationEffort+2**len(NodeTix)
        if verbose:
            print(len(NodeTix))
            print(CalculationEffort)

        # determine if the output is a tensor or scalar
        # then call the correct function
        if len(Toutix)==0:
            Tout=CoreLoopScalar(Cnodes[0][0],T1ix,Cnodes[1][0],T2ix)
        else:
            Tout=CoreLoop(Cnodes[0][0],T1ix,Cnodes[1][0],T2ix,numpy.array(Toutix))

        # convert the computational basis indices into real indices
        for loc,ix in list(enumerate(Toutix)):
            Toutix[loc]=ixTnode[Toutix[loc]]
        Nodes.append((Tout,Toutix))

        # add the new tensor on the list of tensors that can be contracted
        for e in Nodes[NodeToEdgeix][1]:
            if e not in EdgeToNode:
                EdgeToNode[e]=[]
            EdgeToNode[e].append(NodeToEdgeix)

        NodeToEdgeix=NodeToEdgeix+1

    if verbose:
        print('CalculationEffort',CalculationEffort)
    return Tout,CalculationEffort


if __name__ == "__main__":
    ### Generate Grids
    NN=3

    T=[0,0]
    for k in range(2,5):
        TT=numpy.ones(2**k)
        TT[0]=0
        T.append(TT)

    Nodes=[]
    for k1 in range(0,NN):
        for k2 in range(0,NN):
            ix=[0,0,0,0]
            ix[0]=k1*(2*NN-1)+k2
            ix[1]=k1*(2*NN-1)+k2-1
            ix[2]=k1*(2*NN-1)+k2+NN-1
            ix[3]=k1*(2*NN-1)+k2-NN
            if k2==0:
                ix.remove(k1*(2*NN-1)+k2-1)
            if k2==NN-1:
                ix.remove(k1*(2*NN-1)+k2)
            if k1==0:
                ix.remove(k1*(2*NN-1)+k2-NN)
            if k1==NN-1:
                ix.remove(k1*(2*NN-1)+k2+NN-1)
            Nodes.append((T[len(ix)],tuple(ix)))

    EdgeOrder=list(range(0,(NN-1)*NN*2))
    print(Nodes)
    print(EdgeOrder)
    # input()
    t=time.time()
    Tout,CalculationEffort=Main(Nodes,EdgeOrder,.5)
    print(Tout)
    print(time.time()-t)
    sys.exit()

    T=[]
    for k in range(2,5):
        TT=numpy.ones(2**k)
        TT[0]=0
        T.append(TT)

    Nodes=[]
    Nodes.append((T[0],(1,3)))
    Nodes.append((T[1],(1,4,2)))
    Nodes.append((T[0],(2,5)))
    Nodes.append((T[1],(3,6,8)))
    Nodes.append((T[2],(4,6,7,9)))
    Nodes.append((T[1],(5,7,10)))
    Nodes.append((T[0],(8,11)))
    Nodes.append((T[1],(11,9,12)))
    Nodes.append((T[0],(12,10)))

    Main(Nodes,list(range(1,13)))
    sys.exit()
    T1=numpy.array([1,2,3,4])
    T2=numpy.array([1,2,3,4])
    T1ix=(3,4)
    T2ix=(4,5)
    # Toutix=(0,2)
    # CoreLoop(T1,T1ix,T2,T2ix,Toutix)
    Nodes=[]
    Nodes.append((T1,T1ix))
    Nodes.append((T2,T2ix))
    Main(Nodes,[4])




    # sys.exit()
