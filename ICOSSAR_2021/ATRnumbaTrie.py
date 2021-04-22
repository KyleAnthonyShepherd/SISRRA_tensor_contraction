'''
This module calculates the all terminal reliability of undirected graphs

Functions
---------
**ATR(EdgeList,EP,verbose=False,TrieHashThreshold=10) :**  
    Calculates the All Terminal Reliability of a undirected graph defined by EdgeList.

Notes
-----
When these functions are first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

Currently just for All Terminal Reliability. K-Terminal Reliability can be implemented in the future.

Examples
--------
>>> import ATRnumbaTrie
>>> n=5
>>> EdgeList=[]
>>> for k1 in range(0,n-1):
>>>     for k2 in range(k1+1,n):
>>>         EdgeList.append((k1,k2))

>>> EdgeList=numpy.array(EdgeList)
>>> EP=numpy.ones(EdgeList.shape[0])*.5
>>> REL,SetixTotal,SetixMax=ATR(EdgeList,EP)
'''

import sys

import math

import numpy

from numba import jit

# JUST ASSUME ATR, and ASSUME FLOAT

# bi directional edges
@jit(nopython=True,cache=True)
def ATR(EdgeList,EP,verbose=False,TrieHashThreshold=10):
    '''
    The numba optimized function that calculates all terminal reliability (ATR) for an undirected graph defined by a list of edges.

    Parameters
    ----------
    EdgeList : jx2 numpy array, int
        EdgeList[i,0] and EdgeList[i,0] are the labels of the head node and tail node of edge i. j is the number of edges in the graph. Labels must be consecutive from 0 to n-1, where n is the number of nodes in the graph. The algorithm factors the graph from edge 0 to edge j. An optimized edge ordering should be fed into this function.

    EP : 1D numpy array of size n, float
        EP[i] is the probability of edge i existing, probability of edge i not failing.

    verbose : bool
        Controls verbosity to console when the function is called

    TrieHashThreshold : int
        If the edge node set is equal to or exceeds this value, a hash table is used instead of a trie memory structure. The hash table is slower (due to collision checking), but takes up less memory.

    Returns
    -------
    FinalProb : float
        The all terminal reliability of the graph.

    SetixTotal : int
        Total number of subgraphs considered in the algorithm

    SetixMax : int
        Maximum number of subgraphs considered in the algorithm at any one factoring step

    Notes
    -----
    Currently just All Terminal Reliability. K-Terminal Reliability could be implemented.

    Examples
    --------
    >>> n=5
    >>> EdgeList=[]
    >>> for k1 in range(0,n-1):
    >>>     for k2 in range(k1+1,n):
    >>>         EdgeList.append((k1,k2))

    >>> EdgeList=numpy.array(EdgeList)
    >>> EP=numpy.ones(EdgeList.shape[0])*.5
    >>> REL,SetixTotal,SetixMax=ATR(EdgeList,EP)
    '''

    # get number of nodes
    # assuming EdgeList is preprocessed
    NN=numpy.max(EdgeList)+1

    Gworkspace=numpy.ones((2*NN,2*NN),dtype=numpy.int32)*-1
    Gcount=numpy.zeros(2*NN,dtype=numpy.int32)

    # set up graph connectivity
    for Edge in range(0,EdgeList.shape[0]):
        e0=EdgeList[Edge,0]
        e1=EdgeList[Edge,1]
        Gworkspace[e0,Gcount[e0]]=e1
        Gcount[e0]=Gcount[e0]+1
        Gworkspace[e1,Gcount[e1]]=e0
        Gcount[e1]=Gcount[e1]+1

    ExistingNodes=numpy.ones(NN)

    # Trie data structure
    Setlen=numpy.uint64(1)
    Setix=1
    SetixTotal=0
    SetixMax=0
    Setix0=2
    InitSize=numpy.int64(2**4)
    Trie=numpy.zeros((InitSize,Setlen+numpy.uint64(1)),dtype=numpy.int64)
    SetStorage0=numpy.zeros((InitSize,(Setlen)*numpy.uint64(2)),dtype=numpy.int16)
    SetP0=numpy.zeros(InitSize,dtype=numpy.float64)

    # Hash data structure
    HashTable=numpy.zeros(InitSize,dtype=numpy.int64)
    LLix=numpy.ones(InitSize,dtype=numpy.int64)*-1

    # define first set
    SetStorage0[1,:]=numpy.array([0,-1])
    SetP0[1]=1

    # Node to ID vect
    # Used for Trie tracking
    NodeTTrieIx=numpy.ones(NN+1,dtype=numpy.int32)*-1
    NodeTTrieIx[0]=0

    # Path Edge Set
    PathEdgeSet=numpy.zeros(NN,dtype=numpy.int32)
    PathEdgeSet[0]=1

    # Main loop
    for Edge in range(0,EdgeList.shape[0]):
        if verbose:
            print(Edge)
        ee=EdgeList[Edge,:]

        # connectivity check start node
        StartNode=0
        for ix in range(NN-1,-1,-1):
            if Gcount[ix]==0:
                StartNode=ix
                # ExistingNodeSum=ExistingNodeSum-1

        # make graph with edge e removed
        cc=0
        for ix in range(0,Gcount[ee[0]]):
            if Gworkspace[ee[0],ix]!=ee[1]:
                Gworkspace[ee[0],cc]=Gworkspace[ee[0],ix]
                cc=cc+1
        cc=0
        for ix in range(0,Gcount[ee[1]]):
            if Gworkspace[ee[1],ix]!=ee[0]:
                Gworkspace[ee[1],cc]=Gworkspace[ee[1],ix]
                cc=cc+1

        ExistingNodeSum=NN
        for ix in range(0,NN):
            if Gcount[ix]==0:
                ExistingNodeSum=ExistingNodeSum-1

        Gcount[ee[0]]=Gcount[ee[0]]-1
        Gcount[ee[1]]=Gcount[ee[1]]-1

        Gactive=numpy.copy(Gworkspace)
        GcountC=numpy.copy(Gcount)

        #update edge set
        # keep track of added and removed nodes from the set
        NewNode=numpy.array([0,0,0])
        if PathEdgeSet[ee[0]]==0:
            PathEdgeSet[ee[0]]=1
            NewNode[0]=NewNode[0]+1
            NewNode[NewNode[0]]=ee[0]
        if PathEdgeSet[ee[1]]==0:
            PathEdgeSet[ee[1]]=1
            NewNode[0]=NewNode[0]+1
            NewNode[NewNode[0]]=ee[1]

        # RemovedNode=-1
        RemovedNode=numpy.zeros(NN)
        # ExistingNodeSum=NN
        for ix in range(0,NN):
            if Gcount[ix]==0:
                PathEdgeSet[ix]=0
                RemovedNode[ix]=1
                # ExistingNodeSum=ExistingNodeSum-1

        # update Node to ID vect
        NodeTTrieIx[:]=-1
        Setlen=numpy.uint64(0)
        cc=0
        StartNode=0
        for ix in range(NN-1,-1,-1):
            if PathEdgeSet[ix]==1:
                NodeTTrieIx[ix]=cc
                cc=cc+1
                Setlen=Setlen+numpy.uint64(1)
                StartNode=ix

        # create set handling array
        Setworkspace=numpy.zeros((Setlen+numpy.uint64(3),Setlen+numpy.uint64(3)))
        Setworklen=numpy.zeros(Setlen+numpy.uint64(3),dtype=numpy.int32)
        Setmarked=numpy.zeros(2,dtype=numpy.int8)

        # print(Setlen)
        # print(PathEdgeSet)
        # print(ee)
        # print(numpy.where(PathEdgeSet>0))
        # input()
        NewSet=numpy.zeros(Setlen*numpy.uint64(2),dtype=numpy.int64)

        # create set storage arrays
        Setix=1
        # Tlen=1
        # TrieSize=(Setix0*2+1)
        # Trie=numpy.zeros((TrieSize,Setlen+1),dtype=numpy.int64)
        SetStorage=numpy.zeros((Setix0*2+1,(Setlen)*numpy.uint64(2)),dtype=numpy.int16)
        SetP=numpy.zeros(Setix0*2+1,dtype=numpy.float64)

        # SetThresh=0
        if Setlen>TrieHashThreshold:
            Tlen=1
            TrieSize=1
            Trie=numpy.zeros((TrieSize,Setlen+numpy.uint64(1)),dtype=numpy.int64)
            # Hash storage arrays
            HashTable=numpy.ones(int((Setix0*2+1)*1.3),dtype=numpy.int64)*-1
            LLix=numpy.ones(Setix0*2+1,dtype=numpy.int64)*-1
        else:
            Tlen=1
            TrieSize=(Setix0*2+1)
            Trie=numpy.zeros((TrieSize,Setlen+numpy.uint64(1)),dtype=numpy.int64)
            # Hash storage arrays
            HashTable=numpy.ones(1,dtype=numpy.int64)*-1
            LLix=numpy.ones(1,dtype=numpy.int64)*-1


        # create connectivity check arrays
        VisitedNodes=numpy.zeros(NN,dtype=numpy.int32)
        NextNodes=numpy.zeros(NN,dtype=numpy.int32)

        # get edge prob
        EdgeProb=float(EP[Edge])

        if Setlen==0:
            FinalProb=0
            # print(SetStorage0[:Setix0,:])
            # print(SetP0[:Setix0])
            for zz in range(1,Setix0):
                Curset=SetStorage0[zz,:]
                if Curset[1]==-1:
                    FinalProb=FinalProb+SetP0[zz]*EdgeProb
                else:
                    FinalProb=FinalProb+SetP0[zz]
            # print('FinalProb')
            # print(FinalProb)
            # print('Obtained All Terminal Reliability')
            return FinalProb,SetixTotal,SetixMax

        # Graph loop
        for zz in range(1,Setix0):
            # print(SetStorage0)
            Curset=SetStorage0[zz,:]
            setProb=SetP0[zz]
            # Setworkspace[:]=0
            Setworklen[:]=0
            Setmarked[:]=-1
            SetmarkedC=0
            # Setmarked[0]=1
            SetNum=0

            # Gactive[:]=numpy.copy(Gworkspace)
            # Gactive[:]=Gworkspace[:]
            GcountC[:]=Gcount[:]

            # read the set
            # print('set')
            # print(set)
            Cnode=-1
            Mark0=False
            Mark1=False
            for k in range(0,Curset.shape[0]):
                ss=Curset[k]
                if ss<0:
                    Cnode=-1
                    if Setworklen[SetNum]>0:
                        # advance to next set
                        if Mark0:
                            Setmarked[SetmarkedC]=SetNum
                            SetmarkedC=SetmarkedC+1
                        if Mark1:
                            Setmarked[SetmarkedC]=SetNum
                            SetmarkedC=SetmarkedC+1

                        Setworkspace[SetNum,:Setworklen[SetNum]].sort()
                        SetNum=SetNum+1
                    Mark0=False
                    Mark1=False
                else:
                    # if first node seen in set
                    if Cnode==-1:
                        Cnode=ss
                    else:
                        # Add edge connecting nodes
                        Gactive[Cnode,GcountC[Cnode]]=ss
                        GcountC[Cnode]=Gcount[Cnode]+1
                        Gactive[ss,GcountC[ss]]=Cnode
                        GcountC[ss]=GcountC[ss]+1

                    # add to set workspace
                    if RemovedNode[ss]==0: # only if not a removed node
                        Setworkspace[SetNum,Setworklen[SetNum]]=ss
                        Setworklen[SetNum]=Setworklen[SetNum]+1
                        # determine what sets to merge, due to contracted edge
                    if ss==ee[0]:
                        # Setmarked[0]=SetNum
                        Mark0=True
                    if ss==ee[1]:
                        # Setmarked[1]=SetNum
                        Mark1=True

            # add new sets
            for k in range(0,NewNode[0]):
                Setworkspace[SetNum,Setworklen[SetNum]]=NewNode[k+1]
                Setworklen[SetNum]=Setworklen[SetNum]+1
                # Setmarked[1-k]=SetNum # Check this later
                Setmarked[SetmarkedC]=SetNum
                SetmarkedC=SetmarkedC+1
                SetNum=SetNum+1


            # if edge is a loop, no disconnection chance
            if Setmarked[0]==Setmarked[1]:
                ## build new set for storage
                ix=0
                NewSet[:]=-1
                # get set order
                SetSort=numpy.argsort(Setworkspace[:SetNum,0])
                for k in range(0,SetNum):
                    ID=SetSort[k]
                    NewSet[ix:ix+Setworklen[ID]]=Setworkspace[ID,:Setworklen[ID]]
                    ix=ix+Setworklen[ID]+1


                if Setlen>TrieHashThreshold:
                    # GET HASH
                    acc=numpy.uint64(2870177450012600261)
                    for vv in range(0,Setlen*numpy.uint64(2)):
                        acc=acc+numpy.int64(NewSet[vv]).view(numpy.uint64)*numpy.uint64(14029467366897019727)
                        acc=((acc << numpy.uint64(31)) | (acc >> numpy.uint64(33)))
                        acc=acc*11400714785074694791
                    acc = acc+(Setlen*numpy.uint64(2)) ^ (numpy.uint64(2870177450012600261) ^ numpy.uint64(3527539))
                    acc=acc%numpy.uint64(int((Setix0*2+1)*1.3))

                    if HashTable[acc]==-1:
                        SetStorage[Setix,:]=NewSet
                        SetP[Setix]=setProb
                        HashTable[acc]=Setix
                        Setix=Setix+1
                    else:
                        II0=HashTable[acc]
                        while II0!=-1:
                            II=int(II0)
                            if numpy.all(NewSet==SetStorage[II,:]):
                                SetP[II]=SetP[II]+setProb
                                break
                            else:
                                II0=LLix[II]
                        if II0==-1:
                            SetStorage[Setix,:]=NewSet
                            SetP[Setix]=setProb
                            LLix[II]=Setix
                            Setix=Setix+1
                else:
                    # Insert into Trie
                    # print(NewSet)
                    Tix=0
                    for vv in range(0,Setlen*2):
                        ss=NewSet[vv]
                        II=Trie[Tix,NodeTTrieIx[ss]]
                        if II==0:
                            SetStorage[Setix,:]=NewSet
                            SetP[Setix]=setProb
                            Trie[Tix,NodeTTrieIx[ss]]=-Setix
                            Setix=Setix+1
                            break

                        if II<0:
                            if vv==Setlen*2-1:
                                SetP[-II]=SetP[-II]+setProb
                                break
                            InSet=SetStorage[-II,:]
                            Trie[Tix,NodeTTrieIx[ss]]=Tlen
                            Trie[Tlen,NodeTTrieIx[InSet[vv+1]]]=II
                            Tix=int(Tlen)
                            Tlen=Tlen+1
                            # lazy increase size of Trie
                            if Tlen>=TrieSize:
                                Trie=numpy.append(Trie,numpy.zeros((TrieSize,Setlen+1),dtype=numpy.int64),axis=0)
                                TrieSize=TrieSize*2
                        else:
                            Tix=int(II)
                continue

            # if set to be contracted is empty due to deleted node
            if Setmarked[0]==-1 or Setmarked[1]==-1:
                ## build new set for storage
                ix=0
                NewSet[:]=-1
                # get set order
                SetSort=numpy.argsort(Setworkspace[:SetNum,0])
                # print(Setworkspace)
                # print(Setworklen)
                for k in range(0,SetNum):
                    ID=SetSort[k]
                    # print(ID)
                    # print(NewSet)
                    NewSet[ix:ix+Setworklen[ID]]=Setworkspace[ID,:Setworklen[ID]]
                    ix=ix+Setworklen[ID]+1


                if Setlen>TrieHashThreshold:
                    # GET HASH
                    acc=numpy.uint64(2870177450012600261)
                    for vv in range(0,Setlen*numpy.uint64(2)):
                        acc=acc+numpy.int64(NewSet[vv]).view(numpy.uint64)*numpy.uint64(14029467366897019727)
                        acc=((acc << numpy.uint64(31)) | (acc >> numpy.uint64(33)))
                        acc=acc*11400714785074694791
                    acc = acc+(Setlen*numpy.uint64(2)) ^ (numpy.uint64(2870177450012600261) ^ numpy.uint64(3527539))
                    acc=acc%numpy.uint64(int((Setix0*2+1)*1.3))

                    if HashTable[acc]==-1:
                        SetStorage[Setix,:]=NewSet
                        SetP[Setix]=setProb*EdgeProb
                        HashTable[acc]=Setix
                        Setix=Setix+1
                    else:
                        II0=HashTable[acc]
                        while II0!=-1:
                            II=int(II0)
                            if numpy.all(NewSet==SetStorage[II,:]):
                                SetP[II]=SetP[II]+setProb*EdgeProb
                                break
                            else:
                                II0=LLix[II]
                        if II0==-1:
                            SetStorage[Setix,:]=NewSet
                            SetP[Setix]=setProb*EdgeProb
                            LLix[II]=Setix
                            Setix=Setix+1
                else:
                    # Insert into Trie
                    # print(NewSet)
                    Tix=0
                    for vv in range(0,Setlen*2):
                        ss=NewSet[vv]
                        II=Trie[Tix,NodeTTrieIx[ss]]
                        if II==0:
                            SetStorage[Setix,:]=NewSet
                            SetP[Setix]=setProb*EdgeProb
                            Trie[Tix,NodeTTrieIx[ss]]=-Setix
                            Setix=Setix+1
                            break

                        if II<0:
                            if vv==Setlen*2-1:
                                SetP[-II]=SetP[-II]+setProb*EdgeProb
                                break
                            InSet=SetStorage[-II,:]
                            Trie[Tix,NodeTTrieIx[ss]]=Tlen
                            Trie[Tlen,NodeTTrieIx[InSet[vv+1]]]=II
                            Tix=int(Tlen)
                            Tlen=Tlen+1
                            # lazy increase size of Trie
                            if Tlen>=TrieSize:
                                Trie=numpy.append(Trie,numpy.zeros((TrieSize,Setlen+1),dtype=numpy.int64),axis=0)
                                TrieSize=TrieSize*2
                        else:
                            Tix=int(II)
                continue


            VisitedNodes[:]=0
            VisitedNodes[EdgeList[-1,0]]=1
            NextNodes[:]=-1
            NextNodes[0]=EdgeList[-1,0]
            NNix=1

            for con in range(0,NN):
                ConNode=NextNodes[con]
                if ConNode==-1:
                    break
                for e in range(0,GcountC[ConNode]):
                    AdjNode=Gactive[ConNode,e]
                    if VisitedNodes[AdjNode]==0:
                        VisitedNodes[AdjNode]=1
                        NextNodes[NNix]=AdjNode
                        NNix=NNix+1

            if NNix==ExistingNodeSum:
                Connected=True
            else:
                Connected=False

            # print(EdgeList[Edge,:])
            # print(Curset)
            # print(Connected)
            # print(VisitedNodes)
            # input()

            ########################### connectivity check
            # VisitedNodes[:]=0
            # VisitedNodes[:StartNode+1]=1
            # NextNodes[:]=-1
            # NextNodes[StartNode]=StartNode
            # NNix=StartNode+1
            #
            # for con in range(StartNode,NN):
            #     ConNode=NextNodes[con]
            #     if ConNode==-1:
            #         break
            #     for e in range(0,GcountC[ConNode]):
            #         AdjNode=Gactive[ConNode,e]
            #         if VisitedNodes[AdjNode]==0:
            #             VisitedNodes[AdjNode]=1
            #             NextNodes[NNix]=AdjNode
            #             NNix=NNix+1
            #
            # if NNix-StartNode==ExistingNodeSum:
            #     Connected=True
            # else:
            #     Connected=False
            ########################### connectivity check END

            # print('Connected',Connected)
            # print(VisitedNodes)
            if Connected==True:
                # consider not contracted set case
                ## build new set for storage

                ix=0
                NewSet[:]=-1
                # get set order
                SetSort=numpy.argsort(Setworkspace[:SetNum,0])
                for k in range(0,SetNum):
                    ID=SetSort[k]
                    NewSet[ix:ix+Setworklen[ID]]=Setworkspace[ID,:Setworklen[ID]]
                    ix=ix+Setworklen[ID]+1

                if Setlen>TrieHashThreshold:
                    # GET HASH
                    acc=numpy.uint64(2870177450012600261)
                    for vv in range(0,Setlen*numpy.uint64(2)):
                        acc=acc+numpy.int64(NewSet[vv]).view(numpy.uint64)*numpy.uint64(14029467366897019727)
                        acc=((acc << numpy.uint64(31)) | (acc >> numpy.uint64(33)))
                        acc=acc*11400714785074694791
                    acc = acc+(Setlen*numpy.uint64(2)) ^ (numpy.uint64(2870177450012600261) ^ numpy.uint64(3527539))
                    acc=acc%numpy.uint64(int((Setix0*2+1)*1.3))

                    if HashTable[acc]==-1:
                        SetStorage[Setix,:]=NewSet
                        SetP[Setix]=setProb*(1-EdgeProb)
                        HashTable[acc]=Setix
                        Setix=Setix+1
                    else:
                        II0=HashTable[acc]
                        while II0!=-1:
                            II=int(II0)
                            if numpy.all(NewSet==SetStorage[II,:]):
                                SetP[II]=SetP[II]+setProb*(1-EdgeProb)
                                break
                            else:
                                II0=LLix[II]
                        if II0==-1:
                            SetStorage[Setix,:]=NewSet
                            SetP[Setix]=setProb*(1-EdgeProb)
                            LLix[II]=Setix
                            Setix=Setix+1
                else:
                    # Insert into Trie
                    # print(NewSet)
                    Tix=0
                    for vv in range(0,Setlen*2):
                        ss=NewSet[vv]
                        II=Trie[Tix,NodeTTrieIx[ss]]
                        # print(ss,II)
                        if II==0:
                            SetStorage[Setix,:]=NewSet
                            SetP[Setix]=setProb*(1-EdgeProb)
                            Trie[Tix,NodeTTrieIx[ss]]=-Setix
                            Setix=Setix+1
                            break

                        if II<0:
                            if vv==Setlen*2-1:
                                SetP[-II]=SetP[-II]+setProb*(1-EdgeProb)
                                break
                            InSet=SetStorage[-II,:]
                            Trie[Tix,NodeTTrieIx[ss]]=Tlen
                            Trie[Tlen,NodeTTrieIx[InSet[vv+1]]]=II
                            Tix=int(Tlen)
                            Tlen=Tlen+1
                            # lazy increase size of Trie
                            if Tlen>=TrieSize:
                                Trie=numpy.append(Trie,numpy.zeros((TrieSize,Setlen+1),dtype=numpy.int64),axis=0)
                                TrieSize=TrieSize*2
                        else:
                            Tix=int(II)
                    # print('Trie')
                    # print(NodeTTrieIx)
                    # print(Trie)

            # only need to consider contracted set case
            ## build new set for storage
            # merge sets
            for k in range(0,Setworklen[Setmarked[1]]):
                Setworkspace[Setmarked[0],Setworklen[Setmarked[0]]]=Setworkspace[Setmarked[1],k]
                Setworklen[Setmarked[0]]=Setworklen[Setmarked[0]]+1
                Setworkspace[Setmarked[1],k]=-1

            Setworkspace[Setmarked[0],:Setworklen[Setmarked[0]]].sort()
            ix=0
            NewSet[:]=-1
            # get set order
            SetSort=numpy.argsort(Setworkspace[:SetNum,0])
            for k in range(1,SetNum):
                ID=SetSort[k]
                NewSet[ix:ix+Setworklen[ID]]=Setworkspace[ID,:Setworklen[ID]]
                ix=ix+Setworklen[ID]+1

            if Setlen>TrieHashThreshold:
                # GET HASH
                acc=numpy.uint64(2870177450012600261)
                for vv in range(0,Setlen*numpy.uint64(2)):
                    acc=acc+numpy.int64(NewSet[vv]).view(numpy.uint64)*numpy.uint64(14029467366897019727)
                    acc=((acc << numpy.uint64(31)) | (acc >> numpy.uint64(33)))
                    acc=acc*11400714785074694791
                acc = acc+(Setlen*numpy.uint64(2)) ^ (numpy.uint64(2870177450012600261) ^ numpy.uint64(3527539))
                acc=acc%numpy.uint64(int((Setix0*2+1)*1.3))

                if HashTable[acc]==-1:
                    SetStorage[Setix,:]=NewSet
                    SetP[Setix]=setProb*EdgeProb
                    HashTable[acc]=Setix
                    Setix=Setix+1
                else:
                    II0=HashTable[acc]
                    while II0!=-1:
                        II=int(II0)
                        if numpy.all(NewSet==SetStorage[II,:]):
                            SetP[II]=SetP[II]+setProb*EdgeProb
                            break
                        else:
                            II0=LLix[II]
                    if II0==-1:
                        SetStorage[Setix,:]=NewSet
                        SetP[Setix]=setProb*EdgeProb
                        LLix[II]=Setix
                        Setix=Setix+1
            else:
                # Insert into Trie
                # print(NewSet)
                Tix=0
                for vv in range(0,Setlen*2):
                    ss=NewSet[vv]
                    II=Trie[Tix,NodeTTrieIx[ss]]
                    if II==0:
                        SetStorage[Setix,:]=NewSet
                        SetP[Setix]=setProb*EdgeProb
                        Trie[Tix,NodeTTrieIx[ss]]=-Setix
                        Setix=Setix+1
                        break

                    if II<0:
                        if vv==Setlen*2-1:
                            SetP[-II]=SetP[-II]+setProb*EdgeProb
                            break
                        InSet=SetStorage[-II,:]
                        Trie[Tix,NodeTTrieIx[ss]]=Tlen
                        Trie[Tlen,NodeTTrieIx[InSet[vv+1]]]=II
                        Tix=int(Tlen)
                        Tlen=Tlen+1
                        # lazy increase size of Trie
                        if Tlen>=TrieSize:
                            Trie=numpy.append(Trie,numpy.zeros((TrieSize,Setlen+1),dtype=numpy.int64),axis=0)
                            TrieSize=TrieSize*2
                    else:
                        Tix=int(II)
            continue

        SetStorage0=numpy.copy(SetStorage)
        SetP0=numpy.copy(SetP)
        Setix0=int(Setix)

        # print(ee)
        # print(SetStorage0)
        # print(SetP0)
        # print(Trie[:Tlen,:])
        # print(Tlen/Setix)
        # print(Tlen)
        # print('ErrorCheck')
        # print(SetStorage)
        # print(SetP)
        # print(numpy.sum(SetP))
        # print(Curset)
        # print(ee)
        # print(Setmarked)
        # input()

        SetixTotal=SetixTotal+Setix
        SetixMax=max(SetixMax,Setix)
        if verbose:
            print(Setix)
            print(SetixTotal)
            print(TrieHashThreshold)


if __name__ == "__main__":

    # tests are stored here

    ############ Complete Graph Test
    n=5
    EdgeList=[]
    for k1 in range(0,n-1):
        for k2 in range(k1+1,n):
            EdgeList.append((k1,k2))
    ############ Complete Graph Test END

    ############ Grid Graph Test
    # n=12
    # EdgeList=[]
    # for k1 in range(0,n):
    #     for k2 in range(0,n-1):
    #         EdgeList.append((n*k1+k2,n*k1+k2+1))
    #
    # for k1 in range(0,n-1):
    #     for k2 in range(0,n):
    #         EdgeList.append((n*k1+k2,n*(k1+1)+k2))
    #
    # EdgeList=numpy.array(EdgeList)
    # ix=numpy.argsort(EdgeList[:,0])
    # EdgeList=EdgeList[ix,:]
    ############ Grid Graph Test END
    EdgeList=numpy.array(EdgeList)
    EP=numpy.ones(EdgeList.shape[0])*.5
    REL,SetixTotal,SetixMax=ATR(EdgeList,EP)
    print(REL)
    print(REL*2**EdgeList.shape[0])
