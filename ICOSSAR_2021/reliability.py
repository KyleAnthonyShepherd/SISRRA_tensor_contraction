'''
This module calculates the reliability of undirected graphs.
Using tree decomposition algorithms, these functions optimize the edge factor ordering before the reliability calculations are performed.

Functions
---------
**ATR(EdgeList,ATRverbose=False,TreeWidthCalcTime=6) :**
    Calculates the All Terminal Reliability of a undirected graph defined by EdgeList.

**EdgeCover(EdgeList,ECverbose=False,TreeWidthCalcTime=6) :**
    Calculates the Edge Cover of a undirected graph defined by EdgeList.

Notes
-----
When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

The tree decomposition algorithms are written for UNIX operating environments, so windows environments must have the ability to run "wsl.exe" to properly execute these code packages.

Currently just for All Terminal Reliability. K-Terminal Reliability can be implemented in the future.

Examples
--------
>>> import reliability
>>> n=5
>>> EdgeList=[]
>>> for k1 in range(0,n-1):
>>>     for k2 in range(k1+1,n):
>>>         EdgeList.append((k1,k2))

>>> EdgeList=numpy.array(EdgeList)
>>> EP=numpy.ones(EdgeList.shape[0])*.5
>>> REL,SetixTotal,SetixMax=reliability.ATR(EdgeList,EP)
'''

import os
import sys
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,cwd)

import time

import numpy

import general_utilities
import tensor_network_contraction
import ContractionOrdering
import ATRnumbaTrie
from tree_width import TreeWidth

# input a list of edges, E
# E[x]=(n1,n2,p)
# n1=head of edge
# n2=tail of edge
# p=probability of edge existing, float between 0 and 1

# each node is labeled as an integer from 0 to n.

def ATR(EdgeList,ATRverbose=False,TreeWidthCalcTime=6,ID=''):
    '''
    This function determines the All Terminal Reliability of a given graph. It calls the functions to optimize the edge contraction order, and calls the function to calculate the All Terminal Reliability.

    Parameters
    ----------
    EdgeList : list of tuples, (int,int,float)
        EdgeList[i] is the ith edge of the graph.
        EdgeList[i][0] and EdgeList[i][1] are the node labels that the edge connects.
        EdgeList[i][2] is the probability of the edge not failing, the edge reliability.

    ATRverbose : bool
        bool controlling the amount of text output to the console

    TreeWidthCalcTime : int
        length of time in seconds to compute the tree decomposition of the graph. First this length of time is used to attempt to exactly solve for a tree decomposition. If the solver does not finish in time, an approximate solver is used for this length of time.

    ID : string
        When this function is called in a parallel application, a unique ID should be passed so the tree decomposition programs do not overwrite each others output

    Returns
    -------
    Result : dictionary
        Result['REL']=All Terminal Reliability of the graph
        Result['CalculationEffort']=Total subgraphs computed
        Result['RELtime']=Wall clock computation time
        Result['TreeWidthSize']=Treewidth
        Result['PathWidthSize']=Pathwidth

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Each run is not identical, the treewidth solvers use their own internal Pseudo-RNG and do not expose a way to set this seed. Therefore, while Result['REL'] should be the same each time, the computational effort may not be the same on each run.

    Examples
    --------
    >>> EdgeList=[  (0,1,.5),
                    (0,2,.5),
                    (0,3,.5),
                    (1,2,.5),
                    (1,3,.5),
                    (2,3,.5)]
    >>> TreeWidthCalcTime=60
    >>> ID='64'
    >>> Results=ATR(EdgeList,TreeWidthCalcTime=6,ID=ID)
    '''
    TreeWidthSize=None
    PathWidthSize=None
    # if the treewidth calculation time is not set to zero, optimize the edge ordering
    if TreeWidthCalcTime!=0:
        Fname=TreeWidth.TreeWidthExact(EdgeList,seconds=TreeWidthCalcTime,ID=ID+'ATR')
        print('Obtained Tree Decomposition')

        # GET EDGE CONTRACTION ORDER
        file=open(Fname)
        lines=file.readlines()
        file.close()
        os.remove(os.getcwd()+'/'+Fname)
        bags=[]
        tree=[]
        for l in lines:
            if l[0]=='c':
                continue
            if l[0]=='s':
                data=l.replace('\n','').split(' ')
                tree=[[] for i in range(0,int(data[2]))]
                TreeWidthSize=data[3]
                continue
            if l[0]=='b':
                data=l.replace('\n','').split(' ')
                d=[int(i)-1 for i in data[2:]]
                bags.append(d)
                continue
            data=l.replace('\n','').split(' ')

            tree[int(data[0])-1].append(int(data[1])-1)
            tree[int(data[1])-1].append(int(data[0])-1)

        # get the pathwidth optimized order
        order,size=ContractionOrdering.ContractionOrder(bags,tree)

        NodeOrder,PathWidthSize=ContractionOrdering.GetNodeOrdering(order,bags,tree,verbose=False)

        Gix=dict()
        for ix,n in enumerate(NodeOrder):
            Gix[n]=ix

        ELin=[]
        for e in EdgeList:
            Edge=[Gix[e[0]],Gix[e[1]]]
            Edge.sort()
            ELin.append((Edge[0],Edge[1],e[2]))
        ELin.sort()

        EP=[]
        for ix,e in enumerate(ELin):
            ELin[ix]=(e[0],e[1])
            EP.append(e[2])

        ELin=numpy.array(ELin)
        EP=numpy.array(EP)

        # start wall clock time
        t=time.time()
        # call main calculation function
        REL,SetixTotal,SetixMax=ATRnumbaTrie.ATR(ELin,EP,verbose=ATRverbose)

    # if treewidth calculation time set to zero,
    # assume the edgelist is already in an optimal order
    # some classes of graphs such as grids have easy to specify optimal orders
    else:
        ELin=[]
        for e in EdgeList:
            Edge=[e[0],e[1]]
            Edge.sort()
            ELin.append((Edge[0],Edge[1],e[2]))
        ELin.sort()

        EP=[]
        for ix,e in enumerate(ELin):
            ELin[ix]=(e[0],e[1])
            EP.append(e[2])

        ELin=numpy.array(ELin,dtype=numpy.int64)
        EP=numpy.array(EP)

        # start wall clock time
        t=time.time()
        # call main calculation function
        REL,SetixTotal,SetixMax=ATRnumbaTrie.ATR(ELin,EP,verbose=ATRverbose)

    RELtime=time.time()-t
    Result=dict()
    Result['REL']=REL
    Result['CalculationEffort']=SetixTotal
    Result['RELtime']=RELtime
    Result['TreeWidthSize']=TreeWidthSize
    Result['PathWidthSize']=PathWidthSize
    return Result

def EdgeCover(EdgeList,ECverbose=False,TreeWidthCalcTime=6,ID=''):
    '''
    This function determines the Edge Cover of a given graph, the probability the remaining edges cover all nodes in the graph. It calls the functions to optimize the edge contraction order, and calls the tensor network contraction function to calculate the Edge Cover.

    Parameters
    ----------
    EdgeList : list of tuples, (int,int,float)
        EdgeList[i] is the ith edge of the graph.
        EdgeList[i][0] and EdgeList[i][1] are the node labels that the edge connects.
        EdgeList[i][2] is the probability of the edge not failing, the edge reliability.

    ECverbose : bool
        bool controlling the amount of text output to the console

    TreeWidthCalcTime : int
        length of time in seconds to compute the tree decomposition of the graph. First this length of time is used to attempt to exactly solve for a tree decomposition. If the solver does not finish in time, an approximate solver is used for this length of time.

    ID : string
        When this function is called in a parallel application, a unique ID should be passed so the tree decomposition programs do not overwrite each others output

    Returns
    -------
    Result : dictionary
        Result['EC']=Edge Cover probability of the graph
        Result['CalculationEffort']=Total subgraphs computed
        Result['ECtime']=Wall clock computation time
        Result['LineGraphWidth']=Line Graph Treewidth

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Each run is not identical, the treewidth solvers use their own internal Pseudo-RNG and do not expose a way to set this seed. Therefore, while Result['REL'] should be the same each time, the computational effort may not be the same on each run.

    Examples
    --------
    >>> EdgeList=[  (0,1,.5),
                    (0,2,.5),
                    (0,3,.5),
                    (1,2,.5),
                    (1,3,.5),
                    (2,3,.5)]
    >>> TreeWidthCalcTime=60
    >>> ID='64'
    >>> Results=EdgeCover(EdgeList,TreeWidthCalcTime=6,ID=ID)
    '''
    LineGraphWidth=None
    if TreeWidthCalcTime!=0:
        Gline=general_utilities._EdgeListtoLine(EdgeList)
        Gix,ixTdict,dictTix=general_utilities._DictToix(Gline)
        EdgeListLine=general_utilities._ixDictToEdgeList(Gix)
        Fname=TreeWidth.TreeWidthExact(EdgeListLine,seconds=TreeWidthCalcTime,ID=ID+'EC')
        print('Obtained Tree Decomposition')

        #### GET EDGE CONTRACTION ORDER
        # read tree decomposition
        file=open(Fname)
        lines=file.readlines()
        file.close()
        os.remove(os.getcwd()+'/'+Fname)
        bags=[]
        tree=[]
        for l in lines:
            if l[0]=='c':
                continue
            if l[0]=='s':
                data=l.replace('\n','').split(' ')
                tree=[[] for i in range(0,int(data[2]))]
                LineGraphWidth=data[3]
                continue
            if l[0]=='b':
                data=l.replace('\n','').split(' ')
                d=[int(i)-1 for i in data[2:]]
                bags.append(d)
                continue
            data=l.replace('\n','').split(' ')

            tree[int(data[0])-1].append(int(data[1])-1)
            tree[int(data[1])-1].append(int(data[0])-1)

        # Get True Bags
        EdgeBags=[]
        for b in bags:
            BB=[]
            for n in b:
                BB.append(ixTdict[n])
            EdgeBags.append(BB)

        # contract in tree decomposition, leafs inward
        EdgeOrder=[]
        MutTree=[list(tree[i]) for i in range(0,len(tree))]
        for k in range(0,len(tree)-2):
            for i,t in enumerate(MutTree):
                if len(t)==1:
                    EC=set(EdgeBags[i]).difference(set(EdgeBags[t[0]]))
                    EdgeOrder.extend(EC)
                    MutTree[t[0]].remove(i)
                    MutTree[i].remove(t[0])
                    break
        for i,t in enumerate(MutTree):
            if len(t)==1:
                EC=set(EdgeBags[i]).union(set(EdgeBags[t[0]]))
                EdgeOrder.extend(EC)
                break

        # create Node Tensors
        G=general_utilities._EdgeListtoG(EdgeList)
        Nodes=[]
        for n in G:
            TT=numpy.ones(2**len(G[n]))
            TT[0]=0
            Edges=[]
            # for e in G[n]:
            #     EE=[n,e]
            #     EE.sort()
            #     EE=tuple(EE)
            #     Edges.append(EE)
            Nodes.append((TT,tuple(G[n])))

        # REL,SetixTotal,SetixMax=tensor_network_contraction.ATR(ELin,EP,verbose=ATRverbose)
        t=time.time()
        EC,CalculationEffort=tensor_network_contraction.Main(Nodes,EdgeOrder,verbose=ECverbose)
        # print(EC)
    else:
        # create Node Tensors
        G=general_utilities._EdgeListtoG(EdgeList)
        Nodes=[]
        for n in G:
            TT=numpy.ones(2**len(G[n]))
            TT[0]=0
            Edges=[]
            # for e in G[n]:
            #     EE=[n,e]
            #     EE.sort()
            #     EE=tuple(EE)
            #     Edges.append(EE)
            Nodes.append((TT,tuple(G[n])))

        # REL,SetixTotal,SetixMax=tensor_network_contraction.ATR(ELin,EP,verbose=ATRverbose)
        t=time.time()
        EC,CalculationEffort=tensor_network_contraction.Main(Nodes,EdgeList,verbose=ECverbose)
        # print(EC)

    ECtime=time.time()-t
    Result=dict()
    Result['EC']=EC
    Result['CalculationEffort']=CalculationEffort
    Result['ECtime']=ECtime
    Result['LineGraphWidth']=LineGraphWidth
    return Result

if __name__ == "__main__":
    # example for the graph K4,
    # each edge probability of existing=0.5
    EdgeList=[]
    EdgeList.append((0,1,.5))
    EdgeList.append((0,2,.5))
    EdgeList.append((0,3,.5))
    EdgeList.append((1,2,.5))
    EdgeList.append((1,3,.5))
    EdgeList.append((2,3,.5))

    ############ Complete Graph Test
    # n=5
    # EdgeList=[]
    # for k1 in range(0,n-1):
    #     for k2 in range(k1+1,n):
    #         EdgeList.append((k1,k2,.5))
    # ATR(EdgeList,ATRverbose=True)
    ############ Complete Graph Test END

    ############ Grid Graph Test
    n=21
    EdgeList=[]
    for k1 in range(0,n):
        for k2 in range(0,n-1):
            EdgeList.append((n*k1+k2,n*k1+k2+1,.5))

    for k1 in range(0,n-1):
        for k2 in range(0,n):
            EdgeList.append((n*k1+k2,n*(k1+1)+k2,.5))

    # EdgeList=numpy.array(EdgeList)
    # ix=numpy.argsort(EdgeList[:,0])
    # EdgeList=EdgeList[ix,:]
    EdgeList.sort()
    EC=EdgeCover(EdgeList,TreeWidthCalcTime=0)
    print(EC)
    sys.exit()
    # ATR(EdgeList,ATRverbose=True,TreeWidthCalcTime=0)
    ############ Grid Graph Test END

    ############ Random Cubic Graph Test
    n=60
    seed=1
    numpy.random.seed(seed)
    EdgeList=cubic_graph_utilities.ConnectedRandomCubic(n,EP=.5)
    print(EdgeList)
    EC=EdgeCover(EdgeList)
    sys.exit()
    Gline=general_utilities._EdgeListtoLine(EdgeList)
    G,ixTdict,dictTix=general_utilities._DictToix(Gline)
    EdgeListLine=general_utilities._ixDictToEdgeList(G)
    print(Gline)
    print(G)
    print(EdgeListLine)
    input()

    REL=ATR(EdgeList,ATRverbose=True)
    print(REL)

    EdgeList=cubic_graph_utilities.ConnectedRandomCubic(n,EP=.5)
    REL=ATR(EdgeList,ATRverbose=True)
    print(REL)
    ############ Random Cubic Graph End
