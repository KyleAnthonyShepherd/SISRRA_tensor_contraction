'''
This module determines the best edge contraction order to use if a tree decomposition is given. The returned order optimizes for pathwidth

Functions
----------
**ContractionOrder(bags,tree) :**  
    Main function to determine the optimal contraction ordering from a tree decomposition.

**GetOrder(r,bags,tree,MasterTree,SolvedOrder) :**  
    Recursive function for determining the optimal contraction ordering.

**GetOrderSize(order,bags,tree,verbose=False) :**  
    Function for measuring the computational complexity of a given ordering.

**GetNodeOrdering(order,bags,tree,verbose=False) :**  
    Function to convert a bag ordering into a node ordering.

Notes
-----
Bag order is optimized using a greed local optimization recursive algorithm.

Node contraction order within a bag is not optimized.

Examples
--------
>>> import ContractionOrdering
>>> bags = [[0, 1],
            [1, 2, 3],
            [2, 4, 5],
            [2, 3, 5],
            [3, 5, 6]]
>>> tree = [[1],
            [0, 3],
            [3],
            [1, 2, 4],
            [3]]
>>> order,size=ContractionOrdering.ContractionOrder(bags,tree)
>>> NodeOrder,PathWidthSize=ContractionOrdering.GetNodeOrdering(order,bags,tree,verbose=False)
'''

import os
import sys

import itertools

import numpy

def GetEdgeOrdering(order,bags,tree,G):

    edges=[]

    EdgeSet=set()
    removed=set()
    NodeRemove=set()
    MaxVal=0
    # for o in reversed(order):
    # print(order)
    for o in order:
        removed.add(o)
        CandidateSet = set(bags[o])
        TotalSet=set()
        # print(tree[o])
        for e in tree[o]:
            if e not in removed:
                TotalSet.update(bags[e])
        noderemove=CandidateSet-TotalSet
        for n in noderemove:
            NodeRemove.add(n)
            for e in G[n]:
                if e not in NodeRemove:
                    edges.append((n,e))
        # print(noderemove)
        EdgeSet.update(bags[o])
        EdgeSet=EdgeSet.union(CandidateSet)-noderemove
        # print(EdgeSet)
        # print(len(EdgeSet))
        MaxVal=max([MaxVal,len(EdgeSet)])
        # print(MaxVal)

    # print('MaxVal='+str(MaxVal))
    # for e in edges:
    #     print(e)
    # input()
    return edges

def GetNodeOrdering(order,bags,tree,verbose=False):
    '''
    Helper function to convert an optimized bag order into an optimized node contraction ordering.

    Parameters
    ----------
    order : list, int
        The optimized bag contraction order.

    bags : list of lists, int
        Tree decomposition algorithms output the tree bags in the format
        b 3 2 3 11 12 28 31 43 49
        where each number is a node label.
        bags[i] is the ith bag containing a list of the node labels in that bag.
        bags[i][j] is the jth node label in bag i.

    tree : list of lists, int
        tree[i] contains a list of the bags connected to bag i.

    verbose : bool
        boolean for verbose console output

    Returns
    -------
    NodeOrder : list
        A list of the optimized node contraction order

    PathWidth : int
        The pathwidth of the given order.

    Notes
    -----
    No attempt is made to optimize the node order within a given bag
    '''

    EdgeSet=set()
    removed=set()
    PathWidth=0
    NodeOrder=[]

    # for each bag
    for o in order:
        removed.add(o)
        CandidateSet = set(bags[o])
        TotalSet=set()

        for e in tree[o]:
            if e not in removed:
                TotalSet.update(bags[e])

        # determine the nodes to contract
        # if the bag being contracted contains the last instance of a node,
        # that node needs to be contracted
        noderemove=CandidateSet-TotalSet
        NodeOrder.extend(list(noderemove))
        if verbose:
            print(noderemove)
        EdgeSet.update(bags[o])
        EdgeSet=EdgeSet.union(CandidateSet)-noderemove

        # keep track of the edge set size, equal to Pathwidth
        ###
        PathWidth=max([PathWidth,len(EdgeSet)])
        ###

    return NodeOrder,PathWidth

def GetOrderSizeLineGraph(order,bags,tree,verbose=False):
    # print('tree='+str(tree))
    # print('bags='+str(bags))
    EdgeSet=set()
    EC=0
    for b in bags:
        if b==[]:
            continue
        EdgeSet.update(b)
        BM=numpy.max(numpy.array(b))
        EC=max([EC,BM])
    Ndegree=numpy.zeros(EC+1)
    Nseen=numpy.zeros(EC+1)
    EdgeSet=list(EdgeSet)
    for e in EdgeSet:
        Ndegree[e[0]]=Ndegree[e[0]]+1
        Ndegree[e[1]]=Ndegree[e[1]]+1
    # print(Ndegree)

    EdgeSet=set()
    EdgeRemoved=[]
    removed=set()
    MaxVal=0
    # for o in reversed(order):
    # print(order)
    for o in order:
        removed.add(o)
        CandidateSet = set(bags[o])
        TotalSet=set()
        for e in tree[o]:
            if e not in removed:
                TotalSet.update(bags[e])
        EdgeRemoved.extend(list(CandidateSet-TotalSet))

    for e in EdgeRemoved:
        # print(e)
        Nseen[e[0]]=Nseen[e[0]]+1
        Nseen[e[1]]=Nseen[e[1]]+1
        Ndegree[e[0]]=Ndegree[e[0]]-1
        Ndegree[e[1]]=Ndegree[e[1]]-1

        SS=Ndegree*Nseen
        # print(Ndegree)
        # print(Nseen)
        # print(SS)
        # input()
        NodeCount=numpy.where(SS>0)[0].shape[0]
        MaxVal=max([MaxVal,NodeCount])

    # print('MaxVal='+str(MaxVal))
    return MaxVal

BELLNUM=[]
BELLNUM.append(1)
BELLNUM.append(1)
BELLNUM.append(2)
BELLNUM.append(5)
BELLNUM.append(15)
BELLNUM.append(52)
BELLNUM.append(203)
BELLNUM.append(877)
BELLNUM.append(4140)
BELLNUM.append(21147)
BELLNUM.append(115975)
BELLNUM.append(678570)
BELLNUM.append(4213597)
BELLNUM.append(27644437)
BELLNUM.append(190899322)
BELLNUM.append(1382958545)
BELLNUM.append(10480142147)
BELLNUM.append(82864869804)
BELLNUM.append(682076806159)
BELLNUM.append(5832742205057)
BELLNUM.append(51724158235372)
BELLNUM.append(474869816156751)
BELLNUM.append(4506715738447323)
BELLNUM.append(44152005855084346)
BELLNUM.append(445958869294805289)
BELLNUM.append(4638590332229999353)
BELLNUM.append(49631246523618756274)

def GetOrderSize(order,bags,tree,verbose=False):
    '''
    This function calculates the computational cost of a particular bag ordering.

    The node edge set, |EdgeSet|, the number of nodes in common between bags[0:n] and bags[n+1:B] is used to determine the complexity. B is the number of bags.

    Complexity is calculated by sum( BELL(|EdgeSet|) ) for n from 0 to B.

    Parameters
    ----------
    order : list, int
        The bag contraction order to be measured

    bags : list of lists, int
        Tree decomposition algorithms output the tree bags in the format
        b 3 2 3 11 12 28 31 43 49
        where each number is a node label.
        bags[i] is the ith bag containing a list of the node labels in that bag.
        bags[i][j] is the jth node label in bag i.

    tree : list of lists, int
        A deep copy of the whole bag tree.

    verbose : bool
        boolean for verbose console output

    Returns
    -------
    MaxVal : int
        The total computational complexity of the contraction ordering.

    Notes
    -----
    The computational worst case for a node edge set of size n is assumed, BELL(n).
    '''

    EdgeSet=set()
    removed=set()
    MaxVal=0

    for o in order:
        removed.add(o)
        CandidateSet = set(bags[o])
        TotalSet=set()
        for e in tree[o]:
            if e not in removed:
                TotalSet.update(bags[e])
        noderemove=CandidateSet-TotalSet
        if verbose:
            print(noderemove)
        EdgeSet.update(bags[o])
        EdgeSet=EdgeSet.union(CandidateSet)-noderemove
        ###
        # MaxVal=max([MaxVal,len(EdgeSet)])
        # MaxVal=MaxVal+2**len(EdgeSet)
        if len(EdgeSet)>26:
            MaxVal=MaxVal+BELLNUM[26]*2**(len(EdgeSet)-26)
        else:
            MaxVal=MaxVal+BELLNUM[len(EdgeSet)]
        ###

    return MaxVal

def GetOrder(r,bags,tree,MasterTree,SolvedOrder):
    '''
    The main recursive function for optimizing the bag contraction order.

    For each possible branch contraction order, this function is recursively called.

    Parameters
    ----------
    r : int
        The current bag subroot

    bags : list of lists, int
        Tree decomposition algorithms output the tree bags in the format
        b 3 2 3 11 12 28 31 43 49
        where each number is a node label.
        bags[i] is the ith bag containing a list of the node labels in that bag.
        bags[i][j] is the jth node label in bag i.

    tree : list of lists, int
        The subtree passed to the recursive function.
        tree[i] contains a list of the bags connected to bag i.

    MasterTree : list of lists, int
        A deep copy of the whole bag tree.

    SolvedOrder : dict
        SolvedOrder[o] stores the optimal branch contraction order for bag o.

    Returns
    -------
    OptimumOrder : list
        The optimal contraction order for the branches and bags below bag r

    Notes
    -----
    Only the local best contraction order is chosen for bag r.
    '''
    # determine leaf and branch bags
    LeafStack=[]
    BranchStack=[]
    for e in tree[r]:
        if len(tree[e])==1:
            LeafStack.append(e)
        else:
            BranchStack.append(e)
    if LeafStack!=[]:
        BranchStack.append(LeafStack)

    # permute over all possible branch contractions
    orderings=list(itertools.permutations(BranchStack))

    # test to see which order is faster
    OrderSize=[]
    ForderList=[]
    for ordering in orderings:
        order=[]
        # create order
        for o in ordering:
            if isinstance(o, int):
                order.extend([o])
            else:
                order.extend(o)

        # recursively test order
        Forder=[]
        for o in order:

            # if the bag along the branch is not a leaf
            if len(MasterTree[o])>1:
                # if this bag has been solved already
                # used the cached answer
                if o in SolvedOrder:
                    OptimumOrder=SolvedOrder[o]
                    Forder.extend(OptimumOrder)
                else:
                    # build subtree to send to recursive function
                    subtree=[[] for k in range(0,len(tree))]
                    subtree[o]=list(tree[o])

                    if r in subtree[o]:
                        subtree[o].remove(r)
                    GrowList=set(subtree[o])
                    SeeList=set([o])
                    while len(GrowList)>0:
                        for n in list(GrowList):
                            SeeList.add(n)
                            subtree[n]=list(tree[n])
                            for e in tree[n]:
                                GrowList.add(e)

                        GrowList=GrowList-SeeList

                    # send to recursive function
                    OptimumOrder=GetOrder(o,bags,subtree,MasterTree,SolvedOrder)
                    # when recursive function returns, insert it into current
                    # Forder to be evaluated,
                    # and cache the result in SolvedOrder
                    Forder.extend(OptimumOrder)
                    SolvedOrder[o]=OptimumOrder

            # add bag into contraction order
            Forder.append(o)
        # add contraction order to list of contraction orders to test
        ForderList.append(list(Forder))

        # measure the computational complexity of the current order, and
        # save to a list
        OrderSize.append(GetOrderSize(Forder,bags,MasterTree))
    # extract the best order
    BestOrder=numpy.argmin(OrderSize)

    return ForderList[BestOrder]

def ContractionOrder(bags,tree):
    '''
    Determines the optimum order to contract the bags in the tree decomposition. Specifically finds a contraction order to minimize Pathwidth.

    For each possible tree root, a greedy algorithm is used to optimize the bag contraction order. Starting at the root, the branch to contract first is recursively determined.

    Parameters
    ----------
    bags : list of lists, int
        Tree decomposition algorithms output the tree bags in the format
        b 3 2 3 11 12 28 31 43 49
        where each number is a node label.
        bags[i] is the ith bag containing a list of the node labels in that bag.
        bags[i][j] is the jth node label in bag i.

    tree : list of lists, int
        tree[i] contains a list of the bags connected to bag i.

    Returns
    -------
    order : list
        The optimal bag contraction order.

    size : int
        BELL(n). where n is the maximum size of the node edge set as the bags are contracted.

    Notes
    -----
    As a greedy algorithm, the optimal solution, the optimal Pathwidth, is not guaranteed.

    The structure of the tree decomposition can affect the Pathwidth of the optimized order. Some grid graphs can have an optimal tree decomposition as measured by Treewidth, but the optimal Pathwidth cannot be extracted from the optimal tree decomposition.

    Only the bag contraction order is optimized. The contraction order of nodes within a bag is not optimized. For large bags, the order in which the nodes in the bag are contracted can affect the computational time of any algorithm that ingests the "optimized" order.

    Examples
    --------
    >>> bags = [[0, 1],
                [1, 2, 3],
                [2, 4, 5],
                [2, 3, 5],
                [3, 5, 6]]
    >>> tree = [[1],
                [0, 3],
                [3],
                [1, 2, 4],
                [3]]
    >>> (order,size) = ContractionOrder(bags,tree)
    '''
    ### Auto contraction order

    MasterTree=[list(i) for i in tree]

    MasterOrderSize=[]

    MasterForderList=[]

    # loop over each possible tree root
    for r,root in enumerate(MasterTree):
        if len(MasterTree[r])==1:
            continue
        SolvedOrder=dict()

        # determine leaf and branch bags
        LeafStack=[]
        BranchStack=[]
        for e in root:
            if len(MasterTree[e])==1:
                LeafStack.append(e)
            else:
                BranchStack.append(e)
        # bundle all leaf bags together
        # the order of the leaf bag contraction does not matter
        BranchStack.append(LeafStack)

        # permute over all possible branch contractions
        orderings=list(itertools.permutations(BranchStack))

        # test to see which order is faster
        OrderSize=[]
        ForderList=[]
        for ordering in orderings:
            order=[]
            # create order
            for o in ordering:
                if isinstance(o, int):
                    order.extend([o])
                else:
                    order.extend(o)

            # recursively test order
            Forder=[]
            for o in order:

                # if the bag along the branch is not a leaf
                if len(MasterTree[o])>1:
                    # if this bag has been solved already
                    # used the cached answer
                    if o in SolvedOrder:
                        OptimumOrder=SolvedOrder[o]
                        Forder.extend(OptimumOrder)
                    else:
                        # build subtree to send to recursive function
                        subtree=[[] for k in range(0,len(tree))]
                        subtree[o]=list(tree[o])

                        if r in subtree[o]:
                            subtree[o].remove(r)
                        GrowList=set(subtree[o])
                        SeeList=set([o])
                        while len(GrowList)>0:
                            for n in list(GrowList):
                                SeeList.add(n)
                                subtree[n]=list(tree[n])
                                for e in tree[n]:
                                    GrowList.add(e)

                            GrowList=GrowList-SeeList

                        # send to recursive function
                        OptimumOrder=GetOrder(o,bags,subtree,MasterTree,SolvedOrder)
                        # when recursive function returns, insert it into current
                        # Forder to be evaluated,
                        # and cache the result in SolvedOrder
                        Forder.extend(OptimumOrder)
                        SolvedOrder[o]=OptimumOrder

                # add bag into contraction order
                Forder.append(o)
            # add contraction order to list of contraction orders to test
            ForderList.append(list(Forder))
            # measure the computational complexity of the current order, and
            # save to a list
            OrderSize.append(GetOrderSize(Forder,bags,MasterTree))

        # extract the best order
        BestOrder=numpy.argmin(OrderSize)
        # return the best order
        # print(OrderSize)
        # print('Prime BestOrder='+str(BestOrder))
        # print(ForderList[BestOrder])
        # print('root='+str(r))

        order=ForderList[BestOrder]
        order.append(r)

        # add best order for root r into the list
        MasterForderList.append(list(order))
        MasterOrderSize.append(GetOrderSize(order,bags,MasterTree))
        # print(MasterOrderSize)

    # print(MasterForderList[numpy.argmin(MasterOrderSize)])
    print('Obtained Smallest Path Contraction')

    # find the best bag tree root
    order=MasterForderList[numpy.argmin(MasterOrderSize)]
    size=numpy.min(MasterOrderSize)
    return (order,size)
