'''
This module contains functions to perform miscellaneous tasks on cubic graphs. Mostly for generating uniform connected random cubic graphs.

Functions
---------
**SetSeed(seed) :**
    Sets seed of numba runtime

**ConnectedRandomCubicGenerate(pegs,G) :**
    numba optimized function for generating random uniform connected random cubic graphs. A 1-Flipper Markov Chain Monte Carlo (MCMC) algorithm is used to uniformly generate these random cubic graphs.
    Algorithm from:
    Tomás Feder, Adam Guetz, Milena Mihail, and Amin Saberi. A local switch markov chain on given degree graphs with application in connectivity of peer-to-peer networks. In 2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS’06), pages 69-76. IEEE, 2006
    Code written by me.

**ConnectedRandomCubic(n,EP=None) :**
    Returns a random uniform connected random cubic graph of size n by calling ConnectedRandomCubicGenerate

**GenConnectedRandomCubic(ID0=0,ID1=10000,Vmin=20,Vmax=400,seed=42) :**
    Helper function to pre-generate a list of random connected cubic graphs.


Notes
-----
When the graph generation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.
'''

import os
import sys
# ensures local modules are imported properly
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,cwd)

import numpy
from numba import jit

import general_utilities
import ContractionOrdering
from tree_width import TreeWidth

@jit(nopython=True,cache=True)
def SetSeed(seed):
    '''
    This function sets the seed of the numba runtime.

    Parameters
    ----------
    seed : int
        seeds the numba RNG

    Examples
    --------
    >>> seed=42
    >>> SetSeed(seed)
    '''
    numpy.random.seed(seed)
    return

@jit(nopython=True,cache=True)
def ConnectedRandomCubicGenerate(pegs):
    '''
    This function generates random uniform connected cubic graphs. A 1-Flipper Markov Chain Monte Carlo (MCMC) algorithm is used to uniformly generate these random cubic graphs.
    Algorithm from:
    Tomás Feder, Adam Guetz, Milena Mihail, and Amin Saberi. A local switch markov chain on given degree graphs with application in connectivity of peer-to-peer networks. In 2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS’06), pages 69-76. IEEE, 2006
    Code written by me.

    Parameters
    ----------
    pegs : 1D numpy array of size n, int
        Array containing the sequence of node degrees. Currently is overridden to only generate cubic graphs.

    Returns
    -------
    G : nx(V+1) numpy array, int
        The randomly generated graph. G[n,0] is the (degree-1) of node n. G[n,1:degree+1] contains the labels of the nodes connected to node n by an edge

    Notes
    -----
    When the reliability calculation is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Currently just for cubic graphs. Graphs with arbitrary degree sequences can be implemented in the future.

    Examples
    --------
    >>> n=20
    >>> pegs=numpy.ones(n,dtype=numpy.int8)*3
    >>> G=ConnectedRandomCubicGenerate(pegs)
    '''

    # sometimes the last two pegs are on the same node, so if this occurs, just try to make another graph
    Error=True
    while Error==True:

        # set the number of pegs on each node to 3
        N=pegs.shape[0]
        pegs[:]=3

        # First build a random spanning tree
        # this ensures connectivity
        # Each node is limited to 3 edges
        G=numpy.ones((N,3+1),dtype=numpy.int64)*-1
        G[:]=-1
        G[:,0]=1

        Etree=numpy.ones(N)
        Ftree=numpy.zeros(N)


        TT=numpy.where(Etree==1)[0]
        C1=TT[numpy.random.randint(0,TT.shape[0])]
        Etree[C1]=0
        Ftree[C1]=1

        for k in range(0,N-1):
            TT=numpy.where(Ftree==1)[0]
            C1=TT[numpy.random.randint(0,TT.shape[0])]
            TT=numpy.where(Etree==1)[0]
            C2=TT[numpy.random.randint(0,TT.shape[0])]

            Etree[C2]=0
            Ftree[C2]=1

            pegs[C1]=pegs[C1]-1
            pegs[C2]=pegs[C2]-1
            if pegs[C1]==0:
                Ftree[C1]=0
            if pegs[C2]==0:
                Ftree[C2]=0

            G[C1,G[C1,0]]=C2
            G[C1,0]=G[C1,0]+1
            G[C2,G[C2,0]]=C1
            G[C2,0]=G[C2,0]+1
        # Finish building the random spanning tree

        ### Randomly connect pegs together
        Pexist=1
        Edge=numpy.zeros((N*N*9,2),dtype=numpy.int64)
        # loop for number of needed edges (hardcoded for cubic)
        for zz in range(N-1,int(N*3/2)):
            Edge[:]=0
            Eix=0
            PT=0
            # loop over each possible head and tail node for each edge.
            # build a list of possible edges to add
            for k1 in range(0,pegs.shape[0]):
                if pegs[k1]==0:
                    continue
                for k2 in range(0,pegs.shape[0]):
                    if pegs[k2]==0:
                        continue
                    if k1==k2:
                        continue
                    bb=False
                    for z in range(1,4):
                        if k2 == G[k1,z]:
                            bb=True
                            break
                    if bb==True:
                        continue
                    for p in range(0,pegs[k1]*pegs[k2]):
                        Edge[Eix,0]=k1
                        Edge[Eix,1]=k2
                        Eix=Eix+1
            Error=False
            if Eix==0:
                print('forced loop in spanning tree, trying again')
                Error=True
                break
            # select a random edge to add
            Rix=numpy.random.randint(0,Eix)

            k1=Edge[Rix,0]
            k2=Edge[Rix,1]
            pegs[k1]=pegs[k1]-1
            pegs[k2]=pegs[k2]-1
            ix=G[k1,0]
            G[k1,ix]=k2
            G[k1,0]=G[k1,0]+1
            ix=G[k2,0]
            G[k2,ix]=k1
            G[k2,0]=G[k2,0]+1
        ### Finished Randomly connecting pegs together

    ### Edge Mixing: 1-flipper
    Switch=0
    Swap=0
    # perform n*log(N) random path selections
    # viger2005efficient empirically suggests O(n) "SWAPS" are enough
    # A "SWAP" is a non-rejected edge flip
    # The log(N) is to ensure more than enough random path selections are performed to ensure O(n) not-rejected "SWAPS" are performed.
    while Switch<int(N*numpy.log(N)):

        # find a random path of length 3 (3 edges, 4 nodes)

        # pick a random starting node, Node 1
        N1=numpy.random.randint(0,N)
        # pick a random edge to follow, and get Node 2
        E1=numpy.random.randint(0,3)
        N2=G[N1,E1+1]

        # look at edges off of Node 3
        V1=-1
        V2=numpy.array([-1,-1])
        for k1 in range(1,4):
            # reject edge if it goes back to Node 1
            if G[N2,k1]==N1:
                continue

            # reject edge if the node it goes to, Node 3, connects to Node 1
            CONN=False
            for k2 in range(1,4):
                if G[G[N2,k1],k2]==N1:
                    CONN=True
            if CONN==False:
                if V1==-1:
                    V1=G[N2,k1]
                    V2[0]=G[N2,k1]
                else:
                    V2[1]=G[N2,k1]

        # randomly choose an edge that does not go back to Node 1
        N3=V2[numpy.random.randint(0,2)]

        V1=-1
        V2=numpy.array([-1,-1])
        # if the prior selected edge was not a rejected edge.
        if N3!=-1:
            for k1 in range(1,4):
                # reject edge if it goes back to Node 2
                if G[N3,k1]==N2:
                    continue

                # reject edge if the node it goes to, Node 4, connects to Node 2
                CONN=False
                for k2 in range(1,4):
                    if G[G[N3,k1],k2]==N2:
                        CONN=True
                if CONN==False:
                    if V1==-1:
                        V1=G[N3,k1]
                        V2[0]=G[N3,k1]
                    else:
                        V2[1]=G[N3,k1]

        # randomly choose an edge that does not go back to Node 2
        N4=V2[numpy.random.randint(0,2)]

        # swap end nodes, if all chosen edges are valid.
        if N4!=-1:
            for k1 in range(1,4):
                if G[N4,k1]==N3:
                    G[N4,k1]=N2
                if G[N3,k1]==N4:
                    G[N3,k1]=N1
                if G[N2,k1]==N1:
                    G[N2,k1]=N4
                if G[N1,k1]==N2:
                    G[N1,k1]=N3
            # count the number of successful flips
            Swap=Swap+1
        # count the number of attempted flips
        Switch=Switch+1

    return G

def ConnectedRandomCubic(n,EP=None):
    '''
    This is the user called function that generates random uniform connected cubic graphs. It ingests the user desired graph size n, and creates the peg array, degree sequence array, to feed into the ConnectedRandomCubicGenerate function.

    Parameters
    ----------
    n : int
        size of the desired random cubic graph. Must be even.

    EP : float
        The probability of an edge not failing. This value is set if the user desires the output graph to be probabilistic, usually for reliability measurements.

    Returns
    -------
    ee : list of tuples
        Returns the random cubic graph as an edge list.
        If EP=None, each tuple contains two values (n1,n2). These are the labels of the nodes connected by that edge.
        if EP!=None, each tuple contains two values (n1,n2,EP). These are the labels of the nodes connected by that edge, and the probability of the edge not failing EP.

    Notes
    -----
    When the graph generation by ConnectedRandomCubicGenerate is first executed, numba performs jit compilation. This causes the first execution to take several seconds to run, but the compilation is cached and future runs with different inputs do not need to recompile.

    Examples
    --------
    >>> n=20
    >>> EdgeList=ConnectedRandomCubic(n)

    >>> n=20
    >>> EP=0.99
    >>> EdgeList=ConnectedRandomCubic(n,EP)
    '''
    # check to make sure n is even
    if (n%2)==1:
        raise RuntimeError("n is not even")

    # set up arrays that go into ConnectedRandomCubic
    pegs=numpy.ones(n,dtype=numpy.int8)*3
    # G=numpy.ones((n,4),dtype=numpy.int64)*-1
    # G[:,0]=1

    G=ConnectedRandomCubicGenerate(pegs)
    # output an edge list
    if EP==None:
        # if no edge failure rate is specified
        ee=[]
        for k in range(0,n):
            for e in range(1,4):
                Edge=[k,G[k,e]]
                Edge.sort()
                ee.append(tuple(Edge))
        ee=list(set(ee))
        ee.sort()
    else:
        # if an edge failure rate is specified
        ee=[]
        for k in range(0,n):
            for e in range(1,4):
                Edge=[k,G[k,e]]
                Edge.sort()
                ee.append((Edge[0],Edge[1],EP))
        ee=list(set(ee))
        ee.sort()

    return ee

def GenConnectedRandomCubic(ID0=0,ID1=10000,Vmin=20,Vmax=400,seed=42):
    '''
    Helper function to generate a file of pre-generated random cubic graphs.

    Parameters
    ----------
    ID0 : int
        Start ID of generated graphs

    ID1 : int
        End ID of generated graphs. ID1-ID0 graphs are generated.

    Vmin : int
        Smallest random cubic graph to generate, number of nodes.

    Vmax : int
        Largest random cubic graph to generate, number of nodes.

    seed : int
        seed for the random cubic graph generation process.


    Returns
    -------
    file : list of tuples
        Creates a file in the folder "data" named
        "ConnectedRandomCubic-'+str(ID0)+'-'+str(ID1)+'-Vmin-'+str(Vmin)+'-Vmax-'+str(Vmax)+'-seed-'+str(seed)"
        | delimated file.
        header:
        Graph ID|Graph size in number of nodes|Edge List deliminated by ;

    Notes
    -----
    Graph sizes between Vmin and Vmax inclusive are uniformly randomly selected.

    Examples
    --------
    >>> ID0=0
    >>> ID1=10000
    >>> Vmin=20
    >>> Vmax=50
    >>> seed=42
    >>> GenConnectedRandomCubic(ID0,ID1,Vmin,Vmax,seed)
    '''

    if (Vmin % 2) != 0:
        raise Exception('Vmin is not Even')

    if (Vmax % 2) != 0:
        raise Exception('Vmax is not Even')

    numpy.random.seed(seed)
    SetSeed(seed) # sets seed for the numba runtime
    filename='data/ConnectedRandomCubic-'+str(ID0)+'-'+str(ID1)+'-Vmin-'+str(Vmin)+'-Vmax-'+str(Vmax)+'-seed-'+str(seed)
    file=open(filename,'w')
    for k in range(0,ID1):
        N=int(numpy.random.randint(Vmin//2,Vmax//2+1)*2)
        EdgeList=ConnectedRandomCubic(N,EP=.5)
        if k<ID0:
            continue
        file.write(str(k)+'|')
        file.write(str(N)+'|')
        for ee in EdgeList:
            file.write(str(ee)+';')
        file.write('\n')
    file.close()


def CubicTreeWidthParallel(n):
    '''
    function called by multiprocessing to calculate the Treewidth, Pathwidth, and LineGraphwidth of a graph.

    Parameters
    ----------
    n : tuple
        n[0] contains the graph defined by an edgelist, list of tuples
        n[1] contains the ID of the graph from the random generation process, int
        n[2] contains the size of the graph in terms of the number of nodes, int

    Returns
    -------
    (n[1],n[2],TreeWidthSize,PathWidthSize,LineGraphWidth) : tuple
        (Graph ID, Graph node count,Treewidth,Pathwidth,LineGraphwidth)

    Notes
    -----
    Graph sizes between Vmin and Vmax inclusive are uniformly randomly selected.

    Examples
    --------
    >>> EdgeList=[  (0,1,.5),
                    (0,2,.5),
                    (0,3,.5),
                    (1,2,.5),
                    (1,3,.5),
                    (2,3,.5)]
    >>> ID=0
    >>> NodeCount=4
    >>> n=(EdgeList,ID,NodeCount)
    >>> CubicTreeWidthParallel(n)
    '''

    # N=n[0]
    print('k',n[1])
    EdgeList=n[0]

    # only calculate the treewidth for 6 seconds
    TreeWidthCalcTime=6

    # sometimes the tree decomposition program does not play nice with the
    # multiprocessing module (wsl.exe conflicts?)
    # so if an error occurs, try again
    while True:
        try:
            Fname=TreeWidth.TreeWidthApprox(EdgeList,seconds=TreeWidthCalcTime,ID=n[1])
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

            order,size=ContractionOrdering.ContractionOrder(bags,tree)

            NodeOrder,PathWidthSize=ContractionOrdering.GetNodeOrdering(order,bags,tree,verbose=False)

            Gline=general_utilities._EdgeListtoLine(EdgeList)
            Gix,ixTdict,dictTix=general_utilities._DictToix(Gline)
            EdgeListLine=general_utilities._ixDictToEdgeList(Gix)
            Fname=TreeWidth.TreeWidthApprox(EdgeListLine,seconds=TreeWidthCalcTime,ID=n[1])
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
            return (n[1],n[2],TreeWidthSize,PathWidthSize,LineGraphWidth)
        except:
            pass
