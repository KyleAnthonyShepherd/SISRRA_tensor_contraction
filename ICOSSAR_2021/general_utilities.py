'''
This module contains functions to general utility tasks. The functions here are mostly responsible for converting from one graph format to another, and converting node labels from arbitrary strings to integers from 0 to |V|-1
'''

def _EdgeListtoG(EdgeList):
    # Build G
    G=dict()
    for ee in EdgeList:
        if ee[0] not in G:
            G[ee[0]]=[]
        if ee[1] not in G:
            G[ee[1]]=[]
        G[ee[0]].append(ee)
        G[ee[1]].append(ee)

    return G

def _EdgeListtoLine(EdgeList):
    # Build G
    if len(EdgeList[0])==3:
        G=dict()
        for ee in EdgeList:
            if ee[0] not in G:
                G[ee[0]]=[]
            if ee[1] not in G:
                G[ee[1]]=[]
            G[ee[0]].append((ee[1],ee[2]))
            G[ee[1]].append((ee[0],ee[2]))

        Gline=dict()
        for n in EdgeList:
            ee=[n[0],n[1]]
            ee.sort()
            n=tuple([ee[0],ee[1],n[2]])
            # n=tuple([ee[0],ee[1]])
            Gline[n]=[]
            for e in G[n[0]]:
                Edge=[n[0],e[0]]
                Edge.sort()
                Gline[n].append(tuple([Edge[0],Edge[1],e[1]]))
            for e in G[n[1]]:
                Edge=[n[1],e[0]]
                Edge.sort()
                Gline[n].append(tuple([Edge[0],Edge[1],e[1]]))
            Gline[n].remove(n)
            Gline[n].remove(n)
    else:
        G=dict()
        for ee in EdgeList:
            if ee[0] not in G:
                G[ee[0]]=[]
            if ee[1] not in G:
                G[ee[1]]=[]
            G[ee[0]].append(ee[1])
            G[ee[1]].append(ee[0])

        Gline=dict()
        for n in EdgeList:
            ee=[n[0],n[1]]
            ee.sort()
            n=tuple([ee[0],ee[1]])
            Gline[n]=[]
            for e in G[n[0]]:
                Edge=[n[0],e]
                Edge.sort()
                Gline[n].append(tuple(Edge))
            for e in G[n[1]]:
                Edge=[n[1],e]
                Edge.sort()
                Gline[n].append(tuple(Edge))
            Gline[n].remove(n)
            Gline[n].remove(n)

    return Gline

# convert dict to [0,|E|] format
def _DictToix(Gdict):
    ix=0
    ixTdict=dict()
    dictTix=dict()
    G=dict()
    for n in Gdict:
        GL=[]
        if n not in dictTix:
            dictTix[n]=ix
            ixTdict[ix]=n
            ix=ix+1
        for e in Gdict[n]:
            if e not in dictTix:
                dictTix[e]=ix
                ixTdict[ix]=e
                ix=ix+1
            GL.append(dictTix[e])
        G[dictTix[n]]=list(GL)

    return G,ixTdict,dictTix

def _ixDictToEdgeList(G):
    ee=[]
    keys=list(G.keys())
    keys.sort()
    for n in keys:
        for e in G[n]:
            ee.append((n,e))
    ee=list(set(ee))
    ee.sort()
    return ee
