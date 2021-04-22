'''
This module contains functions to initiate the analysis for the ICOSSAR 2021 paper.
'''
import os
import sys
# ensures local modules are imported properly
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,cwd)

import numpy

import networkx

import cubic_graph_utilities
import reliability

def MCtrials_vs_e():
    # calculates the number of monte carlo trials needed to exclude e

    # Evenly sample ptrue values along the log scale
    S=numpy.linspace(numpy.log(1/1000000),numpy.log(1/2),100)
    print(S)

    NN=[]
    file=open('analysis/MCtheory','w')
    for pt in S:
        # convert to true ptrue value
        pt=numpy.exp(pt)
        print(pt)
        Ne=[pt]
        # for a few e values
        for e in [1.10,1.05,1.01]:

            P0=pt/e

            # chi squared values for 95% confidence interval
            C2=1.9205
            # determine number of needed MC trials NT
            NT=C2/(pt*numpy.log(pt)+(1-pt)*numpy.log(1-pt)-pt*numpy.log(P0)-(1-pt)*numpy.log(1-P0))

            Ne.append(NT)

        NN.append(Ne)
        file.write(str(Ne[0])+'|'+str(Ne[1])+'|'+str(Ne[2])+'|'+str(Ne[3])+'\n')

def GridRel_vs_n():
    # 3 different edge reliabilities
    for p in [.5,.9,.99]:
        file=open('analysis/GridsNvary'+str(int(p*100)),'w')
        # different grid sizes
        for k in range(2,13):
            # make grid edge list
            n1=int(k)
            n2=int(k)
            EdgeList=[]
            for k1 in range(0,n1):
                for k2 in range(0,n2-1):
                    EdgeList.append((n2*k1+k2,n2*k1+k2+1,p))

            for k1 in range(0,n1-1):
                for k2 in range(0,n2):
                    EdgeList.append((n2*k1+k2,n2*(k1+1)+k2,p))

            EdgeList.sort()

            # get grid Edge Cover values
            Result=reliability.EdgeCover(EdgeList,ECverbose=False,TreeWidthCalcTime=0)
            EC=Result['EC']
            CalculationEffortEC=Result['CalculationEffort']
            ECtime=Result['ECtime']

            # get grid All Terminal Reliability values
            Result=reliability.ATR(EdgeList,ATRverbose=False,TreeWidthCalcTime=0)
            REL=Result['REL']
            CalculationEffortREL=Result['CalculationEffort']
            RELtime=Result['RELtime']

            print(EC)
            print(REL)
            print((1-REL)/(1-EC))
            print(k)
            file.write(str(k)+'|')
            file.write(str(REL)+'|')
            file.write(str(RELtime)+'|')
            file.write(str(CalculationEffortREL)+'|')
            file.write(str(EC)+'|')
            file.write(str(ECtime)+'|')
            file.write(str(CalculationEffortEC)+'\n')
        file.close()

def GridRel_vs_n_MCtrials():
    # for each tested edge reliability
    for p in [.5,.9,.99]:
        file=open('analysis/GridsNvary'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        file=open('analysis/GridsNvaryMCTRIALS'+str(int(p*100)),'w')
        for l in lines:
            data=l.replace('\n','').split('|')

            pt=float(data[1])
            P0=float(data[4])

            C2=1.9205
            NT=C2/(pt*numpy.log(pt)+(1-pt)*numpy.log(1-pt)-pt*numpy.log(P0)-(1-pt)*numpy.log(1-P0))

            file.write(l[:-1]+'|'+str(NT)+'\n')
        file.close()

def GridRel_vs_len():
    # 3 different edge reliabilities
    for p in [.5,.9,.99]:
        file=open('analysis/GridsLvary'+str(int(p*100)),'w')
        # different grid lengths
        for k in range(8,65):
            print(k)
            n1=int(k)
            n2=8 # assume grid 8 noded wide
            EdgeList=[]
            for k1 in range(0,n1):
                for k2 in range(0,n2-1):
                    EdgeList.append((n2*k1+k2,n2*k1+k2+1,p))

            for k1 in range(0,n1-1):
                for k2 in range(0,n2):
                    EdgeList.append((n2*k1+k2,n2*(k1+1)+k2,p))

            EdgeList.sort()
            Result=reliability.EdgeCover(EdgeList,ECverbose=False,TreeWidthCalcTime=0)
            EC=Result['EC']
            CalculationEffortEC=Result['CalculationEffort']
            ECtime=Result['ECtime']

            Result=reliability.ATR(EdgeList,ATRverbose=False,TreeWidthCalcTime=0)
            REL=Result['REL']
            CalculationEffortREL=Result['CalculationEffort']
            RELtime=Result['RELtime']

            print(EC)
            print(REL)
            print((1-REL)/(1-EC))
            file.write(str(k)+'|')
            file.write(str(REL)+'|')
            file.write(str(RELtime)+'|')
            file.write(str(CalculationEffortREL)+'|')
            file.write(str(EC)+'|')
            file.write(str(ECtime)+'|')
            file.write(str(CalculationEffortEC)+'\n')
        file.close()

def GridRel_vs_len_MCtrials():
    # for each tested edge reliability
    for p in [.5,.9,.99]:
        file=open('analysis/GridsLvary'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        file=open('analysis/GridsLvaryMCTRIALS'+str(int(p*100)),'w')
        for l in lines:
            data=l.replace('\n','').split('|')

            pt=float(data[1])
            P0=float(data[4])

            C2=1.9205
            NT=C2/(pt*numpy.log(pt)+(1-pt)*numpy.log(1-pt)-pt*numpy.log(P0)-(1-pt)*numpy.log(1-P0))

            file.write(l[:-1]+'|'+str(NT)+'\n')
        file.close()

def InfRepeat(filename):
    # k=0

    # check which graph IDS have already been processed
    file=open('analysis/TreeWidthData-'+filename[filename.rfind(r"/")+1:],'r')
    IDlist=set()
    for line in file:
        l=line.split('|')
        IDlist.add(l[0])
    file.close()

    # load the graph from the text file into memory
    # yield value for parallel processing
    file=open(filename,'r')
    for line in file:
        l=line.split('|')
        if l[0] in IDlist:
            continue
        EdgeList=[]
        edges=l[2].split(';')
        for ee in edges[:-1]:
            ee=ee[1:-1].split(',')
            EdgeList.append((int(ee[0]),int(ee[1]),float(ee[2])))
        # print((EdgeList,l[0],l[1]))
        yield (EdgeList,l[0],l[1])
        # yield (int(numpy.random.randint(10,201)*2),str(k))
        # k=k+1

def CubicGraphTreeWidths(filename):
    from multiprocessing import Pool

    file=open('analysis/TreeWidthData-'+filename[filename.rfind(r"/")+1:],'a')
    print(filename)
    # filename="ConnectedRandomCubic-0-10000-Vmin-20-Vmax-400-seed-42"
    # for n in InfRepeat(filename):
    #     G=cubic_graph_utilities.CubicTreeWidthParallel(n)
    #     file.write(str(G[0])+'|'+str(G[1])+'|'+str(G[2])+'|'+str(G[3])+'|'+str(G[4])+'\n')
    #     file.flush()
    #     os.fsync(file)
    #     print(G)

    p=Pool(6)
    for G in p.imap_unordered(cubic_graph_utilities.CubicTreeWidthParallel, InfRepeat(filename)):
        file.write(str(G[0])+'|'+str(G[1])+'|'+str(G[2])+'|'+str(G[3])+'|'+str(G[4])+'\n')
        file.flush()
        os.fsync(file)
        print(G)


        # if G[1]!='Error' and G[2]!='Error':
        #     file.write(str(G[0])+'|'+str(G[1])+'|'+str(G[2])+'|'+str(G[3])+'\n')
        #     file.flush()
        #     os.fsync(file)
        #     print(G)

    p.close()
    file.close()

def RandomCubicFileRead(filename):
    # k=0

    # check which graph IDS have already been processed
    file=open('analysis/Rcubic-'+filename[filename.rfind(r"/")+1:],'r')
    IDlist=set()
    for line in file:
        l=line.split('|')
        IDlist.add(l[0])
    file.close()

    # load the graph from the text file into memory
    # yield value for parallel processing
    file=open(filename,'r')
    for line in file:
        l=line.split('|')
        if l[0] in IDlist:
            continue
        EdgeList=[]
        edges=l[2].split(';')
        for ee in edges[:-1]:
            ee=ee[1:-1].split(',')
            EdgeList.append((int(ee[0]),int(ee[1]),float(ee[2])))
        # print((EdgeList,l[0],l[1]))
        yield (EdgeList,l[0],l[1])
        # yield (int(numpy.random.randint(10,201)*2),str(k))
        # k=k+1

def RandomCubicRelProcess(n):
    # while True:
    #     try:
    EdgeList=n[0]
    k=n[1]
    n=n[2]
    Result=reliability.EdgeCover(EdgeList,ECverbose=False,TreeWidthCalcTime=6,ID=k)
    EC=Result['EC']
    CalculationEffortEC=Result['CalculationEffort']
    ECtime=Result['ECtime']
    LineGraphWidth=Result['LineGraphWidth']

    Result=reliability.ATR(EdgeList,ATRverbose=False,TreeWidthCalcTime=6,ID=k)
    REL=Result['REL']
    CalculationEffortREL=Result['CalculationEffort']
    RELtime=Result['RELtime']
    TreeWidthSize=Result['TreeWidthSize']
    PathWidthSize=Result['PathWidthSize']

    EL=[]
    for e in EdgeList:
        EL.append((e[0],e[1]))
    G=networkx.from_edgelist(EL)
    G6=networkx.to_graph6_bytes(G,header=False)
    R=dict()
    R['k']=str(k)
    R['n']=str(n)
    R['G6']=G6.decode("utf-8")[:-1]
    R['REL']=str(REL)
    R['RELtime']=str(RELtime)
    R['TreeWidthSize']=str(TreeWidthSize)
    R['PathWidthSize']=str(PathWidthSize)
    R['CalculationEffortREL']=str(CalculationEffortREL)
    R['EC']=str(EC)
    R['ECtime']=str(ECtime)
    R['LineGraphWidth']=str(LineGraphWidth)
    R['CalculationEffortEC']=str(CalculationEffortEC)

    return R
        # except:
        #     pass

def RandomCubicRel(filename):

    file=open('analysis/Rcubic-'+filename[filename.rfind(r"/")+1:],'a')

    # for n in RandomCubicFileRead(filename):
    #     R=RandomCubicRelProcess(n)
    #     file.write(R['k']+'|')
    #     file.write(R['n']+'|')
    #     file.write(R['G6']+'|')
    #     file.write(R['REL']+'|')
    #     file.write(R['RELtime']+'|')
    #     file.write(R['TreeWidthSize']+'|')
    #     file.write(R['PathWidthSize']+'|')
    #     file.write(R['CalculationEffortREL']+'|')
    #     file.write(R['EC']+'|')
    #     file.write(R['ECtime']+'|')
    #     file.write(R['LineGraphWidth']+'|')
    #     file.write(R['CalculationEffortEC']+'\n')
    #     file.flush()
    #     os.fsync(file)

    from multiprocessing import Pool
    p=Pool(6)
    for R in p.imap_unordered(RandomCubicRelProcess, RandomCubicFileRead(filename)):
        file.write(R['k']+'|')
        file.write(R['n']+'|')
        file.write(R['G6']+'|')
        file.write(R['REL']+'|')
        file.write(R['RELtime']+'|')
        file.write(R['TreeWidthSize']+'|')
        file.write(R['PathWidthSize']+'|')
        file.write(R['CalculationEffortREL']+'|')
        file.write(R['EC']+'|')
        file.write(R['ECtime']+'|')
        file.write(R['LineGraphWidth']+'|')
        file.write(R['CalculationEffortEC']+'\n')
        file.flush()
        os.fsync(file)
    file.close()


def RandomCubicRel_MCtrials(filename):
    file=open('analysis/Rcubic-'+filename[filename.rfind(r"/")+1:],'r')
    lines=file.readlines()
    file.close()

    file=open('analysis/RcubicMCTRIALS-'+filename[filename.rfind(r"/")+1:],'w')
    for l in lines:
        data=l.replace('\n','').split('|')
        # sometimes the G6 format string contains a '|'
        if len(data)==12:
            pt=float(data[3])
            P0=float(data[8])
            D=data[:2]
            D.extend(data[3:])

        elif len(data)==13:
            pt=float(data[4])
            P0=float(data[9])
            D=data[:2]
            D.extend(data[4:])

        elif len(data)>13:
            print(l)
            input()

        C2=1.9205
        NT=C2/(pt*numpy.log(pt)+(1-pt)*numpy.log(1-pt)-pt*numpy.log(P0)-(1-pt)*numpy.log(1-P0))

        file.write('|'.join(D)+'|'+str(NT)+'\n')
    file.close()

def PowerGridsREL():
    file=open('data/PowerGrids','r')
    lines=file.readlines()
    file.close()

    for p in [.5,.9,.99]:
        file=open('analysis/PowerGridsREL'+str(int(p*100)),'w')
        for II in lines: #59
            data=II.replace('\n','').split('|')

            # Trivial or unsolvable power grid instances
            if int(data[0])==27 or int(data[0])==54 or int(data[0])==57:
                continue

            G=eval(data[1])
            EdgeList=[]
            for n in G:
                for e in G[n]:
                    Edge=[n,e]
                    Edge.sort()
                    EdgeList.append((Edge[0],Edge[1],p))
            EdgeList=list(set(EdgeList))
            EdgeList.sort()

            Result=reliability.EdgeCover(EdgeList,ECverbose=False,TreeWidthCalcTime=60)
            EC=Result['EC']
            CalculationEffortEC=Result['CalculationEffort']
            ECtime=Result['ECtime']
            LineGraphWidth=Result['LineGraphWidth']

            Result=reliability.ATR(EdgeList,ATRverbose=False,TreeWidthCalcTime=60)
            REL=Result['REL']
            CalculationEffortREL=Result['CalculationEffort']
            RELtime=Result['RELtime']
            TreeWidthSize=Result['TreeWidthSize']
            PathWidthSize=Result['PathWidthSize']

            file.write(str(data[0])+'|')
            file.write(str(len(G))+'|')
            file.write(str(len(EdgeList))+'|')
            file.write(str(REL)+'|')
            file.write(str(RELtime)+'|')
            file.write(str(TreeWidthSize)+'|')
            file.write(str(PathWidthSize)+'|')
            file.write(str(CalculationEffortREL)+'|')
            file.write(str(EC)+'|')
            file.write(str(ECtime)+'|')
            file.write(str(LineGraphWidth)+'|')
            file.write(str(CalculationEffortEC)+'\n')
            file.flush()
            os.fsync(file)

        file.close()

def PowerGridsREL_vs_MCtrials():
    for p in [.5,.9,.99]:
        file=open('analysis/PowerGridsREL'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        file=open('analysis/PowerGridsRELMC'+str(int(p*100)),'w')
        for l in lines:
            data=l.replace('\n','').split('|')
            if data[4]=='NA':
                continue

            pt=float(data[3])
            P0=float(data[8])

            C2=1.9205
            NT=C2/(pt*numpy.log(pt)+(1-pt)*numpy.log(1-pt)-pt*numpy.log(P0)-(1-pt)*numpy.log(1-P0))

            file.write(l[:-1]+'|'+str(NT)+'\n')
        file.close()
