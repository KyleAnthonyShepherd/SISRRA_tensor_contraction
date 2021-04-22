'''
This module contains functions to create the figures seen in the ICOSSAR 2021 paper. 
'''
import numpy

import matplotlib
matplotlib.use('Agg')  # no UI backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def MCtrials_vs_e_Figure():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 2 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])
    # plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    file=open('analysis/MCtheory','r')
    lines=file.readlines()
    file.close()

    X=[]
    Y1=[]
    Y2=[]
    Y3=[]
    for l in lines:
        data=l.replace('\n','').split('|')
        X.append((float(data[0]),float(data[1]),float(data[2]),float(data[3])))

    X=numpy.array(X)
    plot.plot(X[:,0],X[:,1],label='$\epsilon$='+"1.1")
    plot.plot(X[:,0],X[:,2],label='$\epsilon$='+"1.05")
    plot.plot(X[:,0],X[:,3],label='$\epsilon$='+"1.01")

    # plot.plot(numpy.exp(S),NN,label='$\epsilon$='+"{:.2f}".format(e-1))
    plot.set_xscale("log")
    plot.set_yscale("log")
    plot.set_title('Needed Bernoulli Monte Carlo Trials')
    plot.set_ylabel('Monte Carlo Trials')
    plot.set_xlabel('$P_{true}$')
    plot.legend()
    fig.savefig("figures/MCTrials.png",dpi=600)

def TensorGraphFigure():

    # set up plotting area
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    # dy = 6 #plot size in inches
    dy = 2.0 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0 / dx, 0 / dy, 1.75 / dx, 1.75 / dy])
    plotb = fig.add_axes([1.75 / dx, 0 / dy, 1.75 / dx, 1.75 / dy])

    ### Tensor Graph Drawing
    import matplotlib.lines as lines
    def CC(x,y,r,color):
        return plt.Circle((x, y), r, color=color)
    def LL(x1,x2,y1,y2):
        return lines.Line2D([x1,x2], [y1,y2],lw=2, color='black',zorder=-1)
    def TTc(AA,x,y,text):
        AA.text(x,y, text, horizontalalignment='center', verticalalignment='center', fontsize=8)
    def TTl(AA,x,y,text):
        AA.text(x,y, text, verticalalignment='center', fontsize=8)

    plot.axis('off')
    plotb.axis('off')

    fig.text(.5,.9,r'$(x_1 \vee x_2 \vee x_3) \wedge (x_3 \vee x_4 \vee x_5) \wedge $'+'\n'+r'$(x_2 \vee x_4 \vee x_6 \vee x_7) \wedge (x_1 \vee x_6) \wedge (x_5 \vee x_7)$',horizontalalignment='center', verticalalignment='center',fontsize=12)

    ### PLOT A
    plot.add_artist(LL(.2,.2,.2,.8))
    TTl(plot,.2,.5,'$X_1$')
    plot.add_artist(LL(.2,.5,.8,.5))
    TTl(plot,.36,.65,'$X_2$')
    plot.add_artist(LL(.2,.8,.8,.8))
    plot.text(.5,.75,'$X_3$', horizontalalignment='center', verticalalignment='center', fontsize=8)
    plot.add_artist(LL(.5,.8,.5,.8))
    TTl(plot,.675,.65,'$X_4$')
    plot.add_artist(LL(.8,.8,.2,.8))
    TTl(plot,.8,.5,'$X_5$')
    plot.add_artist(LL(.2,.5,.2,.5))
    TTl(plot,.25,.35,'$X_6$')
    plot.add_artist(LL(.8,.5,.2,.5))
    TTl(plot,.66,.35,'$X_7$')

    plot.add_artist(CC(.2,.8,.04,'#777777'))
    TTc(plot,.2,.8,'C1')
    plot.add_artist(CC(.8,.8,.04,'#777777'))
    TTc(plot,.8,.8,'C2')
    plot.add_artist(CC(.5,.5,.04,'#777777'))
    TTc(plot,.5,.5,'C3')
    plot.add_artist(CC(.2,.2,.04,'#777777'))
    TTc(plot,.2,.2,'C4')
    plot.add_artist(CC(.8,.2,.04,'#777777'))
    TTc(plot,.8,.2,'C5')

    plot.text(.5,.13,'[a]', horizontalalignment='center', fontsize=16)
    fig.add_artist(LL(.5,.5,0,.75))
    ### PLOT B
    plotb.add_artist(LL(.2,.5,.525,.525))
    TTl(plotb,.375,.555,'$X_2$')
    plotb.add_artist(LL(.2,.8,.5,.8))
    TTl(plotb,.475,.71,'$X_3$')
    plotb.add_artist(LL(.5,.8,.5,.8))
    TTl(plotb,.675,.65,'$X_4$')
    plotb.add_artist(LL(.8,.8,.2,.8))
    TTl(plotb,.8,.5,'$X_5$')
    plotb.add_artist(LL(.2,.5,.475,.475))
    TTl(plotb,.35,.425,'$X_6$')
    plotb.add_artist(LL(.8,.5,.2,.5))
    TTl(plotb,.66,.35,'$X_7$')

    plotb.add_artist(CC(.2,.5,.04,'#777777'))
    # TTc(plotb,.2,.5,'C14')
    plotb.text(.2,.565,'C14', horizontalalignment='center', verticalalignment='center', fontsize=8)
    plotb.add_artist(CC(.8,.8,.04,'#777777'))
    TTc(plotb,.8,.8,'C2')
    plotb.add_artist(CC(.5,.5,.04,'#777777'))
    TTc(plotb,.5,.5,'C3')
    # plotb.add_artist(CC(.2,.2,.08,'#777777'))
    # TTc(plotb,.2,.2,'C4')
    plotb.add_artist(CC(.8,.2,.04,'#777777'))
    TTc(plotb,.8,.2,'C5')
    # plot.add_artist(CC(.25,.5,.75,'#444444'))
    plotb.text(.5,.13,'[b]', horizontalalignment='center', fontsize=16)
    fig.savefig("figures/TensorGraphExample.png",dpi=600)

def GridGraphComputeTimeFigure():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 3.5 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 2 / dy, 2.75 / dx, 1.25 / dy])
    plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    for ix,p in enumerate([.5,.9,.99]):
        file=open('GridsNvaryMCTRIALS'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        X=[]
        REL=[]
        RELsubg=[]
        EC=[]
        ECfp=[]
        for l in lines[2:]:
            data=l.replace('\n','').split('|')

            X.append(int(data[0]))

            REL.append(float(data[2]))
            RELsubg.append(float(data[3]))
            EC.append(float(data[5]))
            ECfp.append(float(data[6]))

    plot.plot(X,RELsubg,label='$Rel_{ATR}(G)$ Subgraphs')
    plot.plot(X,ECfp,label='$EC(G)$ Floating Point Operations')
    plot2.plot(X,REL,label='$Rel_{ATR}(G)$ solve time')
    plot2.plot(X,EC,label='$EC(G)$ solve time')

    plot.set_xticklabels([])
    plot.set_xticks([])
    # plot1a.set_yticklabels([])
    # plot1a.set_yticks([])
    plot.set_yscale("log")
    plot2.set_yscale("log")

    plot.set_title('Grid Graph Computational Difficulty [a]')
    plot2.set_title('Grid Graph Wall Clock Solve Time [b]')
    plot2.set_xlabel('Grid Dimension $n$')
    plot.set_ylabel('Operation Count')
    plot2.set_ylabel('Time (seconds)')

    plot.legend()
    plot2.legend()

    # plot.text(8,1000,'Fig b', horizontalalignment='center', fontsize=16)

    fig.savefig("figures/GridGraphComputeTime.png",dpi=600)

def GridGraphComplexityFigure():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 7
    plt.rc('axes', titlesize=8)  #titlesize
    plt.rc('legend', fontsize=6)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 3.25 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 1.75 / dy, 1.0 / dx, 1.0 / dy])
    plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 1.0 / dx, 1.0 / dy])
    plot3 = fig.add_axes([2.15 / dx, 1.75 / dy, 1.0 / dx, 1.0 / dy])
    plot4 = fig.add_axes([2.15 / dx, 0.5 / dy, 1.0 / dx, 1.0 / dy])

    for ix,p in enumerate([.5,.9,.99]):
        file=open('GridsNvaryMCTRIALS'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        X=[]
        REL=[]
        RELsubg=[]
        EC=[]
        ECfp=[]
        for l in lines[2:]:
            data=l.replace('\n','').split('|')

            X.append(int(data[0]))

            REL.append(float(data[2]))
            RELsubg.append(float(data[3]))
            EC.append(float(data[5]))
            ECfp.append(float(data[6]))

    plot.plot(X,RELsubg,label='Subgraphs')
    plot.plot(X,ECfp,label='Floating Point \n Operations')
    plot2.plot(X,REL,label='$Rel_{ATR}(G)$')
    plot2.plot(X,EC,label='$EC(G)$')

    plot.set_xticklabels([])
    plot.set_xticks([])
    # plot1a.set_yticklabels([])
    # plot1a.set_yticks([])
    plot.set_yscale("log")
    plot2.set_yscale("log")

    plot.set_title('$nxn$ Grid [a]')
    plot2.set_title('Wall Clock Time [b]')
    plot2.set_xlabel('Grid Dimension $n$')
    plot.set_ylabel('Operation Count')
    plot2.set_ylabel('Time (seconds)')

    fig.suptitle('Grid Graph Computational Difficulty')

    plot.legend()
    plot2.legend()

    # plot.text(8,1000,'Fig b', horizontalalignment='center', fontsize=16)

    # fig.savefig("figures/GridGraphComputeTime.png",dpi=600)
    # sys.exit()

    ## test 2: Grid Graph 8xn MC
    # plt.rc('xtick', labelsize=7)
    # plt.rc('ytick', labelsize=7) #labelsizes
    # plt.rcParams['axes.labelsize'] = 8
    # plt.rc('axes', titlesize=10)  #titlesize
    # plt.rc('legend', fontsize=6)  #legendsize
    # dx = 3.5 #plot size in inches
    # dy = 3.5 #plot size in inches
    # fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    # plot = fig.add_axes([0.5 / dx, 2 / dy, 2.75 / dx, 1.25 / dy])
    # plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    for ix,p in enumerate([.5,.9,.99]):
        file=open('GridsLvaryMCTRIALS'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        X=[]
        REL=dict()
        RELsubg=[]
        EC=dict()
        ECfp=[]
        for l in lines[2:]:
            data=l.replace('\n','').split('|')

            X.append(int(data[0]))

            if int(data[0]) not in REL:
                REL[int(data[0])]=[]
            REL[int(data[0])].append(float(data[2]))
            if int(data[0]) not in EC:
                EC[int(data[0])]=[]
            EC[int(data[0])].append(float(data[5]))

            RELsubg.append(float(data[3]))
            ECfp.append(float(data[6]))

    RELt=[]
    ECt=[]
    for x in X:
        RELt.append(numpy.average(REL[x]))
        ECt.append(numpy.average(EC[x]))


    plot3.plot(X,numpy.array(RELsubg)/1000,label='Subgraphs')
    plot3.plot(X,numpy.array(ECfp)/1000,label='Floating Point \n Operations')
    plot4.plot(X,RELt,label='$Rel_{ATR}(G)$ (sec)')
    plot4.plot(X,numpy.array(ECt)*100,label='$EC(G)$ \n (sec*100)')

    plot3.set_xticklabels([])
    plot3.set_xticks([])
    plot3.set_yticks([int(i) for i in plot3.get_yticks()])
    plot3.set_yticklabels(plot3.get_yticks(),fontsize=6)
    # plot1a.set_yticklabels([])
    # plot1a.set_yticks([])
    # plot.set_yscale("log")
    # plot2.set_yscale("log")

    plot3.set_title('8xn Grid [c]')
    plot4.set_title('Wall Clock Time [d]')
    plot4.set_xlabel('Grid Dimension $n$')
    plot3.set_ylabel('Operation Count (thousands)')
    plot4.set_ylabel('Time')

    plot3.legend()
    plot4.legend()

    fig.savefig("figures/GridGraphComputeTime8.png",dpi=600)

def RandomCubicGraphFigure():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 3.5 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.6 / dx, .5 / dy, 2.75 / dx, 2.75 / dy])
    # plot2 = fig.add_axes([0.6 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    file=open('RcubicMCTRIALS64','r')
    lines=file.readlines()
    file.close()

    MCtrials=dict()
    for l in lines:
        data=l.replace('\n','').split('|')
        n=int(data[1])
        if n not in MCtrials:
            MCtrials[n]=[]
        MCtrials[n].append((float(data[2]),float(data[7]),float(data[7])-float(data[2]),float(data[11])))

    X=[]
    MIN=[]
    MAX=[]
    p10=[]
    p25=[]
    p50=[]
    p75=[]
    p90=[]

    AVG=[]
    STD=[]
    for k in range(50,19,-2):
        X.append(k)
        D=numpy.array(MCtrials[k])
        # print(numpy.unique(D[:,1]))
        # y=numpy.array(Y[k])

        xix=numpy.argsort(D[:,2])
        AVG.append(numpy.average(D[:,0]))
        STD.append(numpy.std(D[:,0]))

        MIN.append(numpy.min(D[:,0]))
        p10.append(numpy.percentile(D[:,0],10))
        p25.append(numpy.percentile(D[:,0],25))
        p50.append(numpy.percentile(D[:,0],50))
        p75.append(numpy.percentile(D[:,0],75))
        p90.append(numpy.percentile(D[:,0],90))
        MAX.append(numpy.max(D[:,0]))

        # AVG.append(numpy.average(D[:,1]))
        # STD.append(numpy.std(D[:,1]))

        # plot.plot(numpy.ones(D[:,0].shape[0])*k,D[:,0],'.',label='n='+str(k))
        # plot2.plot(numpy.ones(D[:,0].shape[0])*k,D[:,1],'.',label='n='+str(k))
        # plot.set_xscale("log")
        # plot.set_yscale("log")
        # plot2.set_yticks([i for i in numpy.linspace(numpy.min(D[:,1]),numpy.max(D[:,1]),6)])
        # print(numpy.average(D[:,1]))
        # plot2.set_yticklabels(plot2.get_yticks())
        # plot.set_title('MC trials needed to rule out $EC_{cubic}(G)$')
        # plot.set_xlabel('Needed MC Trials at 95% confidence')
        # plot.set_ylabel('$P(\# MC\ trials < x)$')
        # plot.legend()
        # fig.savefig("RcubicSanityCheck.png",dpi=600)
        # input()
    plot.plot(X,MIN,label="Minimum")
    plot.plot(X,p10,label="$10^{th}$ percentile")
    plot.plot(X,p25,label="$25^{th}$ percentile")
    plot.plot(X,p50,label="$50^{th}$ percentile")
    plot.plot(X,p75,label="$75^{th}$ percentile")
    plot.plot(X,p90,label="$90^{th}$ percentile")
    plot.plot(X,MAX,label="Maximum")
    plot.legend(loc='upper left')
    # plot.set_yscale("log")
    # plot2.plot(X,STD)
    print(numpy.min(MIN))
    print(numpy.max(MAX))

    plot.set_title('$(1-Rel_{ATR}(G_{rc}))$ vs |V|')
    # plot.set_xticklabels([])
    # plot.set_xticks([])
    plot.set_xlabel('|V|')
    plot.set_ylabel('$(1-Rel_{ATR}(G_{rc}))$')

    # plot2.set_title('Standard Deviation of $Rel_{ATR}(G_{rc})$ vs |V| [b]')
    # plot2.set_xlabel('|V|')
    # plot2.set_ylabel('std($Rel_{ATR}(G_{rc})$)')

    a,b=numpy.polyfit(numpy.array(X),numpy.array(AVG),1)
    print(a,b)
    a,b=numpy.polyfit(numpy.array(X),numpy.array(STD),1)
    print(a,b)
    # plot.plot(X,a*numpy.array(X)+b)
    # plot.plot(X,1-numpy.array(X)*.01**3)

    fig.savefig("figures/RcubicReliability.png",dpi=600)

def RandomCubicMonteCarlo():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 2 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    file=open('RcubicMCTRIALS64','r')
    lines=file.readlines()
    file.close()

    MCtrials=dict()
    for l in lines:
        data=l.replace('\n','').split('|')
        n=int(data[1])
        if n not in MCtrials:
            MCtrials[n]=[]
        MCtrials[n].append(float(data[11]))

    Y=dict()
    for k in range(20,52,2):
        MCtrials[k].sort()
        Y[k]=numpy.linspace(0,1,len(MCtrials[k]))
    # MCtrials=numpy.array(MCtrials)
    # Y=numpy.linspace(0,1,MCtrials.shape[0])
    for k in range(50,19,-6):
        plot.plot(MCtrials[k],Y[k],label='n='+str(k))
    plot.set_xscale("log")
    # plot.set_yscale("log")
    plot.set_title('MC trials needed to rule out $EC_{cubic}(G)$')
    plot.set_xlabel('Needed MC Trials at 95% confidence')
    plot.set_ylabel('$P(\# MC\ trials < x)$')
    plot.legend()
    fig.savefig("figures/RcubicTrials.png",dpi=600)

def RandomCubicWidthFigure():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 3.0 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 1.75 / dy, 2.75 / dx, 1.0 / dy])
    plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.0 / dy])

    file=open('TreeWidthData','r')
    lines=file.readlines()
    file.close()

    TWvPW=dict()
    TWvLGW=dict()
    X=[]
    TW=[]
    PW=[]
    LGW=[]
    for l in lines:
        data=l.replace('\n','').split('|')
        n=int(data[0])
        X.append(n)
        TW.append(int(data[1]))
        PW.append(int(data[2]))
        LGW.append(int(data[3]))

    for k in range(0,len(X)):
        if X[k] not in TWvPW:
            TWvPW[X[k]]=[]
        if X[k] not in TWvLGW:
            TWvLGW[X[k]]=[]
        TWvPW[X[k]].append(PW[X[k]]/TW[X[k]])
        TWvLGW[X[k]].append(LGW[X[k]]/TW[X[k]])

    XR=[]
    TWvPWy=[]
    TWvLGWy=[]
    keys=list(TWvPW.keys())
    keys.sort()
    for n in keys:
        TWvPWy.append(numpy.average(TWvPW[n]))
        TWvLGWy.append(numpy.average(TWvLGW[n]))
        XR.append(n)

    print(numpy.polyfit(X,TW,1))

    plot.plot(X,TW,'.',markersize=2,alpha=.05,label='Treewidth')
    plot.plot(X,PW,'.',markersize=2,alpha=.05,label='Pathwidth')
    plot.plot(X,LGW,'.',markersize=2,alpha=.05,label='$tw(LG(G))$')
    # plot.set_xscale("log")
    # plot.set_yscale("log")
    plot.set_xticklabels([])
    plot.set_xticks([])
    plot.set_title('Random Cubic Graph Width [a]')
    # plot.set_xlabel('$n$')
    plot.set_ylabel('Width')
    legn=plot.legend(markerscale=5)
    for lh in legn.legendHandles:
        lh._legmarker.set_alpha(1)

    # plot2.plot(XR,TWvPWy)
    # plot2.plot(XR,TWvLGWy)
    plot2.plot([], [])
    plot2.plot(X,numpy.array(PW)/numpy.array(TW),'.',markersize=2,alpha=.2,label=r'$\frac{Pathwidth}{Treewidth}$')
    plot2.plot(X,numpy.array(LGW)/numpy.array(TW),'.',markersize=2,alpha=.2,label=r'$\frac{tw(LG(G))}{Treewidth}$')
    plot2.set_title('Width Ratios [b]')
    plot2.set_xlabel('$|V|$')
    plot2.set_ylabel('Width Ratio')
    legn=plot2.legend(markerscale=5)
    for lh in legn.legendHandles:
        lh._legmarker.set_alpha(1)

    plot.set_xlim(20, 400)
    plot2.set_xlim(20, 400)
    fig.savefig("figures/RcubicWidth.png",dpi=600)

def GridGraphnxnMonteCarlo():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 3.5 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 2 / dy, 2.75 / dx, 1.25 / dy])
    plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    for ix,p in enumerate([.5,.9,.99]):
        file=open('GridsNvaryMCTRIALS'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        X=[]
        REL=[]
        EC=[]
        Y=[]
        for l in lines:
            data=l.replace('\n','').split('|')

            X.append(int(data[0]))

            REL.append(float(data[1]))
            EC.append(float(data[4]))
            Y.append(float(data[7][:-1]))

        plot.plot(X,REL,'-1',label='$Rel(G)$, $p='+"{:1.2}".format(1-p)+'$')
        plot.plot(X,EC,':2',label='$EC(G)$, $p='+"{:1.2}".format(1-p)+'$')
        plot2.plot(X,Y,label='$p='+"{:1.2}".format(1-p)+'$')

    plot.set_xticklabels([])
    plot.set_xticks([])
    # plot1a.set_yticklabels([])
    # plot1a.set_yticks([])
    plot2.set_yscale("log")

    plot.set_title('Grid Graph $Rel(G)$ and $EC(G)$ [a]')
    plot2.set_title('MC Trials Needed to Exclude $EC(G)$ [b]')
    plot2.set_xlabel('Grid Dimension $n$')
    plot.set_ylabel('Probability')
    plot2.set_ylabel('Monte Carlo Trials')
    plot2.set_ylim(1, 100000000)

    handles, labels = plot.get_legend_handles_labels()
    plot.legend(fontsize=6,handles=handles[::-1])
    handles, labels = plot2.get_legend_handles_labels()
    plot2.legend(fontsize=6,loc='lower right',handles=handles[::-1])

    fig.savefig("figures/GridGraphMCtrials.png",dpi=600)

def TreewidthCompare():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 3.5 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot1 = fig.add_axes([0.5 / dx, 2 / dy, 2.0 / dx, 1.25 / dy])
    plot1a = fig.add_axes([2.6 / dx, 2 / dy, 0.375 / dx, 1.25 / dy])
    plot1b = fig.add_axes([3.075 / dx, 2 / dy, 0.375 / dx, 1.25 / dy])

    plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.0 / dx, 1.25 / dy])
    plot2a = fig.add_axes([2.6 / dx, 0.5 / dy, 0.375 / dx, 1.25 / dy])
    plot2b = fig.add_axes([3.075 / dx, 0.5 / dy, 0.375 / dx, 1.25 / dy])

    file=open('TreeWidthData','r')
    lines=file.readlines()
    file.close()

    RCedges=[]
    RCTW=[]
    RCLGW=[]
    for l in lines:
        data=l.replace('\n','').split('|')
        RCedges.append(int(int(data[0])*3/2))
        RCTW.append(int(data[1]))
        RCLGW.append(int(data[3]))

    file=open('PowerGridsREL99','r')
    lines=file.readlines()
    file.close()

    PGedges=[]
    PGTW=[]
    PGLGW=[]
    for l in lines:
        data=l.replace('\n','').split('|')
        if data[4]=='NA':
            continue
        PGedges.append(int(data[2]))
        PGTW.append(int(data[5]))
        PGLGW.append(int(data[10]))

    SQedges=[]
    SQTW=[]
    SQLGW=[]
    for k in range(2,22):
        SQedges.append(k*(k-1)*2)
        SQTW.append(k)
        SQLGW.append(k+1)

    plot1.plot(SQedges,SQTW,'.',markersize=4,label='Square Grid TW')
    plot1.plot(RCedges,RCTW,'.',markersize=2,alpha=.05,label='Random Cubic TW')
    plot1.plot(PGedges,PGTW,'.',markersize=2,label='Power Grid TW')

    plot2.plot(SQedges,SQLGW,'.',markersize=4,label='Square Grid $tw(LG(G))$')
    plot2.plot(RCedges,RCLGW,'.',markersize=2,alpha=.05,label='Random Cubic $tw(LG(G))$')
    plot2.plot(PGedges,PGLGW,'.',markersize=2,label='Power Grid $tw(LG(G))$')
    plot1.set_xlim(0, 200)

    plot1a.plot(SQedges,SQTW,'.',markersize=4,label='Square Grid TW')
    plot1a.plot([], [])
    plot1a.plot(PGedges,PGTW,'.',markersize=2,label='Power Grid TW')
    plot1b.plot(SQedges,SQTW,'.',markersize=4,label='Square Grid TW')
    plot1b.plot([], [])
    plot1b.plot(PGedges,PGTW,'.',markersize=2,label='Power Grid TW')
    plot1a.set_xlim(360, 425)
    plot1b.set_xlim(755, 845)
    plot1a.set_ylim(0, 30)
    plot1b.set_ylim(0, 30)

    plot2a.plot(SQedges,SQLGW,'.',markersize=4,label='Square Grid TW')
    plot2a.plot([], [])
    plot2a.plot(PGedges,PGLGW,'.',markersize=2,label='Power Grid TW')
    plot2b.plot(SQedges,SQLGW,'.',markersize=4,label='Square Grid TW')
    plot2b.plot([], [])
    plot2b.plot(PGedges,PGLGW,'.',markersize=2,label='Power Grid TW')
    plot2a.set_xlim(360, 425)
    plot2b.set_xlim(755, 845)
    plot2a.set_ylim(0, 40)
    plot2b.set_ylim(0, 40)


    plot2.set_xlim(0, 200)
    plot1.set_ylim(0, 30)
    plot2.set_ylim(0, 40)

    plot1a.set_xticklabels([])
    plot1a.set_xticks([])
    plot1b.set_xticklabels([])
    plot1b.set_xticks([])
    plot1a.set_yticklabels([])
    plot1a.set_yticks([])
    plot1b.set_yticklabels([])
    plot1b.set_yticks([])

    plot2.set_xticks([int(i) for i in plot2.get_xticks()])
    plot2.set_xticklabels(plot2.get_xticks(),rotation='vertical')
    plot2a.set_xticks([360, 390, 425])
    plot2a.set_xticklabels(plot2a.get_xticks(),rotation='vertical')
    plot2b.set_xticks([755, 800, 845])
    plot2b.set_xticklabels(plot2b.get_xticks(),rotation='vertical')

    plot2a.set_yticklabels([])
    plot2a.set_yticks([])
    plot2b.set_yticklabels([])
    plot2b.set_yticks([])

    plot1.set_xticklabels([])
    plot1.set_xticks([])
    plot1.set_title('Treewidth Comparison [a]')
    plot2.set_title('$tw(LG(G))$ Comparison [b]')
    plot2.set_xlabel('$|E|$')
    plot1.set_ylabel('Treewidth')
    plot2.set_ylabel('$tw(LG(G))$')

    legn=plot1.legend(markerscale=2,fontsize=6)
    for lh in legn.legendHandles:
        lh._legmarker.set_alpha(1)
    legn=plot2.legend(markerscale=2,fontsize=6)
    for lh in legn.legendHandles:
        lh._legmarker.set_alpha(1)

    fig.savefig("figures/WidthCompare.png",dpi=600)

def PowerGridMonteCarlo():
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7) #labelsizes
    plt.rcParams['axes.labelsize'] = 8
    plt.rc('axes', titlesize=10)  #titlesize
    plt.rc('legend', fontsize=7)  #legendsize
    dx = 3.5 #plot size in inches
    dy = 2 #plot size in inches
    fig = plt.figure(figsize=(dx, dy)) #makes figure/canvas space
    plot = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])
    # plot2 = fig.add_axes([0.5 / dx, 0.5 / dy, 2.75 / dx, 1.25 / dy])

    width=.25
    for ix,p in enumerate([.5,.9,.99]):
        file=open('PowerGridsRELMC'+str(int(p*100)),'r')
        lines=file.readlines()
        file.close()

        X=[]
        Y=[]
        for l in lines:
            data=l.replace('\n','').split('|')
            # X.append(int(data[2]))
            X.append(int(data[0]))
            # X.append(None)
            # print(float(data[-1][:-1]))
            # Y.append(0)
            Y.append(float(data[-1][:-1]))
            # Y.append(None)
        print(numpy.where(numpy.array(Y)>1000000)[0].shape)


        # print(X)
        # print(Y)
        # plot.plot(Y,X,'.')
        plot.bar(numpy.array(X)-(ix-1)*width,Y,width,label="p="+'{:0.2f}'.format(1-p))
        plot.set_yscale("log")
        # plot.set_xscale("log")
    # plot.set_xlim(1, 58)
    plot.set_xticks(list(range(1,56,4))+[58])
    plot.set_title('Power Grid MC Trials to Rule Out $EC(G)$')
    plot.set_xlabel('Power Grid ID')
    plot.set_ylabel('Mc Trials')
    plot.legend()
    fig.savefig("figures/PowerGridMC.png",dpi=600)
