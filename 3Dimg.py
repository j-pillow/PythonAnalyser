# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm

import plotFuncs as pf
import utilFuncs as uf

# ===== /Import Modules/ =====

# ===== Define Functions =====

# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':
    
    variables = ["cnnShowerScore","hitPurity","hitCompleteness","energy","energyCorrected","mcInitEnergy","nHits","correctedHit3DX","correctedHit3DY","correctedHit3DZ","correctedEvals","correctedAvePos","correctedPvec","correctedSvec","correctedTvec","recursiveEvals","recursiveAvePos","recursivePvec","recursiveSvec","recursiveTvec",]

    #rfile = up.open( "./data/singleParticles/electrons/singleElectron_1GeV_trimmed.root" )
    #rfile = up.open( "./data/PDSPProd2/PDSPProd2_1GeV_cnn.root" )
    rfile = up.open( "r5809_test.root" )
    df    = rfile["pdAnaTree/AnaTree"].pandas.df( variables, flatten=False )

    df    = df[ df.cnnShowerScore < 2.7 ]
    #print(len(df))
    #df    = df[ df.nHits > 200 ]
    #print(len(df))

    for index, row in tqdm(df.iterrows(), total=len(df)):

        #if row.nHits < 8:
        #    continue
        #if index < 3:
        #    continue
        #if index > 3:
        #    break

        xzy    = np.array([ np.array([i,k,j]) for i,j,k in zip(row.correctedHit3DX,row.correctedHit3DY,row.correctedHit3DZ)])
        cent   = np.array([ row["correctedAvePos.fX"], row["correctedAvePos.fZ"], row["correctedAvePos.fY"] ]) 
        pAxis  = np.array([ row["correctedPvec.fX"], row["correctedPvec.fZ"], row["correctedPvec.fY"] ]) 
        sAxis  = np.array([ row["correctedSvec.fX"], row["correctedSvec.fZ"], row["correctedSvec.fY"] ]) 
        tAxis  = np.array([ row["correctedTvec.fX"], row["correctedTvec.fZ"], row["correctedTvec.fY"] ]) 

        pStart = cent - (3*np.sqrt(row["correctedEvals.fX"]))*pAxis
        pEnd   = cent + (3*np.sqrt(row["correctedEvals.fX"]))*pAxis
        sStart = cent - (3*np.sqrt(row["correctedEvals.fY"]))*sAxis
        sEnd   = cent + (3*np.sqrt(row["correctedEvals.fY"]))*sAxis
        tStart = cent - (3*np.sqrt(row["correctedEvals.fZ"]))*tAxis
        tEnd   = cent + (3*np.sqrt(row["correctedEvals.fZ"]))*tAxis

#        rxzy    = np.array([ np.array([i,k,j]) for i,j,k in zip(row.recursiveHit3DX,row.recursiveHit3DY,row.recursiveHit3DZ)])
        rcent  = np.array([ row["recursiveAvePos.fX"], row["recursiveAvePos.fZ"], row["recursiveAvePos.fY"] ])
        rpAxis = np.array([ row["recursivePvec.fX"],   row["recursivePvec.fZ"],   row["recursivePvec.fY"]   ])
        rsAxis = np.array([ row["recursiveSvec.fX"],   row["recursiveSvec.fZ"],   row["recursiveSvec.fY"]   ])
        rtAxis = np.array([ row["recursiveTvec.fX"],   row["recursiveTvec.fZ"],   row["recursiveTvec.fY"]   ])

        rpStart = rcent - (150*np.sqrt(row["recursiveEvals.fX"]))*rpAxis
        rpEnd   = rcent + (150*np.sqrt(row["recursiveEvals.fX"]))*rpAxis
        rsStart = rcent - (150*np.sqrt(row["recursiveEvals.fY"]))*rsAxis
        rsEnd   = rcent + (150*np.sqrt(row["recursiveEvals.fY"]))*rsAxis
        rtStart = rcent - (150*np.sqrt(row["recursiveEvals.fZ"]))*rtAxis
        rtEnd   = rcent + (150*np.sqrt(row["recursiveEvals.fZ"]))*rtAxis



        fig = plt.figure(constrained_layout=True,figsize=(15,13))
        ax  = plt.axes( projection='3d' )

        ax.scatter( xzy[:,0], xzy[:,1], xzy[:,2], s=4)

        #ax.plot( [cent[0],pEnd[0]],[cent[1],pEnd[1]],[cent[2],pEnd[2]], c='r', ls='--' )
        #ax.plot( [cent[0],sStart[0]],[cent[1],sStart[1]],[cent[2],sStart[2]], c='g', ls='--' )
        #ax.plot( [cent[0],tStart[0]],[cent[1],tStart[1]],[cent[2],tStart[2]], c='b', ls='--' )

        #ax.plot( [pStart[0],pEnd[0]],[pStart[1],pEnd[1]],[pStart[2],pEnd[2]], c='r', ls='--' )
        #ax.plot( [sStart[0],sEnd[0]],[sStart[1],sEnd[1]],[sStart[2],sEnd[2]], c='g', ls='--' )
        #ax.plot( [tStart[0],tEnd[0]],[tStart[1],tEnd[1]],[tStart[2],tEnd[2]], c='b', ls='--' )
        ##
        #ax.plot( [rcent[0],rpEnd[0]],[rcent[1],rpEnd[1]],[rcent[2],rpEnd[2]], c='r', lw=2 )
        ##ax.plot( [rcent[0],rsEnd[0]],[rcent[1],rsEnd[1]],[rcent[2],rsEnd[2]], c='g', lw=2 )
        ##ax.plot( [rcent[0],rtEnd[0]],[rcent[1],rtEnd[1]],[rcent[2],rtEnd[2]], c='b', lw=2 )

        ##ax.plot( [rpStart[0],rpEnd[0]],[rpStart[1],rpEnd[1]],[rpStart[2],rpEnd[2]], c='r', lw=2 )
        #ax.plot( [rsStart[0],rsEnd[0]],[rsStart[1],rsEnd[1]],[rsStart[2],rsEnd[2]], c='g', lw=2 )
        #ax.plot( [rtStart[0],rtEnd[0]],[rtStart[1],rtEnd[1]],[rtStart[2],rtEnd[2]], c='b', lw=2 )
        
        ax.set_xlabel('x (cm)', fontweight="bold", fontsize=20)
        ax.set_ylabel('z (cm)', fontweight="bold", fontsize=20)
        ax.set_zlabel('y (cm)', fontweight="bold", fontsize=20)
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.tick_params(axis="x",labelsize=10)
        plt.tick_params(axis="y",labelsize=10)
        plt.tick_params(axis="z",labelsize=10)
        plt.title("MC:{} | Reco:{} | CNN:{} | nHits:{} ".format(round(row.mcInitEnergy,3),round(row.energy,3),round(row.cnnShowerScore,5),row.nHits),fontweight="bold",fontsize=20,loc="left")
        #plt.title("MC:{} | Reco:{} | Corrected:{} | Purity:{} | Comp:{} | CNN:{}".format(round(row.mcInitEnergy,3),round(row.energy,3),round(row.energyCorrected,3),round(row.hitPurity,3),round(row.hitCompleteness,3),round(row.cnnShowerScore,5)),fontweight="bold",fontsize=20,loc="left")
        plt.show()
            
            # Transform the space points onto the PCA axes
        xzyT = []
        for point in xzy:
            x = uf.transToPCAJit( point, cent, rtAxis )
            z = uf.transToPCAJit( point, cent, rpAxis )
            y = uf.transToPCAJit( point, cent, rsAxis )
            r = np.sqrt( (x*x) + (y*y) )
            xzyT.append(np.array([x,z,y]))
        shift = abs(min([ i[1] for i in xzyT]))
        xzyT   = np.asarray([ np.array([ i[0], i[1]+shift, i[2] ]) for i in xzyT ])
        
        fig = plt.figure(constrained_layout=False,figsize=(11.69,8.27))
        gs = fig.add_gridspec(  nrows=2, ncols=1, 
                                left=0.05, right=0.99, 
                                top=0.95, bottom=0.05,
                                hspace=0)
        f_ax1 = fig.add_subplot(gs[0,0])
        plt.scatter( xzyT[:,1], xzyT[:,2], s=4 )
#        plt.scatter( xzyOLDT[:,1], xzyOLDT[:,2], s=4 )
        plt.plot( [pStart[1], pEnd[1]], [-30,-30], c='maroon', ls=(0, (5, 10)) )
        plt.scatter( pEnd[1],   -30, marker="|", s=300,c='maroon' )
        plt.scatter( pStart[1], -30, marker="|", s=300,c='maroon' )
        plt.plot( [0, max(xzyT[:,1])], [-25,-25], c='green', ls=(0, (3, 5, 1, 5)) )
        plt.scatter( max(xzyT[:,1]), -25, marker="|", s=300,c='green' )
        plt.scatter( 0,              -25, marker="|", s=300,c='green' )
        plt.axhline(0,lw=1,c='r',ls='--')
#        plt.text(90,-23,"Projection Length",fontweight="bold",fontsize=12,c='green')
#        plt.text(115,-35,"Eigenvalue Length",fontweight="bold",fontsize=12,c='maroon')
        plt.ylabel("Secondary Axis (cm)",fontweight="bold")

        f_ax1 = fig.add_subplot(gs[1,0])
        plt.scatter( xzyT[:,1], xzyT[:,0], s=4 )
#        plt.scatter( xzyOLDT[:,1], xzyOLDT[:,0], s=4 )
        plt.plot( [pStart[1], pEnd[1]], [-5,-5], c='maroon', ls=(0, (5, 10)) )
        plt.scatter( pEnd[1],   -5, marker="|", s=300,c='maroon' )
        plt.scatter( pStart[1], -5, marker="|", s=300,c='maroon' )
        plt.plot( [0, max(xzyT[:,1])], [-4,-4], c='green', ls=(0, (3, 5, 1, 5)) )
        plt.scatter( max(xzyT[:,1]), -4, marker="|", s=300,c='green' )
        plt.scatter( 0,              -4, marker="|", s=300,c='green' )
        plt.axhline(0,lw=1,c='r',ls='--')
        plt.ylabel("Tertiary Axis (cm)",fontweight="bold")
        plt.xlabel("Primary Axis (cm)",fontweight="bold")
#        plt.text(90,-3.7,"Projection Length",fontweight="bold",fontsize=12,c='green')
#        plt.text(115,-5.7,"Eigenvalue Length",fontweight="bold",fontsize=12,c='maroon')
        plt.show()

# ===== /Main Program/ =====

