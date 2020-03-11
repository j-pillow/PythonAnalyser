# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ===== /Import Modules/ =====

# ===== Define Functions =====

#-------------------------------------------------------------------------------

def plotBinEdges( binEdges ):
    """Plots the bin edges"""
    for i in binEdges:
        plt.axvline(x=i, c='r', linewidth=.5, linestyle='--')

#-------------------------------------------------------------------------------

def plotStds( binEdges, binMeans, binStds, plotRange, xPlotMin, xy ):
    """Plots the bin stds. x = 0, y = 2"""

    for i in range(len(binEdges)-1):
        xmin = ( binEdges[i]   - xPlotMin ) / plotRange
        xmax = ( binEdges[i+1] - xPlotMin ) / plotRange

        ymid = binMeans[i][xy]

        ymin = ymid - binStds[i][xy]
        ymax = ymid + binStds[i][xy]
        plt.axhspan(    ymin=ymin, 
                        ymax=ymax, 
                        xmin=xmin, 
                        xmax=xmax, 
                        fc=(0,0.6,0.8,0.5), zorder=6 )

        ymin = ymid - binStds[i][xy] * 2
        ymax = ymid + binStds[i][xy] * 2
        plt.axhspan(    ymin=ymin, 
                        ymax=ymax, 
                        xmin=xmin, 
                        xmax=xmax, 
                        fc=(0,0.6,0.8,0.4), 
                        zorder=5 )

        ymin = ymid - binStds[i][xy] * 3
        ymax = ymid + binStds[i][xy] * 3
        plt.axhspan(    ymin=ymin, 
                        ymax=ymax, 
                        xmin=xmin, 
                        xmax=xmax, 
                        fc=(0,0.6,0.8,0.3), 
                        zorder=4 )

#-------------------------------------------------------------------------------

def plotXYStds( binEdges, xyBinStds, plotRange, xPlotMin, ):
    """Plots the stds for the xy multiplied"""

    for i in range(len(binEdges)-1):
        xmin = ( binEdges[i]   - xPlotMin ) / plotRange
        xmax = ( binEdges[i+1] - xPlotMin ) / plotRange

        ymin = -1*xyBinStds[i]
        ymax = xyBinStds[i]
        plt.axhspan(    ymin=ymin, 
                        ymax=ymax, 
                        xmin=xmin, 
                        xmax=xmax, 
                        fc=(0,0.6,0.8,0.5), 
                        zorder=6 )

        ymin = -1*xyBinStds[i] * 2
        ymax = xyBinStds[i] * 2
        plt.axhspan(    ymin=ymin, 
                        ymax=ymax, 
                        xmin=xmin, 
                        xmax=xmax, 
                        fc=(0,0.6,0.8,0.5), 
                        zorder=5 )

        ymin = -1*xyBinStds[i] * 3
        ymax = xyBinStds[i] * 3
        plt.axhspan(    ymin=ymin, 
                        ymax=ymax, 
                        xmin=xmin, 
                        xmax=xmax, 
                        fc=(0,0.6,0.8,0.5), 
                        zorder=4 )

#-------------------------------------------------------------------------------

def doPlotsMC( xzyeT, binEdges, binMeans, binStds, maxZ, startPos, mcXZYF, mcXZYL,
                binWidth ):
    """Function that makes the 2D plots with the stds"""
    # Unpack the x, z, and y coords
    x = [ i[0] for i in xzyeT ]
    z = [ i[1] for i in xzyeT ]
    y = [ i[2] for i in xzyeT ]

    fig = plt.figure(figsize=(25,13))

    plotRows    = 3
    plotColumns = 1
    plotIndex   = 0

    xPlotMin = -binWidth
    xPlotMax = maxZ + binWidth
    plotRange = xPlotMax - xPlotMin

    #---------
    plotIndex += 1
    plt.subplot( plotRows, plotColumns, plotIndex )

    plt.scatter( z, y, s=2, zorder=10 )
    plt.axhline( y=0, c='r', linewidth=.5, linestyle='--' )

    plotBinEdges( binEdges )
    plotStds( binEdges, binMeans, binStds, plotRange, xPlotMin, 2 )

    plt.axvline( x=startPos,  c='b', linewidth=2, linestyle='--' )
    plt.axvline( x=mcXZYF[1], c='g', linewidth=2, linestyle='--' )
    plt.axvline( x=mcXZYL[1], c='m', linewidth=2, linestyle='--' )

    plt.title( "{} | bin width: {}cm".format(evt,binWidth*X0) )
    plt.ylabel( r'y ($R_m$)' )
    plt.xlabel( r'z ($X_0$)' )
    plt.xlim( xPlotMin, xPlotMax )

    #---------
    plotIndex += 1
    plt.subplot( plotRows, plotColumns, plotIndex )

    plt.scatter( z, x, s=2, zorder=10 )
    plt.axhline( y=0, c='r', linewidth=.5, linestyle='--' )

    plotBinEdges( binEdges )
    plotStds( binEdges, binMeans, binStds, plotRange, xPlotMin, 0 )

    plt.axvline( x=startPos,  c='b', linewidth=2, linestyle='--' )
    plt.axvline( x=mcXZYF[1], c='g', linewidth=2, linestyle='--' )
    plt.axvline( x=mcXZYL[1], c='m', linewidth=2, linestyle='--' )

    plt.ylabel( r'x ($R_m$)' )
    plt.xlabel( r'z ($X_0$)' )
    plt.xlim( xPlotMin, xPlotMax )

    #---------
    plotIndex += 1
    plt.subplot( plotRows, plotColumns, plotIndex )

    plt.axhline( y=0, c='r', linewidth=.5, linestyle='--' )

    plotBinEdges( binEdges )

    plotXYStds( binEdges, xyBinStds, plotRange, xPlotMin )

    plt.axvline( x=startPos,  c='b', linewidth=2, linestyle='--' )
    plt.axvline( x=mcXZYF[1], c='g', linewidth=2, linestyle='--' )
    plt.axvline( x=mcXZYL[1], c='m', linewidth=2, linestyle='--' )

    plt.ylabel( r'xy ($Rm^2$)' )
    plt.xlabel( r'z ($X_0$)' )
    plt.xlim( xPlotMin, xPlotMax )

    plt.show()

#-------------------------------------------------------------------------------

def do3DPlot( xzyeT, startPos, mcXZYF, mcXZYL, pMc ):
    """Makes the 3D plots with the reco and mc info"""

    x = [ i[0] for i in xzyeT ]
    z = [ i[1] for i in xzyeT ]
    y = [ i[2] for i in xzyeT ]

    mx = [ i[0] for i in pMc ]
    mz = [ i[1] for i in pMc ]
    my = [ i[2] for i in pMc ]

    fig = plt.figure(figsize=(15,13))
    ax  = plt.axes( projection='3d' )

    ax.scatter( x, z, y, s=2 )
    ax.scatter( mx, mz, my, s=2 )

    ax.plot( [-1,1], [startPos,startPos], [0,0],  c='b', lw=.5, ls='--' )
    ax.plot( [0,0],  [startPos,startPos], [-1,1], c='b', lw=.5, ls='--' )

    ax.plot( [-1,1], [mcXZYF[1],mcXZYF[1]], [0,0],  c='g', lw=.5, ls='--' )
    ax.plot( [0,0],  [mcXZYF[1],mcXZYF[1]], [-1,1], c='g', lw=.5, ls='--' )

    ax.plot( [-1,1], [mcXZYL[1],mcXZYL[1]], [0,0],  c='m', lw=.5, ls='--' )
    ax.plot( [0,0],  [mcXZYL[1],mcXZYL[1]], [-1,1], c='m', lw=.5, ls='--' )

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.tight_layout()
    ax.set_xlim(ax.get_xlim()[::-1])
    #ax.set_ylim3d(0,28)
    ax.set_ylim3d(0,max(z))
    plt.show()

#-------------------------------------------------------------------------------

def doPlots( evt, xzyeT, binEdges, binMeans, binStds, xyBinStds, maxZ, startPos, 
             binWidth ):
    """Function that makes the 2D plots with the stds"""

        # Set constants
    X0 = 14
    Rm = 10

    # Unpack the x, z, and y coords
    x = [ i[0] for i in xzyeT ]
    z = [ i[1] for i in xzyeT ]
    y = [ i[2] for i in xzyeT ]

#    fig = plt.figure(figsize=(25,13))
    fig = plt.figure(constrained_layout=False,figsize=(25,13))
    gs = fig.add_gridspec(  nrows=3, ncols=1, 
                            left=0.03, right=0.99, 
                            top=0.95, bottom=0.05,
                            hspace=0)

    plotRows    = 3
    plotColumns = 1
    plotIndex   = 0

    xPlotMin = -binWidth
    xPlotMax = maxZ + binWidth
    plotRange = xPlotMax - xPlotMin

    #---------
    plotIndex += 1
#    plt.subplot( plotRows, plotColumns, plotIndex )
    f_ax1 = fig.add_subplot(gs[0,0])

    plt.scatter( z, y, c='b', s=2, zorder=10 )
    plt.axhline( y=0, c='r', linewidth=.5, linestyle='--' )

    plotBinEdges( binEdges )
    plotStds( binEdges, binMeans, binStds, plotRange, xPlotMin, 2 )

    plt.axvline( x=startPos,  c='b', linewidth=2, linestyle='--' )

    plt.title( "{} | bin width: {}cm".format(evt,binWidth*X0) )
    plt.ylabel( r'Shower Secondary Axis ($R_m$)' )
    plt.xlabel( r'Shower Primary Axis ($X_0$)' )
    plt.xlim( xPlotMin, xPlotMax )

    #---------
    plotIndex += 1
#    plt.subplot( plotRows, plotColumns, plotIndex )
    f_ax1 = fig.add_subplot(gs[1,0])

    plt.scatter( z, x, c='b', s=2, zorder=10 )
    plt.axhline( y=0, c='r', linewidth=.5, linestyle='--' )

    plotBinEdges( binEdges )
    plotStds( binEdges, binMeans, binStds, plotRange, xPlotMin, 0 )

    plt.axvline( x=startPos,  c='b', linewidth=2, linestyle='--' )

    plt.ylabel( r'Shower Tertiary Axis ($R_m$)' )
    plt.xlabel( r'Shower Primary Axis ($X_0$)' )
    plt.xlim( xPlotMin, xPlotMax )

    #---------
    plotIndex += 1
#    plt.subplot( plotRows, plotColumns, plotIndex )
    f_ax1 = fig.add_subplot(gs[2,0])

    plt.axhline( y=0, c='r', linewidth=.5, linestyle='--' )

    plotBinEdges( binEdges )

    plotXYStds( binEdges, xyBinStds, plotRange, xPlotMin )

    plt.axvline( x=startPos,  c='b', linewidth=2, linestyle='--' )

    plt.ylabel( r'Secondary $\cdot$ Tertiary ($Rm^2$)' )
    plt.xlabel( r'Shower Primary Axis ($X_0$)' )
    plt.xlim( xPlotMin, xPlotMax )
    
#    gs.update(hspace=0.05)
#    fig.subplots_adjust(hspace=0)

    plt.show()

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':

    print("This is a python file for storing useful plot functions for use in other scripts")

# ===== /Main Program/ =====

