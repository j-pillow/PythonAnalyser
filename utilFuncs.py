# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, prange

# ===== /Import Modules/ =====

# ===== Define Functions =====

def getBinMids(binEdges):
    binMids = []
    for i in range(len(binEdges)-1):
        binMids.append( (binEdges[i]+binEdges[i+1])/2.0 )
    return binMids

#-------------------------------------------------------------------------------

def distPointToLine( x1, x2, x3 ):
    """Finds the distance from a point x3 to a line defined by x1 and x2"""
    return np.linalg.norm( np.cross(x3-x1,x3-x2) ) / np.linalg.norm(x2-x1)

#-------------------------------------------------------------------------------

def linePlaneIntersect( planeNorm, planePoint, pAxis, cent, epsilon=1e-10):
    """Finds the intersection of plane and a line"""

    ndotu = planeNorm.dot(pAxis)
    if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection or line is within plane")

    w = cent - planePoint
    si = -planeNorm.dot(w) / ndotu
    Psi = w + si * pAxis + planePoint
    return Psi

#-------------------------------------------------------------------------------

def rotationFinder( point, newPoint ):
    """Function to find the rotation matrix connecting two points"""
    point    = point/np.linalg.norm(point)
    newPoint = newPoint/np.linalg.norm(newPoint)

    c = np.cross( point, newPoint )
    d = np.dot( point, newPoint )

    i = np.array([[1,0,0],[0,1,0],[0,0,1]])
    v = np.array([[0,-c[2],c[1]],[c[2],0,-c[0]],[-c[1],c[0],0]])

    R = i + v + (np.dot(v,v))*(1/(1+d))
    return R

#-------------------------------------------------------------------------------

def totalRecoE( dfRe ):
    """Function to add up the energy for the primary and all its daughters"""

    showersE = dfRe[ dfRe.IsShower == 1 ].ShowerEnergyFromHits2.sum() # in GeV
    tracksE  = (dfRe[ dfRe.IsTrack == 1 ].KineticEnergy2.sum())/1000 # in MeV

    return (showersE*0.63)/0.7 + tracksE

#-------------------------------------------------------------------------------

def totalRecoEMB( dfRe ):
    """Function to add up the energy for the primary and all its daughters"""

    showersE = dfRe[ dfRe.IsShower == 1 ].ShowerEnergyFromHitsMB2.sum() # in GeV
    tracksE  = (dfRe[ dfRe.IsTrack == 1 ].KineticEnergy2.sum())/1000 # in MeV

    return showersE + tracksE

#-------------------------------------------------------------------------------

def totalRecoSPE( dfRe ):
    """Function to add up the energy for the primary and all its daughters.
    (Space points energy only)"""

    totalSPE = 0

    for index, row in dfRe.iterrows():
        try:
            totalSPE += sum(list(row.SPEVec))
        except:
            continue

    return (totalSPE*0.63)/0.7

#-------------------------------------------------------------------------------

def totalRecoSPEMB( dfRe ):
    """Function to add up the energy for the primary and all its daughters.
    (Space points energy only from my modbox)"""

    totalSPE = 0

    for index, row in dfRe.iterrows():
        try:
            totalSPE += sum(list(row.SPEVecMB))
        except:
            continue

    return totalSPE

#-------------------------------------------------------------------------------

def recursiveFlowFill( df, flowCounter, dfMc ):
    """Fills a list with the number of daughters at each stage of flow for each
    flow level 1 mc daughter"""

    flow = df.iloc[0].mcFlow
    flowCounter[flow-2] += len(df)
    for index, row in df.iterrows():
        dfNextFlow = dfMc[  (dfMc.mcFlow == flow+1) & 
                            (dfMc.mcMotherID == row.mcID) ]
        if len(dfNextFlow) > 0:
            recursiveFlowFill( dfNextFlow, flowCounter, dfMc )

#-------------------------------------------------------------------------------

def largestMCFinder( flow1df, dfMc ):
    """Finds the mc daughter particle with the most number of daughter 
    particles. Deprecated for largestMCFinderJit"""
    
    maxFlow = max(dfMc.mcFlow)
    dfDict = {'mcID':[],'level':[],'nFlow':[]}

    for index, row in flow1df.iterrows():
        #dfDict['mcID'].append(row.mcID)

        flowCounter = [0]*(maxFlow-1)

        flow2df = dfMc[ (dfMc.mcFlow == 2) & (dfMc.mcMotherID == row.mcID) ]
        if len(flow2df) > 0:
            recursiveFlowFill(flow2df, flowCounter, dfMc)

        for index, i in enumerate(flowCounter):
            number = i if i > 0 else np.nan
            dfDict['mcID'].append(row.mcID)
            dfDict['level'].append(index+2)
            dfDict['nFlow'].append(number)

    df = pd.DataFrame(dfDict)
    df = df[["mcID","nFlow"]].groupby("mcID").sum()

    return df.nFlow.idxmax()

#-------------------------------------------------------------------------------

@njit
def recursiveDaughterCounter( mcID, MCna ):

    daughterna = []
    for i in range(len(MCna)):
        if MCna[i][1] == mcID:
            daughterna.append(MCna[i])

    nDaughters = len(daughterna)
    for d in daughterna:
        dID = d[0]
        nDaughters += recursiveDaughterCounter( dID, MCna )

    return nDaughters

#-------------------------------------------------------------------------------

@njit
def largestMCFinderJit( flow1na, MCna ):
    """Finds the mc daughter particle with the most number of daughter 
    particles. Is a compiled numba function"""

    # Select the maximum flow level
    maxFlow = max(MCna[:,2])

    mcIDMAX = 0
    nDaughtersMAX = 0
    for mcID in flow1na[:,0]:
        nDaughters = recursiveDaughterCounter( mcID, MCna )
        if nDaughters > nDaughtersMAX:
            mcIDMAX = mcID
            nDaughtersMAX = nDaughters


    return mcIDMAX

#-------------------------------------------------------------------------------

@njit
def largestMCFinderInTPCJit( flow1na, MCna ):
    """Finds the mc daughter particle with the most number of daughter 
    particles. Ensures the particle is in the TPC. Is a compiled number 
    function"""

    # Select the maximum flow level
    maxFlow = max(MCna[:,2])

    mcIDMAX = 0
    nDaughtersMAX = 0
    for mcID in flow1na[:,0]:
        nDaughters = recursiveDaughterCounter( mcID, MCna )
        if nDaughters > nDaughtersMAX:
            mcIDMAX = mcID
            nDaughtersMAX = nDaughters


    return mcIDMAX

#-------------------------------------------------------------------------------

def mcStartFinder( dfMc, mcID, pAxis, tAxis, sAxis, cent, shift, Rm, X0 ):
    """Finds the true start points from the mc information"""

    #startDF = dfMc[ dfMc.mcMotherID == mcID ].iloc[0]
    startDF = dfMc[ dfMc.mcID == mcID ].iloc[0]
    mcXZYF = np.array([ startDF.mcTrajVecX[0], 
                        startDF.mcTrajVecZ[0], 
                        startDF.mcTrajVecY[0] 
                        ])
    mcXZYL = np.array([ startDF.mcTrajVecX[-1], 
                        startDF.mcTrajVecZ[-1], 
                        startDF.mcTrajVecY[-1] 
                        ])

    # Trans mc point
    mXF = transToPCAJit( mcXZYF, cent, tAxis )
    mZF = transToPCAJit( mcXZYF, cent, pAxis )
    mYF = transToPCAJit( mcXZYF, cent, sAxis )
    mcXZYF = np.array([mXF/Rm,(mZF+shift)/X0,mYF/Rm])

    # Trans mc point
    mXL = transToPCAJit( mcXZYL, cent, tAxis )
    mZL = transToPCAJit( mcXZYL, cent, pAxis )
    mYL = transToPCAJit( mcXZYL, cent, sAxis )
    mcXZYL = np.array([mXL/Rm,(mZL+shift)/X0,mYL/Rm])

    return (mcXZYF,mcXZYL)

#-------------------------------------------------------------------------------

@njit
def closest_point_finder( point, pointsArray ):
    """Finds the closest point in an array of points to a given point. Returns
    the index in the given array"""

    mindex = 0
    mindist = 99999999999999999
    for i in range(len(pointsArray)):
        dist = np.linalg.norm( point - pointsArray[i] )
        if dist < mindist:
            mindist = dist
            mindex = i
    return mindex

#-------------------------------------------------------------------------------

def transToPCA( point, cent, pAxis, sAxis, tAxis ):
    """Function to transform coordinate systems. Takes np.array()"""
    # Get vector for cent -> point
    pVec = point - cent

    # Projection along primary
    pAxis = pAxis/np.linalg.norm(pAxis)
    pProj = np.dot(pVec,pAxis)

    # Projection along secondary
    sAxis = sAxis/np.linalg.norm(sAxis)
    sProj = np.dot(pVec,sAxis)

    # Projection along tertiary
    tAxis = tAxis/np.linalg.norm(tAxis)
    tProj = np.dot(pVec,tAxis)

    # Put them together for a point
    # tProj = x axis, pProj = z axis, sProj = y axis
    newPoint = np.array([tProj,pProj,sProj])

    return newPoint

#-------------------------------------------------------------------------------

@njit
def transToPCAJit( point, cent, axis ):
    """Jit compiled function to tansform coordinate systems. Only returns one
    axis at a time"""
    # Get vector for cent -> point
    pVec = point - cent

    # Projection along primary
    axis = axis/np.linalg.norm(axis)
    proj = np.dot(pVec,axis)

    return proj

#-------------------------------------------------------------------------------

def makeBinEdges( binWidth, maxZ ):
    """Function to create bins of width binWidth up to the bin including maxZ"""
    binEdges = []
    BE = 0
    while BE < maxZ:
        binEdges.append(BE)
        BE += binWidth
        if BE > maxZ:
            binEdges.append(BE) # Upper edge of final bin

    return binEdges

#-------------------------------------------------------------------------------

def binShower( binEdges, xzyeT ):
    """Function to bin the coordinates of the shower into predefined bins. 
    Returns a 2D np.array() """
    bins = [ [] for i in range(len(binEdges)-1) ]

    for point in xzyeT:
        z = point[1]
        for i in range( len(binEdges) - 1 ):
            if binEdges[i] <= z < binEdges[i+1]:
                bins[i].append(point)
                break

    bins = np.asarray( [ np.asarray(i) for i in bins ] )
    return bins

#-------------------------------------------------------------------------------

def sumBinEnergies( binnedPoints, factor=1000 ):
    """Function to sum the energies from each bin of the binned shower. Returns
    a list of energy sums"""

    energies = []

    for tBin in binnedPoints:
        if tBin.size > 0:
            energies.append(sum(tBin[:,3]*factor))
        else:
            energies.append(0)

    return energies

#-------------------------------------------------------------------------------

def aarondEdx( xzydEdxp ):
    """Finds the dE/dx of the shower calorimitry struff from Aaron."""
    
    upsPointIndex = np.argmin( xzydEdxp[:,1] )

    xzye5cm = []
    for i in range(len(xzydEdxp)):
        if np.linalg.norm( xzydEdxp[i][:3] - xzydEdxp[upsPointIndex][:3] ) <= 5:
            xzye5cm.append( xzydEdxp[i] )

    try:
        xzye5cm = np.asarray(xzye5cm)
        dEdx = sum(xzye5cm[:,3]*xzye5cm[:,4])/5
        return dEdx
    except:
        return 0

def shortdEdx( xzye ):
    """Finds the dE/dx of the shower calorimitry struff from Aaron."""
    
    upsPointIndex = np.argmin( xzye[:,1] )

    xzye5cm = []
    for i in range(len(xzye)):
        if np.linalg.norm( xzye[i][:3] - xzye[upsPointIndex][:3] ) <= 5:
            xzye5cm.append( xzye[i] )

    try:
        xzye5cm = np.asarray(xzye5cm)
        dEdx = sum(xzye5cm[:,3])*1000/5
        return dEdx
    except:
        return 0

#-------------------------------------------------------------------------------

def mcdEdx( xzyeT ):
    """Finds the dE/dx for the part of the mc particle that lines up with the
    first 5cm of the reconstructed particle."""

    points0To5  = np.asarray([ i for i in xzyeT if ( 0 <= i[1]*14 <= 5) ])

    dE = (max(points0To5[:,3]) - min(points0To5[:,3]))*1000 if len(points0To5) > 0 else -1
    dx = (max(points0To5[:,1]) - min(points0To5[:,1]))*14   if len(points0To5) > 0 else 1

    return dE/dx

#-------------------------------------------------------------------------------

def mcdEdxNEW( xzyeT ):
    """Finds the dE/dx for the first 5cm of the mc particle inside the TPC."""

    xzyeTStrip = []
    for i in range(len(xzyeT)):
        if xzyeT[i][1] >= -0.49375:
            xzyeTStrip.append(xzyeT[i])

    
    if len(xzyeTStrip) < 2:
        return (0,1)
    
    xzyeTStrip = np.asarray(xzyeTStrip)
    upsPointIndex = np.argmin(xzyeTStrip[:,1])

    xzyeT5cm = []
    for i in range(len(xzyeTStrip)):
        if np.linalg.norm( xzyeTStrip[i][:3] - xzyeTStrip[upsPointIndex][:3] ) <= 5:
            xzyeT5cm.append( xzyeTStrip[i] )

    xzyeT5cm = np.asarray(xzyeT5cm)
    
    end   = np.argmax(xzyeT5cm[:,1])
    start = np.argmin(xzyeT5cm[:,1])

    dE = (xzyeT5cm[start][3] - xzyeT5cm[end][3])*1000
    dx = np.linalg.norm(xzyeT5cm[start][:3] - xzyeT5cm[end][:3])

    return (dE,dx)

#-------------------------------------------------------------------------------

def binStdCalc( binnedPoints ):
    """Function that finds the std in x,z,y,e for each bin. Returns a list of 
    lists"""

    stds = [ [0,0,0,0] for i in range(len(binnedPoints)) ]

    for index, tBin in enumerate(binnedPoints):
        stds[index][0] = np.std([ i[0] for i in tBin ]) if len([ i[0] for i in tBin ]) > 0 else 0.0
        stds[index][1] = np.std([ i[1] for i in tBin ]) if len([ i[1] for i in tBin ]) > 0 else 0.0
        stds[index][2] = np.std([ i[2] for i in tBin ]) if len([ i[2] for i in tBin ]) > 0 else 0.0
        stds[index][3] = np.std([ i[3] for i in tBin ]) if len([ i[3] for i in tBin ]) > 0 else 0.0

    return stds

#-------------------------------------------------------------------------------

def binMeanCalc( binnedPoints ):
    """Function that fins the mean in x,z,y,e for each bin. Returns a list of 
    lists"""

    means = [ [0,0,0,0] for i in range(len(binnedPoints)) ]

    for index, tBin in enumerate(binnedPoints):
        means[index][0] = np.mean([ i[0] for i in tBin ]) if len([ i[0] for i in tBin ]) > 0 else 0
        means[index][1] = np.mean([ i[1] for i in tBin ]) if len([ i[1] for i in tBin ]) > 0 else 0
        means[index][2] = np.mean([ i[2] for i in tBin ]) if len([ i[2] for i in tBin ]) > 0 else 0
        means[index][3] = np.mean([ i[3] for i in tBin ]) if len([ i[3] for i in tBin ]) > 0 else 0

    return means

#-------------------------------------------------------------------------------

def xyBinStdCalc( binStds ):
    """Function that calculates the xy std of each bin. Returns a list"""

    xyStds = []

    for tBin in binStds:
        xyStds.append( tBin[0]*tBin[2] )

    return xyStds

#-------------------------------------------------------------------------------

def startFinder( xyBinStds, threshold ):
    """Function that finds the index of the bin that contains the shower start,
    given a specified threshold. Returns an int"""

    for i in range(1,len(xyBinStds)):
        if (xyBinStds[i] - xyBinStds[i-1]) > threshold:
            return i

#-------------------------------------------------------------------------------

def splitShower( binnedPoints ):
    """Function that finds finds the number, and length, of the 'sections' of a
    binned shower"""

    emptyBin = [ (1 if len(i) > 0 else 0) for i in binnedPoints ]

    nBins = len(emptyBin)

    sectionLengths = []
    secLen = 0
    for i in emptyBin:
        if i == 1:
            if secLen < 0:
                sectionLengths.append(secLen)
                secLen = 1
            else:
                secLen += 1
        else:
            if secLen > 0:
                sectionLengths.append(secLen)
                secLen = -1
            else:
                secLen -= 1
    sectionLengths.append(secLen)

    return sectionLengths

#-------------------------------------------------------------------------------

def dEdxMyStart( myStart, startBinIndex, binEnergySums, binnedPoints ):
    """Function to find the dE/dx using my start as the end point"""

    dE = sum(binEnergySums[:startBinIndex])
    for index in range(len(binnedPoints[startBinIndex])):
        if binnedPoints[startBinIndex][index][1] <= myStart:
            dE += binnedPoints[startBinIndex][index][3]
       
    return dE/(myStart*14)

#-------------------------------------------------------------------------------

def dEdt( binEnergySums, binWidth ):
    """Function that computes the dEdt for a shower"""
    return binEnergySums/binWidth

#-------------------------------------------------------------------------------

def equaliseRowLength( jaggedArray ):
    """Function that appends np.nan to rows to ensure 2D structure"""

    llen = 0
    for i in jaggedArray:
        if len(i) > llen:
            llen = len(i)

    for index in range(len(jaggedArray)):
        if len(jaggedArray[index]) != llen:
            nAppends = llen - len(jaggedArray[index])
            jaggedArray[index] = np.append(jaggedArray[index],[0]*nAppends)

    return np.asarray([i for i in jaggedArray])
#-------------------------------------------------------------------------------

def radialBinShower( radEdges, xzyer ):
    """Function that bins the shower radially"""

    bins = [ [] for i in range(len(radEdges)-1) ]

    for point in xzyer:
        r = point[4]
        for i in range( len(radEdges)-1 ):
            if radEdges[-1] < r:
                bins[-1].append(point)
                break
            if radEdges[i] <= r < radEdges[i+1]:
                bins[i].append(point)
                break

    return np.asarray( [ np.asarray(i) for i in bins ] )


#-------------------------------------------------------------------------------

def radialBinEnergySumsInZ( zBinRadEs ):
    """Function that averages the energies in the radial bins in the z bins"""

    zBinRadEAves = []

    maxZBins = 0
    for shower in zBinRadEs:
        if len(shower) > maxZBins:
            maxZBins = len(shower)

    for zBin in range(maxZBins):
        zBin2DAr = []
        for shower in zBinRadEs:
            try:
                zBin2DAr.append(shower[zBin])
            except:
                zBin2DAr.append(np.array([0]))
        zBin2DAr = equaliseRowLength( np.asarray(zBin2DAr) )
        zBin2DArAves = [ np.nanmean(zBin2DAr[:,i]) for i in range(len(zBin2DAr[0])) ]
        zBinRadEAves.append(zBin2DArAves)

    return equaliseRowLength( np.asarray(zBinRadEAves) )
    

#-------------------------------------------------------------------------------

def normHistToUnity( hist ):

    vals, bins = hist
    maxVal = max(vals)
    normedVals = [ i/maxVal for i in vals ]
    return (normedVals,bins)


#-------------------------------------------------------------------------------

def peakValue( hist ):
    values, edges = hist
    peak   = 0
    for index, i in enumerate(values):
        if i == max(values):
            peak = index
            break
    return edges[peak] + ( (edges[1]-edges[0])/2 )

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':

    print("This is a python file for storing useful functions for use in other scripts")

# ===== /Main Program/ =====

