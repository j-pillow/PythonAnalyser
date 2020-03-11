# ===== Import Modules =====

import argparse
import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
from lmfit import Model
from lmfit.models import ExponentialGaussianModel, ExponentialModel, GaussianModel, SkewedGaussianModel, SkewedVoigtModel, VoigtModel
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import linear_model
from sklearn import preprocessing

import utilFuncs as uf

# ===== /Import Modules/ =====

# ===== Define Functions =====
def makeYaxis( df ):
    maxLen = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        maxLen = len(list(row.dEdt)) if len(list(row.dEdt)) > maxLen else maxLen

    dEdtVec      = [ 0 for i in range(maxLen) ]
    dEdtCountVec = [ 0 for i in range(maxLen) ]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        for index, i in enumerate(list(row.dEdt)):
            dEdtVec[index] += i
            dEdtCountVec[index] += 1

    dEdtVec = [ i/j for i,j in zip( dEdtVec, dEdtCountVec) ]

    dEdtVecAve = []
    for i in range(0,len(dEdtVec),2):
        try:
            dEdtVecAve.append( (dEdtVec[i] + dEdtVec[i+1])/2.0 )
        except:
            dEdtVecAve.append( dEdtVec[i] )
    dEdtVecAve2 = []
    for i in range(0,len(dEdtVecAve),2):
        try:
            dEdtVecAve2.append( (dEdtVecAve[i] + dEdtVecAve[i+1])/2.0 )
        except:
            dEdtVecAve2.append( dEdtVecAve[i] )

    return dEdtVecAve2

def binnedLongitudinalPlot( df ):
    X0 = 14
    Rm = 10

    binStdsTotals = [ 0 for i in range(94) ]
    binStdsTotalsCounts = [ 0 for i in range(94) ]

    for index, row in tqdm(df.iterrows(), total=len(df)):

        xzy   = np.array([ np.array([i,k,j]) for i,j,k in zip(row.correctedHit3DX, row.correctedHit3DY, row.correctedHit3DZ)  ])
        cent  = np.array( [ row["correctedAvePos.fX"], row["correctedAvePos.fZ"], row["correctedAvePos.fY"] ]) 
        pAxis = np.array( [ row["correctedPvec.fX"], row["correctedPvec.fZ"], row["correctedPvec.fY"] ]) 
        sAxis = np.array( [ row["correctedSvec.fX"], row["correctedSvec.fZ"], row["correctedSvec.fY"] ])
        tAxis = np.array( [ row["correctedTvec.fX"], row["correctedTvec.fZ"], row["correctedTvec.fY"] ]) 
        xzyT = []
        for point in xzy:
            x = uf.transToPCAJit( point, cent, tAxis ) 
            z = uf.transToPCAJit( point, cent, pAxis ) 
            y = uf.transToPCAJit( point, cent, sAxis ) 
            r = np.sqrt( (x*x) + (y*y) )
            xzyT.append( np.array( [x,z,y,r] ) )

        shift = abs(min([ i[1] for i in xzyT]))
        xzyT  = np.asarray([ np.array([ i[0]/Rm, (i[1]+shift)/X0, i[2]/Rm, i[3]/Rm ]) for i in xzyT ])

        binWidth = 3/X0
        zMax = max( xzyT[:,1] )
        binEdges = uf.makeBinEdges( binWidth, zMax )

        binnedPoints = uf.binShower( binEdges, xzyT )
        binnedPoints = binnedPoints[:(94 if len(binnedPoints) > 94 else len(binnedPoints))]

        binStds = uf.binStdCalc( binnedPoints )

        xyBinStds = uf.xyBinStdCalc( binStds )
        binStds = np.asarray(binStds)

        for index in range(len(binStdsTotals)):
            try:
                binStdsTotals[index] += xyBinStds[index]
                binStdsTotalsCounts[index] += 1
            except:
                binStdsTotals[index] += 0
                #binStdsTotalsCounts[index] += 0


        #fig = plt.figure(constrained_layout=False,figsize=(11.69,8.27))
        #gs = fig.add_gridspec(  nrows=2, ncols=1,
        #                        left=0.05, right=0.99,
        #                        top=0.95, bottom=0.05,
        #                        hspace=0)
        #f_ax1 = fig.add_subplot(gs[0,0])
        #plt.scatter( xzyT[:,1], xzyT[:,2], s=4 )
        #plt.axhline(0,lw=1,c='r',ls='--')
        #plt.ylabel("Secondary Axis",fontweight="bold")

        #f_ax1 = fig.add_subplot(gs[1,0])
        #plt.scatter( xzyT[:,1], xzyT[:,0], s=4 )
        #plt.axhline(0,lw=1,c='r',ls='--')
        #plt.ylabel("Tertiary Axis",fontweight="bold")
        #plt.xlabel("Primary Axis",fontweight="bold")
        #plt.show()

    return ([ i/j for i,j in zip(binStdsTotals,binStdsTotalsCounts) ], binEdges[:len(binnedPoints)])



# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':

    variables = [   "nHits", "nDaughters",
                    "dEdx", "dEdxAaron",
                    "energy", "energyCorrected", "energyAaron",
                    "depositionCorrection", "missedHitsCorrection", "noHitsCorrection", "mcIDEdiscrep", "contaminationCorrection", "pureIDEcorrection",
                    "mcInitEnergy", "mcDepEnergy", "recoMinusTrueOverTrue", "trueEnergyOfShowerHits",
                    "totalCharge", "hitPurity", "hitCompleteness", "energyPurity",
                    "pandoraAngleToMC", "correctedAngleToMC", "recursiveAngleToMC",
                    "showerStart", "cnnShowerScore",
                    "pandoraEvals", "pandoraAvePos", "pandoraPvec", "pandoraSvec", "pandoraTvec",
                    "correctedEvals", "correctedAvePos", "correctedPvec", "correctedSvec", "correctedTvec",
                    "recursiveEvals", "recursiveAvePos", "recursivePvec", "recursiveSvec", "recursiveTvec",
                    "pandoraPriEigenValLength", "pandoraSecEigenValLength", "pandoraTerEigenValLength",
                    "correctedPriEigenValLength", "correctedSecEigenValLength", "correctedTerEigenValLength",
                    "pandoraPriProjectionLength", "pandoraSecProjectionLength", "pandoraTerProjectionLength",
                    "correctedPriProjectionLength", "correctedSecProjectionLength", "correctedTerProjectionLength",
                    "nonCorrectedHit3DX", "nonCorrectedHit3DY", "nonCorrectedHit3DZ",
                    "correctedHit3DX", "correctedHit3DY", "correctedHit3DZ",
                    "dEdt", "hitCharge", "hitTrueEnergy",
                ]
    dfMain = pd.DataFrame({  
                "nHits":[], "nDaughters":[],
                "dEdx":[], "dEdxAaron":[],
                "energy":[], "energyCorrected":[], "energyAaron":[],
                "depositionCorrection":[], "missedHitsCorrection":[], "noHitsCorrection":[], "mcIDEdiscrep":[], "contaminationCorrection":[], "pureIDEcorrection":[],
                "mcInitEnergy":[], "mcDepEnergy":[], "recoMinusTrueOverTrue":[], "trueEnergyOfShowerHits":[],
                "totalCharge":[], "hitPurity":[], "hitCompleteness":[], "energyPurity":[],
                "pandoraAngleToMC":[], "correctedAngleToMC":[], "recursiveAngleToMC":[],
                "showerStart":[], "cnnShowerScore":[],
                "pandoraEvals":[], "pandoraAvePos":[], "pandoraPvec":[], "pandoraSvec":[], "pandoraTvec":[],
                "correctedEvals":[], "correctedAvePos":[], "correctedPvec":[], "correctedSvec":[], "correctedTvec":[],
                "recursiveEvals":[], "recursiveAvePos":[], "recursivePvec":[], "recursiveSvec":[], "recursiveTvec":[],
                "pandoraPriEigenValLength":[], "pandoraSecEigenValLength":[], "pandoraTerEigenValLength":[],
                "correctedPriEigenValLength":[], "correctedSecEigenValLength":[], "correctedTerEigenValLength":[],
                "pandoraPriProjectionLength":[], "pandoraSecProjectionLength":[], "pandoraTerProjectionLength":[],
                "correctedPriProjectionLength":[], "correctedSecProjectionLength":[], "correctedTerProjectionLength":[],
                "nonCorrectedHit3DX":[], "nonCorrectedHit3DY":[], "nonCorrectedHit3DZ":[],
                "correctedHit3DX":[], "correctedHit3DY":[], "correctedHit3DZ":[],
                "dEdt":[], "hitCharge":[], "hitTrueEnergy":[],
                })

    # ---------------------------------------------------- Parse the file name and any other options

        # Create a parser object
    parser = argparse.ArgumentParser()

        # Add the command line options
    parser.add_argument( "-e", "--energy",   required=True, help="Energy")
    parser.add_argument( "-c", "--campaign", required=True, help="Campaign name")
    parser.add_argument( "-p", "--particle", default=0, help="Particle name")
    parser.add_argument( "-s", "--save",     default="/Users/james/work/ppr/Thesis/Chapters/5Chapter/Images/", help="Save location")

        # Parse the command line options
    args = parser.parse_args()

        # Set objects from the command line options
    try:
        energy   = int(args.energy)
    except:
        energy = float(args.energy)
    campaign = args.campaign
    particle = args.particle 
    saveLoc  = args.save
    saveSuffix = ""
    myFile   = ""

    if campaign == "PDSPProd2":
        myFile = "./data/PDSPProd2/PDSPProd2_{}GeV.root".format(energy)
        saveSuffix = "PDSPProd2"
    elif (campaign == "singleParticles") or (campaign == "single") or (campaign == "singleParticle"):
        if particle == 0:
            print("Please choose a particle type with -p")
            sys.exit()
        elif (particle.lower() == "electron") or (particle.lower() == "ele") or (particle.lower() == "e"):
            myFile = "./data/singleParticles/electrons/singleElectron_{}GeV_trimmed_cnn.root".format(energy)
            saveSuffix = "singleElectron"
        elif (particle.lower() == "photon") or (particle.lower() == "phot"):
            myFile = "./data/singleParticles/photons/singlePhoton_{}GeV_trimmed_cnn.root".format(energy)
            saveSuffix = "singlePhoton"
        elif (particle.lower() == "positron") or (particle.lower() == "pos"):
            myFile = "./data/singleParticles/positrons/singlePositron_{}GeV_trimmed_cnn.root".format(energy)
            saveSuffix = "singlePositron"
        else:
            print("Particle type {} is either not recognised, or ambiguous".format(particle))
            sys.exit()

    else:
        print("-c {} not recognised.".format(campaign))
        sys.exit()


    print()
    print("============================================")
    print()
    print("File: {}".format(myFile))

    # ---------------------------------------------------- Import the root files


    tree   = up.open( myFile )["pdAnaTree/AnaTree"]

    for df in tqdm(up.pandas.iterate(myFile,"pdAnaTree/AnaTree",variables,flatten=False,entrysteps=500),total=tree.numentries/500):
        dfMain = dfMain.append(df,ignore_index=True)


        # Set hit cuts
    hitLowerCut = 0
    hitUpperCut = 100000000000000
    if campaign == "PDSPProd2":
        if energy == 1:
            hitLowerCut = 200
            hitUpperCut = 600
        elif energy == 2:
            hitLowerCut = 450
            hitUpperCut = 1100
        elif energy == 3:
            hitLowerCut = 800
            hitUpperCut = 1600
        elif energy == 6:
            hitLowerCut = 1500
            hitUpperCut = 2800
        elif energy == 7:
            hitLowerCut = 1800
            hitUpperCut = 3300
    else:
        if energy == 0.3:
            hitLowerCut = 80
            hitUpperCut = 250
        if energy == 0.5:
            hitLowerCut = 50
            hitUpperCut = 550
        if energy == 1:
            hitLowerCut = 250
            hitUpperCut = 650
        elif energy == 2:
            hitLowerCut = 500
            hitUpperCut = 1100
        elif energy == 3:
            hitLowerCut = 900
            hitUpperCut = 1600
        elif energy == 4:
            hitLowerCut = 1250
            hitUpperCut = 2000
        elif energy == 5:
            hitLowerCut = 1500
            hitUpperCut = 2500
        elif energy == 6:
            hitLowerCut = 1800
            hitUpperCut = 3000
        elif energy == 7:
            hitLowerCut = 2000
            hitUpperCut = 3500

    df = dfMain.copy(deep=True)
    df = df[ (df.nHits > hitLowerCut) & (df.nHits < hitUpperCut) ]
    df["totalCorrection"] = df.mcIDEdiscrep + df.pureIDEcorrection + df.contaminationCorrection + df.noHitsCorrection + df.missedHitsCorrection + df.depositionCorrection
    df = df[ df.totalCorrection < df.mcInitEnergy ]
    df["recoMinusTrueOverTrueNonCorr"] = (df.energy - df.mcInitEnergy)/df.mcInitEnergy

    df = df[ df.cnnShowerScore > 0.7 ]
    df = df[ df.hitPurity > 0.85 ]
    print(len(df))
    #df = df[ df.nDaughters == 0 ]
    print(len(df))
    # ======================================================
    # =================== Number of Hits ===================
    # ======================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    nhits, nhitsBins = uf.normHistToUnity( np.histogram( dfMain.nHits, range=(0,int(hitUpperCut*1.1)), bins=35 ) )
    plt.hist( nhitsBins[:-1], nhitsBins, weights=nhits, histtype='stepfilled', edgecolor='None', fc='White', alpha=1, zorder=2)
    plt.hist( nhitsBins[:-1], nhitsBins, weights=nhits, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.2, zorder=3)
    plt.hist( nhitsBins[:-1], nhitsBins, weights=nhits, histtype='stepfilled', edgecolor='C0', fc='None', lw=2, label="nHits", zorder=4)
    plt.axvline(hitLowerCut, lw=2, color='C1', zorder=1)
    plt.axvline(hitUpperCut, lw=2, color='C1', zorder=1)
    plt.axvspan(xmin=hitLowerCut,xmax=hitUpperCut, alpha=0.1, color='C1', zorder=1)
    
    plt.xlabel( "nHits", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}nHits_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}nHits_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
    # =================================================
    # =================== CNN Score ===================
    # =================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    cnn, cnnBins = uf.normHistToUnity( np.histogram( dfMain.cnnShowerScore, range=(0,1), bins=35 ) )
    plt.hist( cnnBins[:-1], cnnBins, weights=cnn, histtype='stepfilled', edgecolor='None', fc='White', alpha=1, zorder=2)
    plt.hist( cnnBins[:-1], cnnBins, weights=cnn, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.2, zorder=3)
    plt.hist( cnnBins[:-1], cnnBins, weights=cnn, histtype='stepfilled', edgecolor='C0', fc='None', lw=2, label="nHits", zorder=4)
    #plt.axvline(hitLowerCut, lw=2, color='C1', zorder=1)
    #plt.axvline(hitUpperCut, lw=2, color='C1', zorder=1)
    #plt.axvspan(xmin=hitLowerCut,xmax=hitUpperCut, alpha=0.1, color='C1', zorder=1)
    
    plt.xlabel( "CNN Shower Score", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}cnn_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}cnn_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

    # ===================================================
    # =================== Start Angle ===================
    # ===================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( np.cos(df.recursiveAngleToMC),    range=(0.95,1), bins=40, histtype='stepfilled', edgecolor="None", fc="C0",   alpha=0.1 )
    plt.hist( np.cos(df.correctedAngleToMC), range=(0.95,1), bins=40, histtype='stepfilled', edgecolor="None", fc="C1",   alpha=0.1 )
    plt.hist( np.cos(df.recursiveAngleToMC),    range=(0.95,1), bins=40, histtype='stepfilled', edgecolor="C0",   fc="None", lw=2,     label='Weighted Recursive PCA')
    plt.hist( np.cos(df.correctedAngleToMC), range=(0.95,1), bins=40, histtype='stepfilled', edgecolor="C1",   fc="None", lw=2,     label='Unweighted Single PCA' )
   
    plt.xlabel(r'Angle between PCA and MC Direction ($\cos(\theta)$)', fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16,loc=2)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
  
    print("plt.savefig({}PCA_angle_to_MC_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}PCA_angle_to_MC_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    


    # ===============================================================
    # =================== Purity and Completeness ===================
    # ===============================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( df.hitPurity,       range=(0,1), bins=40, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( df.hitCompleteness, range=(0,1), bins=40, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( df.hitPurity,       range=(0,1), bins=40, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label='Hit Purity' )
    plt.hist( df.hitCompleteness, range=(0,1), bins=40, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label='Hit Completeness' )
    
    plt.xlabel( "Hit Purity and Completeness", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16,loc=2)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    
    print("plt.savefig({}Purity_and_Completeness_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}Purity_and_Completeness_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    


    # ==================================================================
    # =================== Purity and Completeness 2D ===================
    # ==================================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist2d(df.hitPurity, df.hitCompleteness, range=[[0,1],[0,1]], bins=100, cmin=1)

    plt.xlabel('Purity')
    plt.ylabel('Completeness')
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}Purity_and_Completeness_2D_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}Purity_and_Completeness_2D_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()



    # ===================================================
    # =================== Eigenvalues ===================
    # ===================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))


    df["correctedEval0Pcent"] = df["correctedEvals.fX"] / ( df["correctedEvals.fX"] + df["correctedEvals.fY"] + df["correctedEvals.fZ"] )
    df["correctedEval1Pcent"] = df["correctedEvals.fY"] / ( df["correctedEvals.fX"] + df["correctedEvals.fY"] + df["correctedEvals.fZ"] )
    df["correctedEval2Pcent"] = df["correctedEvals.fZ"] / ( df["correctedEvals.fX"] + df["correctedEvals.fY"] + df["correctedEvals.fZ"] )
    
    df["recursiveEval0Pcent"] = df["recursiveEvals.fX"] / ( df["recursiveEvals.fX"] + df["recursiveEvals.fY"] + df["recursiveEvals.fZ"] )
    df["recursiveEval1Pcent"] = df["recursiveEvals.fY"] / ( df["recursiveEvals.fX"] + df["recursiveEvals.fY"] + df["recursiveEvals.fZ"] )
    df["recursiveEval2Pcent"] = df["recursiveEvals.fZ"] / ( df["recursiveEvals.fX"] + df["recursiveEvals.fY"] + df["recursiveEvals.fZ"] )

    pHist, pBins = uf.normHistToUnity( np.histogram(df.correctedEval0Pcent, range=(0,1), bins=100) )
    sHist, pBins = uf.normHistToUnity( np.histogram(df.correctedEval1Pcent, range=(0,1), bins=100) )
    tHist, pBins = uf.normHistToUnity( np.histogram(df.correctedEval2Pcent, range=(0,1), bins=100) )

    plt.hist( pBins[:-1], pBins, weights=pHist, histtype='stepfilled', edgecolor="None", fc="C0", alpha=0.1)
    plt.hist( pBins[:-1], pBins, weights=sHist, histtype='stepfilled', edgecolor="None", fc="C1", alpha=0.1)
    plt.hist( pBins[:-1], pBins, weights=tHist, histtype='stepfilled', edgecolor="None", fc="C2", alpha=0.1)
    plt.hist( pBins[:-1], pBins, weights=pHist, histtype='stepfilled', edgecolor="C0", fc="None", lw=2, label="Primary Axis"  )
    plt.hist( pBins[:-1], pBins, weights=sHist, histtype='stepfilled', edgecolor="C1", fc="None", lw=2, label="Secondary Axis")
    plt.hist( pBins[:-1], pBins, weights=tHist, histtype='stepfilled', edgecolor="C2", fc="None", lw=2, label="Tertiary Axis" )

    plt.xlabel( "Fractional explained variance", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    print("plt.savefig({}fractional_explained_variance_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}fractional_explained_variance_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

    

    # ==============================================
    # =================== Length ===================
    # ==============================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    
    eValLenNew, eValLenNewBins = uf.normHistToUnity( np.histogram( df.correctedPriEigenValLength, range=(0,500), bins=50 ) )
    eValLenNewPeak             = uf.peakValue(       np.histogram( df.correctedPriEigenValLength, range=(0,500), bins=50 ) )
    projLenNew, projLenNewBins = uf.normHistToUnity( np.histogram( df.correctedPriProjectionLength, range=(0,500), bins=50 ) )
    projLenNewPeak             = uf.peakValue(       np.histogram( df.correctedPriProjectionLength, range=(0,500), bins=50 ) )

    plt.hist( eValLenNewBins[:-1], eValLenNewBins, weights=eValLenNew, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1)
    plt.hist( projLenNewBins[:-1], projLenNewBins, weights=projLenNew, histtype='stepfilled', edgecolor='None', fc='C1', alpha=0.1)
    plt.hist( eValLenNewBins[:-1], eValLenNewBins, weights=eValLenNew, histtype='stepfilled', edgecolor='C0', fc='None', lw=2, label="Eigenvalue Length Peak Value: {}cm".format(eValLenNewPeak))
    plt.hist( projLenNewBins[:-1], projLenNewBins, weights=projLenNew, histtype='stepfilled', edgecolor='C1', fc='None', lw=2, label="Max Projection Length Peak Value: {}cm".format(projLenNewPeak))

    plt.xlabel( "PCA Length (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}PCA_Proj_Length_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}PCA_Proj_Length_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

    

    # =============================================
    # =================== dE/dx ===================
    # =============================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    
    dEdx, dEdxBins = uf.normHistToUnity( np.histogram( df.dEdx, range=(0,10), bins=50 ) )
    dEdxPeak = uf.peakValue( np.histogram( df.dEdx, range=(0,10), bins=50 ) )
   
    plt.hist( dEdxBins[:-1], dEdxBins, weights=dEdx, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1)
    plt.hist( dEdxBins[:-1], dEdxBins, weights=dEdx, histtype='stepfilled', edgecolor='C0', fc='None', lw=2, label="dE/dx Peak Value: {}MeV/cm".format(round(dEdxPeak,3)))
   
    plt.xlabel( "dE/dx (MeV/cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    print("plt.savefig({}dEdx_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}dEdx_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

    

    # ====================================================
    # =================== XYZ Position ===================
    # ====================================================

    ncx=[]
    cx=[]
    ncy=[]
    cy=[]
    ncz=[]
    cz=[]
    
    minncx=[]
    mincx=[]
    minncy=[]
    mincy=[]
    minncz=[]
    mincz=[]
    
    maxncx=[]
    maxcx=[]
    maxncy=[]
    maxcy=[]
    maxncz=[]
    maxcz=[]
    for index, row in tqdm(df.iterrows(), total=len(df)):
        rowNCXlist = list(row.nonCorrectedHit3DX)
        rowCXlist  = list(row.correctedHit3DX)
        rowNCYlist = list(row.nonCorrectedHit3DY)
        rowCYlist  = list(row.correctedHit3DY)
        rowNCZlist = list(row.nonCorrectedHit3DZ)
        rowCZlist  = list(row.correctedHit3DZ)

        ncx += rowNCXlist
        cx  += rowCXlist 
        ncy += rowNCYlist
        cy  += rowCYlist 
        ncz += rowNCZlist
        cz  += rowCZlist 

        minncx.append( rowNCXlist[ rowNCZlist.index(min(rowNCZlist)) ] )
        mincx.append(  rowCXlist[  rowCZlist.index(min(rowCZlist))   ] )
        minncy.append( rowNCYlist[ rowNCZlist.index(min(rowNCZlist)) ] )
        mincy.append(  rowCYlist[  rowCZlist.index(min(rowCZlist))   ] )
        minncz.append( min(rowNCZlist) )
        mincz.append(  min(rowCZlist)  )
        
        maxncx.append( rowNCXlist[ rowNCZlist.index(max(rowNCZlist)) ] )
        maxcx.append(  rowCXlist[  rowCZlist.index(max(rowCZlist))   ] )
        maxncy.append( rowNCYlist[ rowNCZlist.index(max(rowNCZlist)) ] )
        maxcy.append(  rowCYlist[  rowCZlist.index(max(rowCZlist))   ] )
        maxncz.append( max(rowNCZlist) )
        maxcz.append(  max(rowCZlist)  )


        # X-Position
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( ncx, range=(-100,000), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( ncx, range=(-100,000), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label="Before SCE Corrections" )
    plt.hist( cx,  range=(-100,000), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( cx,  range=(-100,000), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label="After SCE Corrections" )
    
    plt.xlabel( "X Position (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    
    print("plt.savefig({}x_position_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}x_position_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
        # Y-Position
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( ncy, range=(300,500), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( ncy, range=(300,500), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label="Before SCE Corrections" )
    plt.hist( cy,  range=(300,500), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( cy,  range=(300,500), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label="After SCE Corrections" )
    
    plt.xlabel( "Y Position (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    
    print("plt.savefig({}y_position_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}y_position_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Z-Position
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( ncz, range=(0,300), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( ncz, range=(0,300), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label="Before SCE Corrections" )
    plt.hist( cz,  range=(0,300), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( cz,  range=(0,300), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label="After SCE Corrections" )
    
    plt.xlabel( "Z Position (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    
    print("plt.savefig({}z_position_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}z_position_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # X-Position Min/Max
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( minncx, range=(-100,000), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( minncx, range=(-100,000), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label="Min Before SCE Corrections" )
    plt.hist( maxncx, range=(-100,000), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( maxncx, range=(-100,000), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, ls='--', label="Max Before SCE Corrections" )
    plt.hist( mincx,  range=(-100,000), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( mincx,  range=(-100,000), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label="Min After SCE Corrections" )
    plt.hist( maxcx,  range=(-100,000), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( maxcx,  range=(-100,000), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, ls='--', label="Max After SCE Corrections" )
    
    plt.xlabel( "X Position (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    
    print("plt.savefig({}x_min_max_position_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}x_min_max_position_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
        # Y-Position Min/Max
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( minncy, range=(300,500), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( minncy, range=(300,500), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label="Min Before SCE Corrections" )
    plt.hist( maxncy, range=(300,500), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( maxncy, range=(300,500), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, ls='--', label="Max Before SCE Corrections" )
    plt.hist( mincy,  range=(300,500), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( mincy,  range=(300,500), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label="Min After SCE Corrections" )
    plt.hist( maxcy,  range=(300,500), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( maxcy,  range=(300,500), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, ls='--', label="Max After SCE Corrections" )
    
    plt.xlabel( "Y Position (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    
    print("plt.savefig({}y_min_max_position_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}y_min_max_position_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Z-Position Min/Max
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.subplot(2,1,1)
    plt.hist( minncz, range=(0,50), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( minncz, range=(0,50), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, label="Min Before SCE Corrections" )
    plt.hist( mincz,  range=(0,50), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( mincz,  range=(0,50), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, label="Min After SCE Corrections" )
    plt.subplot(2,1,2)
    plt.hist( maxncz, range=(0,300), bins=50, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( maxncz, range=(0,300), bins=50, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2, ls='--', label="Max Before SCE Corrections" )
    plt.hist( maxcz,  range=(0,300), bins=50, histtype='stepfilled', edgecolor='None', fc='C1',   alpha=0.1 )
    plt.hist( maxcz,  range=(0,300), bins=50, histtype='stepfilled', edgecolor='C1',   fc='None', lw=2, ls='--', label="Max After SCE Corrections" )
    
    plt.xlabel( "Z Position (cm)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    
    print("plt.savefig({}z_min_max_position_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}z_min_max_position_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()


    # ============================================================
    # =================== Longitudinal Profile ===================
    # ============================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    dEdt = (makeYaxis( df ))[:70]
    x = [ (i + 2)/14 for i in range(0,len(dEdt)*4,4) ]
    plt.scatter( x, dEdt )

    plt.xlabel( "dE/dt (MeV/X0)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    #plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    print("plt.savefig({}dEdt_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}dEdt_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()


    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    xyBinStds, binEdges = binnedLongitudinalPlot( df )
    binEdges = [ i for i in np.arange(0,(3.0/14)*len(xyBinStds),3.0/14) ]
    print(len(xyBinStds))
    print(len(binEdges))
    xPlotMin  = -(3.0/14.0)
    xPlotMax  = max(binEdges)+(3.0/14.0)
    plotRange = xPlotMax - xPlotMin

#    plt.scatter(binEdges,xyBinStds)
    for i in binEdges:
        plt.axvline( x=i, c='r', lw=.5, ls='--' )

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
    plt.xlim(xPlotMin,xPlotMax)
    plt.xlabel( r'Primary Axis ($X_0$)', fontweight="bold", fontsize=20 )
    plt.ylabel( r'Cross-sectional Area ($R_m^2$)', fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}binned_longitudinal_shower_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}binned_longitudinal_shower_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

#    plt.show()

    # ==========================================================
    # =================== Transverse Profile ===================
    # ==========================================================
    


    # ====================================================
    # =================== Shower Start ===================
    # ====================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    
    showerStart, showerStartBins = uf.normHistToUnity( np.histogram( df.showerStart, range=(0,5), bins=50 ) )
   
    plt.hist( showerStartBins[:-1], showerStartBins, weights=showerStart, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( showerStartBins[:-1], showerStartBins, weights=showerStart, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )
    
    plt.xlabel( r'Shower MIP Section Length ($X_0$)', fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}showerStart_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}showerStart_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()



    # ==============================================
    # =================== Energy ===================
    # ==============================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    
    energyHist, energyBins = uf.normHistToUnity( np.histogram( df.energy, range=((energy*0.75)-1,(energy*0.75)+1), bins=20 ) )

    plt.hist( energyBins[:-1], energyBins, weights=energyHist, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( energyBins[:-1], energyBins, weights=energyHist, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )
    
    plt.xlabel( "Estimated Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}energy_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}energy_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()


    # ==========================================================
    # =================== Energy Corrections ===================
    # ==========================================================
   
        # Deposition Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    
    depCorrection, depCorrectionBins = uf.normHistToUnity( np.histogram( df.depositionCorrection/df.mcInitEnergy, range=(0,0.04), bins=30 ) )

    plt.hist( depCorrectionBins[:-1], depCorrectionBins, weights=depCorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( depCorrectionBins[:-1], depCorrectionBins, weights=depCorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "Deposition Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}deposition_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}deposition_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Deposition Correction 2D
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    maxDepositionCorrection = max(list( df.depositionCorrection/df.mcInitEnergy ))
    minDepositionCorrection = min(list( df.depositionCorrection/df.mcInitEnergy ))
    plt.hist2d( df.mcInitEnergy, df.depositionCorrection/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), (minDepositionCorrection,maxDepositionCorrection) ), bins=100, cmin=1 )

    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Deposition Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.colorbar()
   
    print("plt.savefig({}initEnergy_vs_deposition_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initEnergy_vs_deposition_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
        # Missed Hits Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    missedHitsCorrection, missedHitsCorrectionBins = uf.normHistToUnity( np.histogram( df.missedHitsCorrection/df.mcInitEnergy, range=(0,0.2), bins=30 ) )

    plt.hist( missedHitsCorrectionBins[:-1], missedHitsCorrectionBins, weights=missedHitsCorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( missedHitsCorrectionBins[:-1], missedHitsCorrectionBins, weights=missedHitsCorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "Missed Hits Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}missed_hits_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}missed_hits_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Missed Hits Correction 2D
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    
    maxMissedHitsCorrection = max(list( df.missedHitsCorrection/df.mcInitEnergy ))
    minMissedHitsCorrection = min(list( df.missedHitsCorrection/df.mcInitEnergy ))
    plt.hist2d( df.mcInitEnergy, df.missedHitsCorrection/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), (minMissedHitsCorrection,maxMissedHitsCorrection) ), bins=100, cmin=1 )

    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Missed Hits Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.colorbar()
   
    print("plt.savefig({}initEnergy_vs_missed_hits_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initEnergy_vs_missed_hits_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # No Hits Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    noHitsCorrection, noHitsCorrectionBins = uf.normHistToUnity( np.histogram( df.noHitsCorrection/df.mcInitEnergy, range=(0.15,0.4), bins=30 ) )

    plt.hist( noHitsCorrectionBins[:-1], noHitsCorrectionBins, weights=noHitsCorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( noHitsCorrectionBins[:-1], noHitsCorrectionBins, weights=noHitsCorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "No Hits Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}no_hits_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}no_hits_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # No Hits Correction 2D
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    maxNoHitsCorrection = max(list( df.noHitsCorrection/df.mcInitEnergy ))
    minNoHitsCorrection = min(list( df.noHitsCorrection/df.mcInitEnergy ))
    plt.hist2d( df.mcInitEnergy, df.noHitsCorrection/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), (minNoHitsCorrection,maxNoHitsCorrection) ), bins=100, cmin=1 )

    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "No Hits Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.colorbar()
   
    print("plt.savefig({}initEnergy_vs_no_hits_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initEnergy_vs_no_hits_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

        # Contamination Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    contaminationCorrection, contaminationCorrectionBins = uf.normHistToUnity( np.histogram( df.contaminationCorrection/df.mcInitEnergy, range=(0,0.3), bins=30 ) )

    plt.hist( contaminationCorrectionBins[:-1], contaminationCorrectionBins, weights=contaminationCorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( contaminationCorrectionBins[:-1], contaminationCorrectionBins, weights=contaminationCorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "Contamination Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}contamination_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}contamination_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Contamination Correction 2D
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    maxContaminationCorrection = max(list( df.contaminationCorrection/df.mcInitEnergy ))
    minContaminationCorrection = min(list( df.contaminationCorrection/df.mcInitEnergy ))
    plt.hist2d( df.mcInitEnergy, df.contaminationCorrection/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), (minContaminationCorrection,maxContaminationCorrection) ), bins=100, cmin=1 )

    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Contamination Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.colorbar()
   
    print("plt.savefig({}initEnergy_vs_contamination_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initEnergy_vs_contamination_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Pure IDE Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    pureIDEcorrection, pureIDEcorrectionBins = uf.normHistToUnity( np.histogram( df.pureIDEcorrection/df.mcInitEnergy, range=(-0.15,0.05), bins=30 ) )

    plt.hist( pureIDEcorrectionBins[:-1], pureIDEcorrectionBins, weights=pureIDEcorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( pureIDEcorrectionBins[:-1], pureIDEcorrectionBins, weights=pureIDEcorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "Energy Estimation Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}estimation_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}estimation_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Pure IDE Correction 2D
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    maxPureIDEcorrection = max(list( df.pureIDEcorrection/df.mcInitEnergy ))
    minPureIDEcorrection = min(list( df.pureIDEcorrection/df.mcInitEnergy ))
    plt.hist2d( df.mcInitEnergy, df.pureIDEcorrection/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), (minPureIDEcorrection,maxPureIDEcorrection) ), bins=100, cmin=1 )

    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Energy Estimation Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.colorbar()
   
    print("plt.savefig({}initEnergy_vs_estimation_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initEnergy_vs_estimation_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # MC IDE Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    mcIDECorrection, mcIDECorrectionBins = uf.normHistToUnity( np.histogram( df.mcIDEdiscrep/df.mcInitEnergy, range=(-0.15,0.05), bins=30 ) )

    plt.hist( mcIDECorrectionBins[:-1], mcIDECorrectionBins, weights=mcIDECorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( mcIDECorrectionBins[:-1], mcIDECorrectionBins, weights=mcIDECorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "MC IDE Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}mcIDE_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}mcIDE_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # MC IDE Correction 2D
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    maxMcIDEdiscrep = max(list( df.mcIDEdiscrep/df.mcInitEnergy ))
    minMcIDEdiscrep = min(list( df.mcIDEdiscrep/df.mcInitEnergy ))
    plt.hist2d( df.mcInitEnergy, df.mcIDEdiscrep/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), (minMcIDEdiscrep,maxMcIDEdiscrep) ), bins=100, cmin=1 )

    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "MC IDE Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.colorbar()
   
    print("plt.savefig({}initEnergy_vs_mcIDE_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initEnergy_vs_mcIDE_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
        
        # Total Correction
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    totalCorrection, totalCorrectionBins = uf.normHistToUnity( np.histogram( (df.mcIDEdiscrep + df.pureIDEcorrection + df.contaminationCorrection + df.noHitsCorrection + df.missedHitsCorrection + df.depositionCorrection)/df.mcInitEnergy, range=(0.15,1), bins=30 ) )

    plt.hist( totalCorrectionBins[:-1], totalCorrectionBins, weights=totalCorrection, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( totalCorrectionBins[:-1], totalCorrectionBins, weights=totalCorrection, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )

    plt.xlabel( "Total Correction / Initial Energy ", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}total_correction_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}total_correction_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

        # Total Correction vs Completeness
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist2d( df.totalCorrection, df.hitCompleteness, bins=50, cmin=1 )

    plt.xlabel( "Total Correction (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Completeness", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}total_correction_vs_completeness_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}total_correction_vs_completeness_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

        # Total Correction vs Purity
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist2d( df.totalCorrection, df.hitPurity, bins=50, cmin=1 )

    plt.xlabel( "Total Correction (GeV)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Purity", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}total_correction_vs_purity_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}total_correction_vs_purity_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
        # Total Correction vs Initial Energy 
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    maxTotalCorrection = max( list(df.totalCorrection/df.mcInitEnergy) )
    minTotalCorrection = min( list(df.totalCorrection/df.mcInitEnergy) )
    plt.hist2d( df.mcInitEnergy, df.totalCorrection/df.mcInitEnergy, range=( ( min(list(df.mcInitEnergy)), max(list(df.mcInitEnergy)) ), ( minTotalCorrection, maxTotalCorrection ) ), bins=100, cmin=1 )
    plt.colorbar()

    plt.ylabel( "Total Correction / Initial Energy", fontweight="bold", fontsize=20 )
    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}total_correction_vs_initial_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}total_correction_vs_initial_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

    
    # ========================================================
    # =================== Corrected Energy ===================
    # ========================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    
    energyCorrectedHist, energyCorrectedBins = uf.normHistToUnity( np.histogram( df.energyCorrected, range=(energy-1,energy+1), bins=20 ) )

    plt.hist( energyCorrectedBins[:-1], energyCorrectedBins, weights=energyCorrectedHist, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( energyCorrectedBins[:-1], energyCorrectedBins, weights=energyCorrectedHist, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )
    
    plt.xlabel( "Estimated Energy Corrected (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}energy_corrected_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}energy_corrected_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
    

    # ===================================================
    # =================== True Energy ===================
    # ===================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    
    mcInitEnergyHist, mcInitEnergyBins = uf.normHistToUnity( np.histogram( df.mcInitEnergy, range=(energy-1,energy+1), bins=20 ) )

    plt.hist( mcInitEnergyBins[:-1], mcInitEnergyBins, weights=mcInitEnergyHist, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( mcInitEnergyBins[:-1], mcInitEnergyBins, weights=mcInitEnergyHist, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )
    
    plt.xlabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}initial_energy_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}initial_energy_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()




    # =================================================================
    # =================== Reco Minus True over True ===================
    # =================================================================
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    recoTrue, recoTrueBins = uf.normHistToUnity( np.histogram( df.recoMinusTrueOverTrue, range=(-0.1,0.1), bins=20 ) )

    plt.hist( recoTrueBins[:-1], recoTrueBins, weights=recoTrue, histtype='stepfilled', edgecolor='None', fc='C0', alpha=0.1 )
    plt.hist( recoTrueBins[:-1], recoTrueBins, weights=recoTrue, histtype='stepfilled', edgecolor='C0', fc='None', lw=2 )
    
    plt.xlabel( "(Reco - True)/True", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
   
    print("plt.savefig({}reco_minus_true_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}reco_minus_true_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()


    # =========================================================
    # =================== Energy Regression ===================
    # =========================================================

    ransacCut = 0
    if energy == 1:
        ransacCut = 105000
    if energy == 2:
        ransacCut = 220000
    if energy == 3:
        ransacCut = 340000
    if energy == 4:
        ransacCut = 470000
    if energy == 5:
        ransacCut = 580000
    if energy == 6:
        ransacCut = 690000
    if energy == 7:
        ransacCut = 820000

    df_ransac = df.copy(deep=True)
    df_ransac = df_ransac[df_ransac.totalCharge > 0]
    #df_train = df_ransac.sample(frac=1.0, random_state=42)
    #df_test  = df_ransac.loc[~df_ransac.index.isin(df_train.index)]
    
    x_train = np.asarray( (np.mean(df_ransac.totalCharge) - df_ransac.totalCharge)/max(list(np.mean(df_ransac.totalCharge) - df_ransac.totalCharge))  ).reshape(-1,1)
    y_train = np.asarray( df_ransac.mcInitEnergy ).reshape(-1,1)
    ransac  = linear_model.RANSACRegressor( random_state=42 ).fit( x_train, y_train )

    inlier_mask  = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    x_test = np.asarray( ( np.mean(df.totalCharge) - df.totalCharge)/max(list(np.mean(df.totalCharge) - df.totalCharge)) ).reshape(-1,1)
    predictions = ransac.predict( x_test )
    
    fig = plt.figure(constrained_layout=True, figsize=(10,10) )
    plt.scatter( x_train[inlier_mask],  y_train[inlier_mask],  label='Inliers',  s=6 )
    plt.scatter( x_train[outlier_mask], y_train[outlier_mask], label='Outliers', s=6 )
    plt.plot( x_test, predictions, c='r' )
    
    plt.xlabel( "Total Charge (ADC)", fontweight="bold", fontsize=20 )
    plt.ylabel( "Initial Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")

    print("plt.savefig({}charge_vs_initEnergy_ransac_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}charge_vs_initEnergy_ransac_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist2d( (df.totalCharge)/max(list(df.totalCharge)), df.mcInitEnergy, bins=100, cmin=1 )
    plt.plot( x_test, predictions, c='r' )

    plt.xlabel( "Total Charge (ADC)",  fontweight="bold", fontsize=20 )
    plt.ylabel( "Inital Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    
    print("plt.savefig({}charge_vs_initEnergy_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}charge_vs_initEnergy_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()


    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    tCharge, tChargeBins = uf.normHistToUnity( np.histogram( df.totalCharge, bins=50 ) )
    tChargePeak          = uf.peakValue(       np.histogram( df.totalCharge, bins=50 ) )
    
    plt.hist( tChargeBins[:-1], tChargeBins, weights=tCharge, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( tChargeBins[:-1], tChargeBins, weights=tCharge, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2 )
    plt.axvline( np.mean(df.totalCharge), c='C1', ls='--', label='Mean' )
    plt.axvline( tChargePeak            , c='C2', ls='--', label='Peak' )

    plt.xlabel( "Total Charge (ADC)",  fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    
    print("plt.savefig({}charge_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}charge_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    tChargeResid, tChargeResidBins = uf.normHistToUnity( np.histogram( np.absolute(tChargePeak - df.totalCharge), bins=50 ) )
    
    plt.hist( tChargeResidBins[:-1], tChargeResidBins, weights=tChargeResid, histtype='stepfilled', edgecolor='None', fc='C0',   alpha=0.1 )
    plt.hist( tChargeResidBins[:-1], tChargeResidBins, weights=tChargeResid, histtype='stepfilled', edgecolor='C0',   fc='None', lw=2 )

    plt.xlabel( "Total Charge (ADC)",  fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.legend(fontsize=16)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    
    print("plt.savefig({}charge_resid_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}charge_resid_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist2d( (np.mean(df.totalCharge) - df.totalCharge)/max(list(np.mean(df.totalCharge) - df.totalCharge)), df.mcInitEnergy, bins=100, cmin=1 )
    plt.plot( x_test, predictions, c='r' )

    plt.xlabel( "Total Charge (ADC)",  fontweight="bold", fontsize=20 )
    plt.ylabel( "Inital Energy (GeV)", fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    
    print("plt.savefig({}charge_resid_vs_initEnergy_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}charge_resid_vs_initEnergy_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()


    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( ransac.predict( np.asarray(df.totalCharge).reshape(-1,1) ), bins=50, range=(0.4,1.4), histtype="stepfilled", edgecolor="None", fc="C0", alpha=0.1 )
    plt.hist( ransac.predict( np.asarray(df.totalCharge).reshape(-1,1) ), bins=50, range=(0.4,1.4), histtype="stepfilled", edgecolor="C0", fc="None", lw=2, label="Ransac" )
    plt.hist( df.mcInitEnergy, bins=50, range=(0.4,1.4), histtype="stepfilled", edgecolor="None", fc="C1", alpha=0.1 )
    plt.hist( df.mcInitEnergy, bins=50, range=(0.4,1.4), histtype="stepfilled", edgecolor="C1", fc="None", lw=2, label="Initial Energy" )

    plt.xlabel( "Estimated Energy",  fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    plt.legend(fontsize=16)
    
    print("plt.savefig({}energy_ransac_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}energy_ransac_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()
    
    fig = plt.figure(constrained_layout=True,figsize=(10,10))

    plt.hist( (ransac.predict( np.asarray(df.totalCharge).reshape(-1,1) ) - np.asarray(df.mcInitEnergy).reshape(-1,1) )/np.asarray(df.mcInitEnergy).reshape(-1,1), bins=40, histtype="stepfilled", edgecolor="None", fc="C0", alpha=0.1 )
    plt.hist( (ransac.predict( np.asarray(df.totalCharge).reshape(-1,1) ) - np.asarray(df.mcInitEnergy).reshape(-1,1) )/np.asarray(df.mcInitEnergy).reshape(-1,1), bins=40, histtype="stepfilled", edgecolor="C0", fc="None", lw=2, label="Ransac" )

    plt.xlabel( "Estimated Energy",  fontweight="bold", fontsize=20 )
    plt.tick_params(axis="x",labelsize=20)
    plt.tick_params(axis="y",labelsize=20)
    plt.title("{}GeV {}".format(energy,saveSuffix),fontweight="bold",fontsize=20,loc="left")
    
    print("plt.savefig({}energy_ransac_resid_{}GeV_{}.pdf)".format(saveLoc,energy,saveSuffix))
    plt.savefig("{}energy_ransac_resid_{}GeV_{}.pdf".format(saveLoc,energy,saveSuffix))
    plt.close()

#    # =====================================================================
#    # =================== Energy Reco vs Number of Hits ===================
#    # =====================================================================
#
#    bins = [ i for i in range(0,650,25) ]
#    yVals = [ 0 for i in range(0,650,25) ]
#    yCounts = [ 0 for i in range(0,650,25) ]
#    rms = [ 0 for i in range(0,650,25) ]
#
#    for index, row in tqdm( df.iterrows(), total=len(df) ):
#        yVals[int(row.nHits/25)] += row.recoMinusTrueOverTrueNonCorr
#        yVals[int(row.nHits/25)] += (row.recoMinusTrueOverTrueNonCorr)**2
#        yCounts[ int(row.nHits/25) ] += 1
#
#    yVals = [ (i/j if j != 0 else 0) for i,j in zip(yVals,yCounts) ]
#    yErrs = [ (np.sqrt(i/j) if j != 0 else 0) for i,j in zip(rms,yCounts) ]
#    xErrs = [ 12.5 for i in range(len(yCounts)) ]
#
#    plt.errorbar( bins, yVals, yerr=yErrs, xerr=xErrs, fmt='.' )
#    plt.show()


# ===== /Main Program/ =====
