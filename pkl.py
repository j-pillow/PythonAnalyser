# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import seaborn as sns

# ===== /Import Modules/ =====

# ===== Define Functions =====

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
#                    "nonCorrectedHit3DX", "nonCorrectedHit3DY", "nonCorrectedHit3DZ",
#                    "correctedHit3DX", "correctedHit3DY", "correctedHit3DZ",
#                    "dEdt", 
#                     "hitCharge", "hitTrueEnergy",
                ]


    print("Open File 1")
    rfile1 = up.open( "./data/singleParticles/electrons/singleElectron_1GeV_trimmed_cnn.root" )
    print("Open File 2")
    rfile2 = up.open( "./data/singleParticles/electrons/singleElectron_2GeV_trimmed_cnn.root" )
    print("Open File 3")
    rfile3 = up.open( "./data/singleParticles/electrons/singleElectron_3GeV_trimmed_cnn.root" )
    print("Open File 4")
    rfile4 = up.open( "./data/singleParticles/electrons/singleElectron_4GeV_trimmed_cnn.root" )
    print("Open File 5")
    rfile5 = up.open( "./data/singleParticles/electrons/singleElectron_5GeV_trimmed_cnn.root" )
    print("Open File 6")
    rfile6 = up.open( "./data/singleParticles/electrons/singleElectron_6GeV_trimmed_cnn.root" )
    print("Open File 7")
    rfile7 = up.open( "./data/singleParticles/electrons/singleElectron_7GeV_trimmed_cnn.root" )
    
    print("Pandas File 1")
    df1 = rfile1["pdAnaTree/AnaTree"].pandas.df(variables)
    print("Pandas File 2")
    df2 = rfile2["pdAnaTree/AnaTree"].pandas.df(variables)
    print("Pandas File 3")
    df3 = rfile3["pdAnaTree/AnaTree"].pandas.df(variables)
    print("Pandas File 4")
    df4 = rfile4["pdAnaTree/AnaTree"].pandas.df(variables)
    print("Pandas File 5")
    df5 = rfile5["pdAnaTree/AnaTree"].pandas.df(variables)
    print("Pandas File 6")
    df6 = rfile6["pdAnaTree/AnaTree"].pandas.df(variables)
    print("Pandas File 7")
    df7 = rfile7["pdAnaTree/AnaTree"].pandas.df(variables)

    df1 = df1[ ( df1.nHits > 250) & ( df1.nHits < 650 ) ]
    df2 = df2[ ( df2.nHits > 500) & ( df2.nHits < 1100 ) ]
    df3 = df3[ ( df3.nHits > 900) & ( df3.nHits < 1600 ) ]
    df4 = df4[ ( df4.nHits > 1250) & ( df4.nHits < 2000 ) ]
    df5 = df5[ ( df5.nHits > 1500) & ( df5.nHits < 2500 ) ]
    df6 = df6[ ( df6.nHits > 1800) & ( df6.nHits < 3000 ) ]
    df7 = df7[ ( df7.nHits > 2000) & ( df7.nHits < 3500 ) ]

    frames = [df1,df2,df3,df4,df5,df6,df7]

    df = pd.concat(frames,ignore_index=True)

    df.to_pickle("./singleElectron_1_to_7_GeV.pkl")

# ===== /Main Program/ =====

