# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit
import scipy.special as sps
from sklearn import preprocessing
from sklearn import linear_model

# ===== /Import Modules/ =====

# ===== Define Functions =====

def gammaFunc( x, a, b, c ):
    return c*b*( ( ((b*x)**(a-1)) * np.exp(-1*b*x) )/sps.gamma(a) )

def gammaFunc2( x, a, b, c ):
    return c*(x**a)*np.exp(-b*x)

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

# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':


    #rfile = up.open( "singleElectron_1GeV_sceON_keepOFF_recomb0715.root" )
    rfileProd2 = up.open( "./data/PDSPProd2/PDSPProd2_1GeV.root" )
    dfProd2 = rfileProd2["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
    #rfile1 = up.open( "./data/singleParticles/electrons/singleElectron_1GeV_trimmed.root" )
    rfile1 = up.open( "./data/PDSPProd2/PDSPProd2_3GeV.root" )
    df1 = rfile1["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
#    rfile2 = up.open( "singleElectron_2GeV_sceON_keepOFF_recomb0715.root" )
#    df2 = rfile2["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
#    rfile3 = up.open( "singleElectron_3GeV_sceON_keepOFF_recomb0715.root" )
#    df3 = rfile3["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
#    rfile4 = up.open( "singleElectron_4GeV_sceON_keepOFF_recomb0715.root" )
#    df4 = rfile4["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
#    rfile5 = up.open( "singleElectron_5GeV_sceON_keepOFF_recomb0715.root" )
#    df5 = rfile5["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
#    rfile6 = up.open( "singleElectron_6GeV_sceON_keepOFF_recomb0715.root" )
#    df6 = rfile6["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )
#    rfile7 = up.open( "singleElectron_7GeV_sceON_keepOFF_recomb0715.root" )
#    df7 = rfile7["pdAnaTree/AnaTree"].pandas.df( ["dEdt"],flatten=False )

    dEdtProd2 = makeYaxis( dfProd2 )
    dEdt1     = makeYaxis( df1     )
#    dEdt2     = makeYaxis( df2     )
#    dEdt3     = makeYaxis( df3     )
#    dEdt4     = makeYaxis( df4     )
#    dEdt5     = makeYaxis( df5     )
#    dEdt6     = makeYaxis( df6     )
#    dEdt7     = makeYaxis( df7     )

    dEdtProd2 = dEdtProd2[:70]
    dEdt1     = dEdt1[:70]
#    dEdt2     = dEdt2[:200]
#    dEdt3     = dEdt3[:200]
#    dEdt4     = dEdt4[:200]
#    dEdt5     = dEdt5[:200]
#    dEdt6     = dEdt6[:200]
#    dEdt7     = dEdt7[:200]

    x = [ (i + 2)/14 for i in range(0,len(dEdtProd2)*4,4) ]
    print(len(x))
    print(len(dEdtProd2))
    
    init_vals = [ 3, 1, 700 ]
    best_vals, covar = curve_fit( gammaFunc, x, dEdtProd2, p0=init_vals )
    print(covar)
    print(np.sqrt(np.diag(covar)))
    print(best_vals)
    y = [ gammaFunc(t, best_vals[0], best_vals[1], best_vals[2]) for t in x ]
    
    best_vals2, covar2 = curve_fit( gammaFunc, x, dEdt1, p0=init_vals )
    print(best_vals2)
    y2 = [ gammaFunc(t, best_vals2[0], best_vals2[1], best_vals2[2]) for t in x ]

    fig = plt.figure(figsize=(10,10))
    plt.errorbar( x, dEdtProd2, xerr=2/14, fmt="o", c='C0' )
    plt.errorbar( x, dEdt1,     xerr=2/14, fmt="o", c='C1' )
    #plt.plot( x, y,  c='r' )
    #plt.plot( x, y2, c='b' )
#    plt.scatter( x, dEdt2,     s=2 )
#    plt.scatter( x, dEdt3,     s=2 )
#    plt.scatter( x, dEdt4,     s=2 )
#    plt.scatter( x, dEdt5,     s=2 )
#    plt.scatter( x, dEdt6,     s=2 )
#    plt.scatter( x, dEdt7,     s=2 )
    plt.show()

# ===== /Main Program/ =====

