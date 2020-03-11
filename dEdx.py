# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import utilFuncs as uf

# ===== /Import Modules/ =====

# ===== Define Functions =====

def peakValue( hist ):
    values, edges = hist
    peak   = 0
    for index, i in enumerate(values):
        if i == max(values):
            peak = index
            break
    return edges[peak] + ( (edges[1]-edges[0])/2 )
        


# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':

    variables = ['dEdx','aarondEdx']
        
    rfile      = up.open( "singleElectron_1GeV_sceON_keepOFF_recomb0715.root" )
    rfile_phot = up.open( "singlePhoton_1GeV_sceON_keepOFF_recomb0715.root"   )
    
    df      = rfile["pdAnaTree/AnaTree"].pandas.df( variables )
    df_phot = rfile_phot["pdAnaTree/AnaTree"].pandas.df( variables )
        
    rfile2 = up.open( "PDSPProd2_1GeV.root" )
    df2    = rfile2["pdAnaTree/AnaTree"].pandas.df( variables )

    hist1, binEdges1 = uf.normHistToUnity(np.histogram( df.dEdx,       range=(0,10), bins=50 ))
    hist2, binEdges2 = uf.normHistToUnity(np.histogram( df_phot.dEdx,  range=(0,10), bins=50 ))
    hist3, binEdges3 = uf.normHistToUnity(np.histogram( df2.dEdx,      range=(0,10), bins=50 ))
    hist4, binEdges4 = uf.normHistToUnity(np.histogram( df2.aarondEdx, range=(0,10), bins=50 ))

   
    fig = plt.figure(figsize=(26,14))
    plt.hist( binEdges1[:-1], binEdges1, weights=hist1, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label='Electrons' )
    plt.hist( binEdges1[:-1], binEdges1, weights=hist2, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label='Photons' )
#    plt.hist( binEdges1[:-1], binEdges1, weights=hist3, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label='PDSPProd2' )
#    plt.hist( binEdges1[:-1], binEdges1, weights=hist4, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label='PDSPProd2' )
    plt.axvline(2.1, c='r', lw=1, ls='--')
    plt.axvline(4.2, c='b', lw=1, ls='--')
    plt.xlabel('dE/dx (MeV/cm)',fontweight="bold")
    #plt.subplots_adjust(left=0.02, bottom=0.04, right=0.98, top=0.97, wspace=0.13, hspace=0.18)
    plt.legend()
    plt.show()
# ===== /Main Program/ =====

