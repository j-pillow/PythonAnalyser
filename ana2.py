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

    #recomb = "06417"
    recomb = "0715"
    energyList = [1,2,3,4,5,6,7] 
    variables = ['dEdx','aarondEdx','energy','hitCorrection','mcDepCorrection','missedHitsCorrection','noHitsCorrection','energyCorrected','mcInitEnergy','recoMinusTrueOverTrue','mcIDEdiscrep']
        
#    rfile2 = up.open( "PDSPProd2_1GeV.root" )
#    df2    = rfile2["pdAnaTree/AnaTree"].pandas.df( variables )
#    
    fig = plt.figure(figsize=(26,14))
    plotCount = 1

    for energy in energyList:
        print("{} GeV".format(str(energy)))
        rfile      = up.open( "singlePositron_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb ) )
        rfile_phot = up.open( "singlePhoton_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb )   )
        
        df      = rfile["pdAnaTree/AnaTree"].pandas.df( variables )
        df_phot = rfile_phot["pdAnaTree/AnaTree"].pandas.df( variables )
        

        peakVal      = peakValue( np.histogram(df.mcDepCorrection/df.mcInitEnergy           *100, range=(0,5), bins=50) )
        peakVal_phot = peakValue( np.histogram(df_phot.mcDepCorrection/df_phot.mcInitEnergy *100, range=(0,5), bins=50) )

        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.mcDepCorrection/df.mcInitEnergy           *100, range=(0,5), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        plt.hist( df_phot.mcDepCorrection/df_phot.mcInitEnergy *100, range=(0,5), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal = peakValue( np.histogram(df2.mcDepCorrection/df2.mcInitEnergy *100, range=(0,5), bins=50) )
        #    plt.hist( df2.mcDepCorrection/df2.mcInitEnergy *100, range=(0,5), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        #    plt.axvline(peakVal, c='g', ls='--', lw=0.5)
            
        plt.xlabel('MC Deposition Correction (% of MC Initial Energy)',fontweight="bold")
        plt.ylabel('{} GeV'.format(str(energy)),fontweight="bold")
        plt.legend()
        

        peakVal      = peakValue( np.histogram(df.missedHitsCorrection/df.mcInitEnergy           *100, range=(0,15), bins=50, density=True) )
        peakVal_phot = peakValue( np.histogram(df_phot.missedHitsCorrection/df_phot.mcInitEnergy *100, range=(0,15), bins=50, density=True) )
        
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.missedHitsCorrection/df.mcInitEnergy           *100, range=(0,15), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        plt.hist( df_phot.missedHitsCorrection/df_phot.mcInitEnergy *100, range=(0,15), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal = peakValue( np.histogram(df2.missedHitsCorrection/df2.mcInitEnergy *100, range=(0,15), bins=50) )
        #    plt.hist( df2.missedHitsCorrection/df2.mcInitEnergy *100, range=(0,15), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        #    plt.axvline(peakVal, c='g', ls='--', lw=0.5)
        
        plt.legend()
        plt.xlabel('Missed Hits Correction (% of MC Initial Energy)',fontweight="bold")
        

        peakVal      = peakValue( np.histogram(df.noHitsCorrection/df.mcInitEnergy           *100, range=(10,30), bins=50) )
        peakVal_phot = peakValue( np.histogram(df_phot.noHitsCorrection/df_phot.mcInitEnergy *100, range=(10,30), bins=50) )
       
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.noHitsCorrection/df.mcInitEnergy           *100, range=(10,30), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        plt.hist( df_phot.noHitsCorrection/df_phot.mcInitEnergy *100, range=(10,30), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal = peakValue( np.histogram(df2.noHitsCorrection/df2.mcInitEnergy *100, range=(10,30), bins=50) )
        #    plt.hist( df2.noHitsCorrection/df2.mcInitEnergy *100, range=(10,30), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        #    plt.axvline(peakVal, c='g', ls='--', lw=0.5)
        
        plt.legend()
        plt.xlabel('No Hits Correction (% of MC Initial Energy)',fontweight="bold")
        

        peakVal      = peakValue( np.histogram(df.hitCorrection/df.mcInitEnergy           *100, range=(-10,10), bins=50) )
        peakVal_phot = peakValue( np.histogram(df_phot.hitCorrection/df_phot.mcInitEnergy *100, range=(-10,10), bins=50) )
        
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.hitCorrection/df.mcInitEnergy           *100, range=(-10,10), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        plt.hist( df_phot.hitCorrection/df_phot.mcInitEnergy *100, range=(-10,10), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak {}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal = peakValue( np.histogram(df2.hitCorrection/df2.mcInitEnergy *100, range=(-10,10), bins=50) )
        #    plt.hist( df2.hitCorrection/df2.mcInitEnergy *100, range=(-10,10), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        #    plt.axvline(peakVal, c='g', ls='--', lw=0.5)
        
        plt.legend()
        plt.xlabel('Hit Correction (% of MC Initial Energy)',fontweight="bold")
       

        peakVal      = peakValue( np.histogram(df.mcIDEdiscrep/df.mcInitEnergy           *100, range=(-10,10), bins=50) )
        peakVal_phot = peakValue( np.histogram(df_phot.mcIDEdiscrep/df_phot.mcInitEnergy *100, range=(-10,10), bins=50) )
        
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.mcIDEdiscrep/df.mcInitEnergy           *100, range=(-10,10), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        plt.hist( df_phot.mcIDEdiscrep/df_phot.mcInitEnergy *100, range=(-10,10), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal = peakValue( np.histogram(df2.mcIDEdiscrep/df2.mcInitEnergy *100, range=(-10,10), bins=50) )
        #    plt.hist( df2.mcIDEdiscrep/df2.mcInitEnergy *100, range=(-10,10), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        #    plt.axvline(peakVal, c='g', ls='--', lw=0.5)
        
        plt.legend()
        plt.xlabel('MC IDE Discrepancy (% of MC Initial Energy)',fontweight="bold")
        
        
        peakVal      = peakValue( np.histogram((df.mcIDEdiscrep+df.hitCorrection+df.missedHitsCorrection+df.mcDepCorrection+df.noHitsCorrection)/df.mcInitEnergy                               *100, range=(0,50), bins=50) )
        peakVal_phot = peakValue( np.histogram((df_phot.mcIDEdiscrep+df_phot.hitCorrection+df_phot.missedHitsCorrection+df_phot.mcDepCorrection+df_phot.noHitsCorrection)/df_phot.mcInitEnergy *100, range=(0,50), bins=50) )
       
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( (df.mcIDEdiscrep+df.hitCorrection+df.missedHitsCorrection+df.mcDepCorrection+df.noHitsCorrection)/df.mcInitEnergy                               *100, range=(0,50), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak Value={}'.format(round(peakVal,3))      )
        plt.hist( (df_phot.mcIDEdiscrep+df_phot.hitCorrection+df_phot.missedHitsCorrection+df_phot.mcDepCorrection+df_phot.noHitsCorrection)/df_phot.mcInitEnergy *100, range=(0,50), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$:   Peak Value={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal = peakValue( np.histogram((df2.mcIDEdiscrep+df2.hitCorrection+df2.missedHitsCorrection+df2.mcDepCorrection+df2.noHitsCorrection)/df2.mcInitEnergy *100, range=(0,50), bins=50) )
        #    plt.hist( (df2.mcIDEdiscrep+df2.hitCorrection+df2.missedHitsCorrection+df2.mcDepCorrection+df2.noHitsCorrection)/df2.mcInitEnergy *100, range=(0,50), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))      )
        #    plt.axvline(peakVal, c='g', ls='--', lw=0.5)
        
        plt.legend()
        plt.xlabel('All Corrections (% of MC Initial Energy)',fontweight="bold")
      
    plt.subplots_adjust(left=0.02, bottom=0.04, right=0.98, top=0.97, wspace=0.13, hspace=0.18)
    plt.show()


# =======================================================================================================================================================================================================================================================
   
  
    fig = plt.figure(figsize=(26,14))
    plotCount = 1
    for energy in energyList:
        print("{} GeV".format(str(energy)))
        rfile      = up.open( "singleElectron_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb ) )
        rfile_phot = up.open( "singlePhoton_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb )   )
        
        df      = rfile["pdAnaTree/AnaTree"].pandas.df( variables )
        df_phot = rfile_phot["pdAnaTree/AnaTree"].pandas.df( variables )

        plt.subplot(len(energyList),4,plotCount)
        plotCount += 1
        edEdx, ebins = uf.normHistToUnity( np.histogram( df.dEdx,      range=(0,10), bins=50 ) )
        pdEdx, pbins = uf.normHistToUnity( np.histogram( df_phot.dEdx, range=(0,10), bins=50 ) )
        plt.hist( ebins[:-1], ebins, weights=edEdx, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$' )
        plt.hist( pbins[:-1], pbins, weights=pdEdx, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$' )
        plt.axvline(2.1, c='r', lw=1, ls='--')
        plt.axvline(4.2, c='b', lw=1, ls='--')
        #if energy == 1:
        #    dEdx, bins = uf.normHistToUnity( np.histogram( df2.dEdx, range=(0,10), bins=50 ) )
        #    plt.hist( bins[:-1], bins, weights=dEdx, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$' )

        plt.xlabel('dE/dx (MeV/cm)',fontweight="bold")
        plt.ylabel('{} GeV'.format(str(energy)),fontweight="bold")
        plt.legend()
        
        plt.subplot(len(energyList),4,plotCount)
        plotCount += 1
        plt.hist( df.energyCorrected,      range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df.energyCorrected),3))           )
        plt.hist( df_phot.energyCorrected, range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$:   $\mu$={}'.format(round(np.mean(df_phot.energyCorrected),3)) )
        #if energy == 1:
        #    plt.hist( df2.energyCorrected, range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df2.energyCorrected),3))      )
        plt.xlabel('Energy (GeV)',fontweight="bold")
        plt.legend()

        plt.subplot(len(energyList),4,plotCount)
        plotCount += 1
        plt.hist( df.mcInitEnergy,      range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df.mcInitEnergy),3))      )
        plt.hist( df_phot.mcInitEnergy, range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$:   $\mu$={}'.format(round(np.mean(df_phot.mcInitEnergy),3)) )
        #if energy == 1:
        #    plt.hist( df2.mcInitEnergy, range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df2.mcInitEnergy),3))      )
        plt.xlabel('Initial MC Energy (GeV)',fontweight="bold")
        plt.legend()
        
        plt.subplot(len(energyList),4,plotCount)
        plotCount += 1
        plt.hist( df.recoMinusTrueOverTrue,      range=(-0.025,0.025), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df.recoMinusTrueOverTrue),3))           )
        plt.hist( df_phot.recoMinusTrueOverTrue, range=(-0.025,0.025), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$:   $\mu$={}'.format(round(np.mean(df_phot.recoMinusTrueOverTrue),3)) )
        #if energy == 1:
        #    plt.hist( df2.recoMinusTrueOverTrue, range=(-0.025,0.025), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df2.recoMinusTrueOverTrue),3))           )
        plt.xlabel('(Reco - True)/True',fontweight="bold")
        plt.legend()

    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.97, wspace=0.13, hspace=0.18)
    plt.show()


# =======================================================================================================================================================================================================================================================


    corrPeaks      = {}
    corrPeaks_phot = {}
    fig = plt.figure(figsize=(26,14))
    plotCount = 1
    for energy in energyList:
        print("{} GeV".format(str(energy)))
        rfile      = up.open( "singleElectron_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb ) )
        rfile_phot = up.open( "singlePhoton_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb )   )
        
        df      = rfile["pdAnaTree/AnaTree"].pandas.df( variables )
        df_phot = rfile_phot["pdAnaTree/AnaTree"].pandas.df( variables )
        

        peakVal      = peakValue( np.histogram(df.mcDepCorrection,      range=(np.mean(df.mcDepCorrection)-(4*np.std(df.mcDepCorrection)),np.mean(df.mcDepCorrection)+(4*np.std(df.mcDepCorrection))), bins=30) )
        peakVal_phot = peakValue( np.histogram(df_phot.mcDepCorrection, range=(np.mean(df.mcDepCorrection)-(4*np.std(df.mcDepCorrection)),np.mean(df.mcDepCorrection)+(4*np.std(df.mcDepCorrection))), bins=30) )

        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.mcDepCorrection,      range=(np.mean(df.mcDepCorrection)-(4*np.std(df.mcDepCorrection)),np.mean(df.mcDepCorrection)+(4*np.std(df.mcDepCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        plt.hist( df_phot.mcDepCorrection, range=(np.mean(df.mcDepCorrection)-(4*np.std(df.mcDepCorrection)),np.mean(df.mcDepCorrection)+(4*np.std(df.mcDepCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal      = peakValue( np.histogram(df2.mcDepCorrection,      range=(np.mean(df.mcDepCorrection)-(4*np.std(df.mcDepCorrection)),np.mean(df.mcDepCorrection)+(4*np.std(df.mcDepCorrection))), bins=30) )
        #    plt.hist( df2.mcDepCorrection,      range=(np.mean(df.mcDepCorrection)-(4*np.std(df.mcDepCorrection)),np.mean(df.mcDepCorrection)+(4*np.std(df.mcDepCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        #    plt.axvline(peakVal,      c='g', ls='--', lw=0.5)
        plt.legend()
        plt.xlabel('MC Deposition Correction (GeV)',fontweight="bold")
        plt.ylabel('{} GeV'.format(str(energy)),fontweight="bold")
        plt.legend()


        peakVal      = peakValue( np.histogram(df.missedHitsCorrection,      range=(np.mean(df.missedHitsCorrection)-(4*np.std(df.missedHitsCorrection)),np.mean(df.missedHitsCorrection)+(4*np.std(df.missedHitsCorrection))), bins=30) )
        peakVal_phot = peakValue( np.histogram(df_phot.missedHitsCorrection, range=(np.mean(df.missedHitsCorrection)-(4*np.std(df.missedHitsCorrection)),np.mean(df.missedHitsCorrection)+(4*np.std(df.missedHitsCorrection))), bins=30) )
        
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.missedHitsCorrection,      range=(np.mean(df.missedHitsCorrection)-(4*np.std(df.missedHitsCorrection)),np.mean(df.missedHitsCorrection)+(4*np.std(df.missedHitsCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        plt.hist( df_phot.missedHitsCorrection, range=(np.mean(df.missedHitsCorrection)-(4*np.std(df.missedHitsCorrection)),np.mean(df.missedHitsCorrection)+(4*np.std(df.missedHitsCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal      = peakValue( np.histogram(df2.missedHitsCorrection,      range=(np.mean(df.missedHitsCorrection)-(4*np.std(df.missedHitsCorrection)),np.mean(df.missedHitsCorrection)+(4*np.std(df.missedHitsCorrection))), bins=30) )
        #    plt.hist( df2.missedHitsCorrection,      range=(np.mean(df.missedHitsCorrection)-(4*np.std(df.missedHitsCorrection)),np.mean(df.missedHitsCorrection)+(4*np.std(df.missedHitsCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        #    plt.axvline(peakVal,      c='g', ls='--', lw=0.5)
        plt.legend()
        plt.xlabel('Missed Hits Correction (GeV)',fontweight="bold")


        peakVal      = peakValue( np.histogram(df.noHitsCorrection,      range=(np.mean(df.noHitsCorrection)-(4*np.std(df.noHitsCorrection)),np.mean(df.noHitsCorrection)+(4*np.std(df.noHitsCorrection))), bins=30) )
        peakVal_phot = peakValue( np.histogram(df_phot.noHitsCorrection, range=(np.mean(df.noHitsCorrection)-(4*np.std(df.noHitsCorrection)),np.mean(df.noHitsCorrection)+(4*np.std(df.noHitsCorrection))), bins=30) )
       
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.noHitsCorrection,      range=(np.mean(df.noHitsCorrection)-(4*np.std(df.noHitsCorrection)),np.mean(df.noHitsCorrection)+(4*np.std(df.noHitsCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        plt.hist( df_phot.noHitsCorrection, range=(np.mean(df.noHitsCorrection)-(4*np.std(df.noHitsCorrection)),np.mean(df.noHitsCorrection)+(4*np.std(df.noHitsCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal      = peakValue( np.histogram(df2.noHitsCorrection,      range=(np.mean(df.noHitsCorrection)-(4*np.std(df.noHitsCorrection)),np.mean(df.noHitsCorrection)+(4*np.std(df.noHitsCorrection))), bins=30) )
        #    plt.hist( df2.noHitsCorrection,      range=(np.mean(df.noHitsCorrection)-(4*np.std(df.noHitsCorrection)),np.mean(df.noHitsCorrection)+(4*np.std(df.noHitsCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        #    plt.axvline(peakVal,      c='g', ls='--', lw=0.5)
        plt.legend()
        plt.xlabel('No Hits Correction (GeV)',fontweight="bold")


        peakVal      = peakValue( np.histogram(df.hitCorrection,      range=(np.mean(df.hitCorrection)-(4*np.std(df.hitCorrection)),np.mean(df.hitCorrection)+(4*np.std(df.hitCorrection))), bins=30) )
        peakVal_phot = peakValue( np.histogram(df_phot.hitCorrection, range=(np.mean(df.hitCorrection)-(4*np.std(df.hitCorrection)),np.mean(df.hitCorrection)+(4*np.std(df.hitCorrection))), bins=30) )
        
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.hitCorrection,      range=(np.mean(df.hitCorrection)-(4*np.std(df.hitCorrection)),np.mean(df.hitCorrection)+(4*np.std(df.hitCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        plt.hist( df_phot.hitCorrection, range=(np.mean(df.hitCorrection)-(4*np.std(df.hitCorrection)),np.mean(df.hitCorrection)+(4*np.std(df.hitCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal      = peakValue( np.histogram(df2.hitCorrection,      range=(np.mean(df.hitCorrection)-(4*np.std(df.hitCorrection)),np.mean(df.hitCorrection)+(4*np.std(df.hitCorrection))), bins=30) )
        #    plt.hist( df2.hitCorrection,      range=(np.mean(df.hitCorrection)-(4*np.std(df.hitCorrection)),np.mean(df.hitCorrection)+(4*np.std(df.hitCorrection))), bins=30, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        #    plt.axvline(peakVal,      c='g', ls='--', lw=0.5)
        plt.legend()
        plt.xlabel('Hit Correction (GeV)',fontweight="bold")


        peakVal      = peakValue( np.histogram(df.mcIDEdiscrep,      range=(np.mean(df.mcIDEdiscrep)-(4*np.std(df.mcIDEdiscrep)),np.mean(df.mcIDEdiscrep)+(4*np.std(df.mcIDEdiscrep))), bins=30) )
        peakVal_phot = peakValue( np.histogram(df_phot.mcIDEdiscrep, range=(np.mean(df.mcIDEdiscrep)-(4*np.std(df.mcIDEdiscrep)),np.mean(df.mcIDEdiscrep)+(4*np.std(df.mcIDEdiscrep))), bins=30) )
        
        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( df.mcIDEdiscrep,      range=(np.mean(df.mcIDEdiscrep)-(4*np.std(df.mcIDEdiscrep)),np.mean(df.mcIDEdiscrep)+(4*np.std(df.mcIDEdiscrep))), bins=30, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        plt.hist( df_phot.mcIDEdiscrep, range=(np.mean(df.mcIDEdiscrep)-(4*np.std(df.mcIDEdiscrep)),np.mean(df.mcIDEdiscrep)+(4*np.std(df.mcIDEdiscrep))), bins=30, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    peakVal      = peakValue( np.histogram(df2.mcIDEdiscrep,      range=(np.mean(df.mcIDEdiscrep)-(4*np.std(df.mcIDEdiscrep)),np.mean(df.mcIDEdiscrep)+(4*np.std(df.mcIDEdiscrep))), bins=30) )
        #    plt.hist( df2.mcIDEdiscrep,      range=(np.mean(df.mcIDEdiscrep)-(4*np.std(df.mcIDEdiscrep)),np.mean(df.mcIDEdiscrep)+(4*np.std(df.mcIDEdiscrep))), bins=30, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        #    plt.axvline(peakVal,      c='g', ls='--', lw=0.5)
        plt.legend()
        plt.xlabel('MC IDE Discrepancy (GeV)',fontweight="bold")


        allCorr      = df.mcIDEdiscrep+df.hitCorrection+df.missedHitsCorrection+df.mcDepCorrection+df.noHitsCorrection
        allCorr_phot = df_phot.mcIDEdiscrep+df_phot.hitCorrection+df_phot.missedHitsCorrection+df_phot.mcDepCorrection+df_phot.noHitsCorrection

        peakVal      = peakValue( np.histogram(allCorr, range=(np.mean(allCorr)-(3*np.std(allCorr)),np.mean(allCorr)+(3*np.std(allCorr))), bins=30) )
        corrPeaks[energy] = peakVal
        peakVal_phot = peakValue( np.histogram(allCorr, range=(np.mean(allCorr)-(3*np.std(allCorr)),np.mean(allCorr)+(3*np.std(allCorr))), bins=30) )
        corrPeaks_phot[energy] = peakVal_phot

        plt.subplot(len(energyList),6,plotCount)
        plotCount += 1
        plt.hist( allCorr,      range=(np.mean(allCorr)-(3*np.std(allCorr)),np.mean(allCorr)+(3*np.std(allCorr))), bins=30, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        plt.hist( allCorr_phot, range=(np.mean(allCorr)-(3*np.std(allCorr)),np.mean(allCorr)+(3*np.std(allCorr))), bins=30, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$: Peak={}'.format(round(peakVal_phot,3)) )
        plt.axvline(peakVal,      c='r', ls='--', lw=0.5)
        plt.axvline(peakVal_phot, c='b', ls='--', lw=0.5)
        #if energy == 1:
        #    allCorr2      = df2.mcIDEdiscrep+df2.hitCorrection+df2.missedHitsCorrection+df2.mcDepCorrection+df2.noHitsCorrection
        #    peakVal      = peakValue( np.histogram(allCorr2, range=(np.mean(allCorr)-(3*np.std(allCorr)),np.mean(allCorr)+(3*np.std(allCorr))), bins=30) )
        #    plt.hist( allCorr2,      range=(np.mean(allCorr)-(3*np.std(allCorr)),np.mean(allCorr)+(3*np.std(allCorr))), bins=30, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: Peak={}'.format(round(peakVal,3))         )
        #    plt.axvline(peakVal,      c='g', ls='--', lw=0.5)
        plt.legend()
        plt.xlabel('All Corrections (GeV)',fontweight="bold")
        
    plt.subplots_adjust(left=0.03, bottom=0.04, right=0.98, top=0.97, wspace=0.13, hspace=0.18)
    plt.show()
    
    fig = plt.figure(figsize=(26,14))
    plotCount = 1
    for energy in energyList:
        print("{} GeV".format(str(energy)))
        rfile      = up.open( "singleElectron_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb ) )
        rfile_phot = up.open( "singlePhoton_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb )   )
        
        df      = rfile["pdAnaTree/AnaTree"].pandas.df( variables )
        df_phot = rfile_phot["pdAnaTree/AnaTree"].pandas.df( variables )
        
        plt.subplot(2,len(energyList),plotCount)
        plt.hist( df.energy + corrPeaks[energy],      range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df.energy + corrPeaks[energy]),3))           )
        plt.hist( df_phot.energy + corrPeaks_phot[energy], range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$:   $\mu$={}'.format(round(np.mean(df_phot.energy + corrPeaks_phot[energy]),3)) )
        #if energy == 1:
        #    plt.hist( df2.energyCorrected, range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df2.energyCorrected),3))      )
        plt.xlabel('Energy (GeV)',fontweight="bold")
        plt.legend()
        
        plt.subplot(2,len(energyList),plotCount+len(energyList))
        plotCount += 1
        plt.hist( ((df.energy + corrPeaks[energy])-df.mcInitEnergy)/df.mcInitEnergy, range=(-1,+1), bins=50, density=True, histtype='stepfilled', edgecolor='r', fc=(1,0,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(((df.energy + corrPeaks[energy])-df.mcInitEnergy)/df.mcInitEnergy),3))      )
        plt.hist( ((df_phot.energy + corrPeaks_phot[energy])-df.mcInitEnergy)/df.mcInitEnergy, range=(-1,+1), bins=50, density=True, histtype='stepfilled', edgecolor='b', fc=(0,0,1,0.1), label=r'$\gamma$:   $\mu$={}'.format(round(np.mean(((df_phot.energy + corrPeaks_phot[energy])-df.mcInitEnergy)/df.mcInitEnergy),3)) )
        #if energy == 1:
        #    plt.hist( df2.energyCorrected, range=(energy-1,energy+1), bins=50, density=True, histtype='stepfilled', edgecolor='g', fc=(0,1,0,0.1), label=r'$e^-$: $\mu$={}'.format(round(np.mean(df2.energyCorrected),3))      )
        plt.xlabel('Energy (GeV)',fontweight="bold")
        plt.legend()

    plt.show()

    recoMean = []
    recoErr = []
    trueMean = []
    trueErr = []
    
    for energy in energyList:
        print("{} GeV".format(str(energy)))
        rfile      = up.open( "singleElectron_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb ) )
        rfile_phot = up.open( "singlePhoton_{}GeV_sceON_keepOFF_recomb{}.root".format( energy, recomb )   )
        
        df      = rfile["pdAnaTree/AnaTree"].pandas.df( variables )
        df_phot = rfile_phot["pdAnaTree/AnaTree"].pandas.df( variables )

        energyAveCorr      = df.energy      + corrPeaks[energy]
        energyAveCorr_phot = df_phot.energy + corrPeaks_phot[energy]
        print("Reco")
        print("Electron: {}GeV | rmse: {}".format(energy, round(np.sqrt(np.mean((energyAveCorr-df.mcInitEnergy)**2)),3)))
        print("Electron: {}GeV | std:  {}".format(energy, round(np.std(energyAveCorr),3)))
        print("Photon:   {}GeV | rmse: {}".format(energy, round(np.sqrt(np.mean((energyAveCorr_phot-df_phot.mcInitEnergy)**2)),3)))
        print("Photon:   {}GeV | std:  {}".format(energy, round(np.std(energyAveCorr_phot),3)))

        print()
        print("True")
        print("Electron: {}GeV | rmse: {}".format(energy, round(np.sqrt(np.mean((df.mcInitEnergy-energy)**2)),3)))
        print("Electron: {}GeV | std:  {}".format(energy, round(np.std(df.mcInitEnergy),3) ) )
        print("Photon:   {}GeV | rmse: {}".format(energy, round(np.sqrt(np.mean((df_phot.mcInitEnergy-energy)**2)),3)))
        print("Photon:   {}GeV | std:  {}".format(energy, round(np.std(df_phot.mcInitEnergy),3) ) )
        print()

        recoMean.append(np.mean(energyAveCorr))
        trueMean.append(np.mean(df.mcInitEnergy))

        trueErr.append(np.std(df.mcInitEnergy))
        recoErr.append(np.std(energyAveCorr))

    print(recoErr)
    print(trueErr)
    plt.errorbar( trueMean, recoMean, yerr=recoErr, xerr=trueErr, fmt="o" )
    plt.plot(np.unique(trueMean), np.poly1d(np.polyfit(trueMean, recoMean, 1))(np.unique(trueMean)))
    plt.show()
        

# ===== /Main Program/ =====

