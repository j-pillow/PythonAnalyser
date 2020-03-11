# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===== /Import Modules/ =====

# ===== Define Functions =====

# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':

    variables = ['hitCharge','hitTrueEnergy']
    rfile = up.open( "PDSPProd2_1GeV.root" )
    tree = up.open( "PDSPProd2_1GeV.root" )["pdAnaTree/AnaTree"]
    #df = rfile["pdAnaTree/AnaTree"].pandas.df( variables, flatten=False )

    dEdx = []
    for df in tqdm(up.pandas.iterate("PDSPProd2_3GeV.root","pdAnaTree/AnaTree",["hitCharge","hitTrueEnergy"],flatten=False,entrysteps=500),total=tree.numentries/500):
        
        for index, row in tqdm(df.iterrows(),total=len(df)):
            dEdx += row.dEdx

    print(len(hitCharge))
    plt.hist2d( hitCharge, hitEnergy, bins=200, range=[[0,3000],[0,20]], cmin=100 )
    plt.colorbar()
    plt.show()

#    plt.hist(df.hitCharge)
#    plt.show()

# ===== /Main Program/ =====

