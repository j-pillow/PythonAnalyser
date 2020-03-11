# ===== Import Modules =====

import numpy as np
import pandas as pd
import uproot as up
import matplotlib.pyplot as plt
import seaborn as sns
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire

# ===== /Import Modules/ =====

# ===== Define Functions =====

# ===== /Define Functions/ =====

# ===== Main Program =====

if __name__ == '__main__':

    rfile = up.open("singleElectron_7GeV_sceON_keepOFF.root")
    df    = rfile["pdAnaTree/AnaTree"].pandas.df( ['hitCharge','correctedHit3DX','correctedHit3DY','correctedHit3DZ','nonCorrectedHit3DX','nonCorrectedHit3DY','nonCorrectedHit3DZ'],flatten=True, entrystart=3, entrystop=4 )

#    df_temp = pd.DataFrame({ 'hitCharge':[], 'correctedHit3DX':[], 'correctedHit3DY':[], 'correctedHit3DZ':[] })
#    for i in range(len(df.iloc[0].correctedHit3DX)):
#        df_temp = df_temp.append( pd.DataFrame({'hitCharge':[df.iloc[0].hitCharge[i]], 'correctedHit3DX':[df.iloc[0].correctedHit3DX[i]], 'correctedHit3DY':[df.iloc[0].correctedHit3DY[i]], 'correctedHit3DZ':[df.iloc[0].correctedHit3DZ[i]]}), ignore_index=True )
#
#    df = df_temp

    cvs = ds.Canvas( plot_width=1000, plot_height=800 )
    agg = cvs.points(df, 'correctedHit3DZ', 'correctedHit3DY', ds.mean('hitCharge'))
    img = tf.shade(agg,cmap=fire,how='log')
    img.to_pil().save("test_yz.png")
    cvs = ds.Canvas( plot_width=1000, plot_height=800 )
    agg = cvs.points(df, 'correctedHit3DZ', 'correctedHit3DX', ds.mean('hitCharge'))
    img = tf.shade(agg,cmap=fire,how='log')
    img.to_pil().save("test_xz.png")

# ===== /Main Program/ =====

