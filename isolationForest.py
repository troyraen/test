import unsup_clust as uc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

f = 'upsilon_features.dat'
# cfeat = 'B-V'
cfeat = 'W3-W4'
df, dfclust, dfclust_feats = uc.load_features(f=f, cfeat=cfeat)


# isolation forest
# dfclust_feats = uc.norm_features(dfclust_feats)
lowtypes = dfclust.loc[dfclust.numinType<100,'newType'].unique()
hitypes = dfclust.loc[dfclust.numinType>1000,'newType'].unique()

dfhi = dfclust_feats.loc[dfclust.newType.isin(hitypes),:]
dflow = dfclust_feats.loc[dfclust.newType.isin(lowtypes),:]

kwargs = {
            # 'n_estimators': 1000,
            'behaviour': 'new',
            # 'max_samples': 1000,
            'random_state': 42,
            'contamination': 'auto',
            'max_features': 3
        }
forest = IsolationForest(**kwargs).fit(dfclust_feats)
predics = forest.predict(dfhi)
plow = forest.predict(dflow)


plt.figure()
plt.hist(predics, label='main sample', alpha=0.5, density=True)
plt.hist(plow, label='outlier classes', alpha=0.5, density=True)
plt.legend()
plt.title('Isolation Forest Results')
plt.show(block=False)
