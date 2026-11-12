import sys, json, os
import numpy as np
from deep_loco import train_eval, meta, X, y, folds, FEATS
kind = sys.argv[1]
res,oof,hist = train_eval(kind, save_dir='models')
np.savez(f'oof_{kind}.npz', oof=oof)
json.dump({kind:res}, open(f'res_{kind}.json','w'), indent=1)
# save minimal history (loss curves) for figures
np.savez(f'hist_{kind}.npz', **{f'f{fi}_{k}':np.array(v) for fi,hh in hist.items() for k,v in hh.items()})
print(kind, {k:round(v,3) for k,v in res.items()})
