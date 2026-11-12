"""
LEGACY - not part of the run order in REPRODUCE.md. Kept as the record of the first submission.
Its inputs are read from the standard folders (paths.py) and its outputs are written to results/legacy/,
so running it never overwrites the outputs of the run order.
Superseded by run_deep.py (step 3).
"""
import sys, json, os
import numpy as np
from deep_loco import train_eval, meta, X, y, folds, FEATS
import paths as P
kind = sys.argv[1]
res,oof,hist = train_eval(kind, save_dir=P.legacy('models'))
np.savez(P.legacy(f'oof_{kind}.npz'), oof=oof)
json.dump({kind:res}, open(P.legacy(f'res_{kind}.json'),'w'), indent=1)
# save minimal history (loss curves) for figures
np.savez(P.legacy(f'hist_{kind}.npz'), **{f'f{fi}_{k}':np.array(v) for fi,hh in hist.items() for k,v in hh.items()})
print(kind, {k:round(v,3) for k,v in res.items()})
