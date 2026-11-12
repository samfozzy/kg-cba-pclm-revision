"""
Folder layout of the pipeline (added in revision, November 2026).

Every script imports its locations from here, so that it can be run from any working directory:

    data/                 inputs: raw downloads (data/raw/) and the harmonized panels built by build_panel.py
    results/              tables, JSON summaries, out-of-fold estimates of the classical models, pso_state.json
    results/oof_parts/    per-model and per-fold out-of-fold estimates, trained networks (.keras) and scalers
    figures/              figures
    results/legacy/       outputs of the scripts of the first submission that are not in the run order
                          (deep_loco.py run directly, run_arch.py, train_final.py, train_enriched.py,
                          analysis.py, figures_part2.py), kept apart so that they never overwrite the outputs
                          of the run order

Before this file existed, some scripts read and wrote these files relative to the pipeline folder and others
relative to the current working directory, so the run order in REPRODUCE.md did not run end to end.
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(CODE)
DATA = os.path.join(BASE, "data")
RAW = os.path.join(DATA, "raw")
RESULTS = os.path.join(BASE, "results")
OOF = os.path.join(RESULTS, "oof_parts")
FIGURES = os.path.join(BASE, "figures")
LEGACY = os.path.join(RESULTS, "legacy")
PSO_STATE = os.path.join(RESULTS, "pso_state.json")
CONTINENTS = os.path.join(CODE, "continents.csv")

for _d in (DATA, RESULTS, OOF, FIGURES):
    os.makedirs(_d, exist_ok=True)


def data(name):
    return os.path.join(DATA, name)


def results(name):
    return os.path.join(RESULTS, name)


def oof(name):
    return os.path.join(OOF, name)


def figures(name):
    return os.path.join(FIGURES, name)


def legacy(name):
    os.makedirs(LEGACY, exist_ok=True)
    return os.path.join(LEGACY, name)
