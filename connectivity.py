#!/usr/bin/env python3
import ase
import sys
from ase.io import read
from ase.build import molecule
from ase.geometry.analysis import Analysis as ana
import ase.neighborlist
import numpy as np

filenames = []
structures = []

if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:]):
        filenames.append(arg)

for i in filenames:
    term = i.strip()[-3:]
    structure = read(i, format=term)
    structures.append(structure)


for k, i in enumerate(structures):
    natoms = len(i.get_atomic_numbers())
    cutoff = ase.neighborlist.natural_cutoffs(i, mult=1.2)
    nl = ase.neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(i)
    cm = nl.get_connectivity_matrix(sparse=False)
    d = np.zeros_like(cm)
    for atm in range(natoms):
        d[atm, atm] = len(np.where(cm[atm, :] == 1)[0])
    eigvals, eigvects = np.linalg.eig(d - cm)
    eigvects = eigvects.T
    for eigval,eigvect in zip(eigvals,eigvects):
        print(eigval,eigvect)
    if len(np.where(abs(eigvals) < 1e-10)[0]) >= 2:
        print(f"{filenames[k]} is disconnected.")
