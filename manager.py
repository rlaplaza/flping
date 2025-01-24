#!/usr/bin/env python3
import ase
import sys
from ase.io import read, write
from ase.build import molecule
from ase.geometry.analysis import Analysis as ana
import ase.neighborlist
import numpy as np
from skspatial.objects import Plane, Points

filenames = []
outnames = []
structures = []
if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:]):
        if "chromosome_0_1.0" in arg:
            filenames.append(arg)

for i in filenames:
    term = i.strip()[-3:]
    structure = read(i, format=term)
    structures.append(structure)
    outname_neutral = i.replace("chromosome_0_1.0", "neutral")
    outname_proton = i.replace("chromosome_0_1.0", "with_proton")
    outname_hydride = i.replace("chromosome_0_1.0", "with_hydride")
    # outnames.append(outname_hydride)
    outnames.append(outname_proton)


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


for i, outname in zip(structures, outnames):
    print(f"\nProceeding to fix structure {i}")
    i.rattle(stdev=0.01)
    analysis = ana(i)
    analysis.clear_cache()
    crd_b1 = None
    crd_n1 = None
    refdist = 10

    crd_ns = []
    n_ns = []
    for n, j in enumerate(i.get_chemical_symbols()):
        if j == "B" and crd_b1 is None:
            crd_b1 = i.get_positions()[n]
            n_b1 = n
        if j == "N" and crd_n1 is None:
            crd_n1 = i.get_positions()[n]
            n_n1 = n
        if j == "N" and crd_n1 is not None:
            crd_ns.append(i.get_positions()[n])
            n_ns.append(n)
    if crd_ns is not None:
        refdist = np.linalg.norm(crd_b1 - crd_n1)
        for crd_n, n in zip(crd_ns, n_ns):
            newdist = np.linalg.norm(crd_n - crd_b1)
            if (
                newdist > refdist
            ):  # IMPORTANT! Adjust as needed, in this case the pyrazole is tricky
                crd_n1 = crd_n
                n_n1 = n
                refdist = newdist

    cutoff = ase.neighborlist.natural_cutoffs(i, mult=1.2)
    nl = ase.neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(i)
    cm = nl.get_connectivity_matrix(sparse=False)

    # Find oepn coordination sites attached to B and N
    sub_b1 = []
    sub_n1 = []
    for n, j in enumerate(cm[n_b1, :]):
        # print(n, j)
        if j and n != n_b1:
            sub_b1.append(i.get_positions()[n])
    for n, j in enumerate(cm[n_n1, :]):
        if j and n != n_n1:
            sub_n1.append(i.get_positions()[n])
    # For B
    subs_b1 = np.array(sub_b1)
    # print(subs_b1, sub_b1)
    c_b1 = np.mean(subs_b1, axis=0)
    # print(n_b1, c_b1)
    v_b1 = c_b1 - crd_b1
    d_b1 = np.linalg.norm(v_b1)
    v_b1 = unit_vector(v_b1)

    # For N
    subs_n1 = np.array(sub_n1)
    # print(subs_n1, sub_n1)
    c_n1 = np.mean(subs_n1, axis=0)
    # print(n_n1, c_n1)
    v_n1 = c_n1 - crd_n1
    d_n1 = np.linalg.norm(v_n1)
    v_n1 = unit_vector(v_n1)

    # Handle structures and add Hs
    print(f"Vectors are {v_b1} of distance {d_b1} and {v_n1} of distance {d_n1}")
    crd_h_n1 = crd_n1 - (1.1 * v_n1)
    if "with_proton" in outname:
        h = molecule("H")
        probe = molecule("Si")
        h.set_positions(np.expand_dims(crd_h_n1, 0))
        probe.set_positions(np.expand_dims(c_n1, 0))
        i = i + h  # + probe
    if "with_hydride" in outname:
        h = molecule("H")
        probe = molecule("Si")
        crds_h_b1 = np.array([crd_b1 - (1.1 * v_b1), crd_b1 + (1.1 * v_b1)])
        dists = np.array(
            [np.linalg.norm(crd_h_b1 - crd_h_n1) for crd_h_b1 in crds_h_b1]
        )
        crd_h_b1 = crds_h_b1[np.argmin(dists)]
        h.set_positions(np.expand_dims(crd_h_b1, 0))
        probe.set_positions(np.expand_dims(c_b1, 0))
        i = i + h  # + probe
    write(outname, i)
