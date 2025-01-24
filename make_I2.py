#!/usr/bin/env python

import os, fnmatch
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import sklearn
import math
from scipy import stats
from itertools import chain
import sklearn as sk
from sklearn.metrics import r2_score
from glob import glob
import matplotlib.colors as mcolors
from matplotlib.ticker import NullFormatter

import ase
import sys
from ase.io import read
from ase.build import molecule
from ase.geometry.analysis import Analysis as ana
import ase.neighborlist
from skspatial.objects import Plane, Points
import json
import pandas as pd
import os

# reading the linker names, and positions of the N atoms using pandas
# try_linkers = pd.read_csv("/home/sdas/MOF-FLP-P2/Linkers/benzene_substructure_search/opt_I2/final_bn_ad.out",delimiter=",")
# filenames = try_linkers.iloc[:, 0]
# filenames = [i[5:] for i in filenames]
# npos = try_linkers.iloc[:, 1]
# distance = try_linkers.iloc[:, 2]
# defining the paths
# pathtostr = "./linker_conformers/final/"

filenames = ["0517_BMe2_11_25_conformer4.xyz"]
npos = [25]
distance = [2.49]

pathtostr = "./"
pathtoI2 = "./Gen_I2/"
pathtoI2_selected = "./Gen_I2_selected/"

###grabbing the structures
structures = []
name_list = []
for i in filenames:
    term = i.strip()[-3:]
    # name = i[48:-4]
    name = i[:-4]
    # name = i[:-4]
    structure = read(pathtostr + i, format=term)
    structures.append(structure)
    name_list.append(name)

# functions
def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


def hh_min_rot(symbols, coords_list, rotvec):
    coords_list = np.array(coords_list)
    hhref = np.linalg.norm(coords_list[-1] - coords_list[-2])
    bref = coords_list[-11]
    print(f"Starting hh distance was {hhref}")
    skeleton = coords_list[:-11]
    rotatable_ref = coords_list[-11:-1].T
    rotatable_new = None
    nh = coords_list[-1]
    for i in np.linspace(0, 360, 720):
        rotmat = g_rot_matrix(i, rotvec)
        rotated = np.dot(rotmat, rotatable_ref).T
        displace = np.array(bref - rotated[0]).reshape(1, 3)
        rotated = rotated + displace
        hhdist = np.linalg.norm(nh - rotated[-1])
        if hhdist < hhref and not clash(skeleton, rotated):
            hhref = hhdist
            rotatable_new = rotated
    if rotatable_new is None:
        rotatable_new = rotatable_ref.T
    return np.vstack([skeleton, rotatable_new, nh])


def clash(skeleton, rotated):
    for i in rotated[::-1]:
        for j in skeleton[::-1]:
            d = np.linalg.norm(i - j)
            if d < 1.1:
                return True
    return False


def write_xyz(path, name, natoms, new_symbols, coords):
    f = open(path + name + "_I2" + ".xyz", "w")
    f.write(str(natoms))
    f.write("\n")
    f.write(name + "_I2")
    f.write("\n")
    for a, b in enumerate(coords):
        line = new_symbols[a]
        line = line + "    " + str(b[0]) + "    " + str(b[1]) + "    " + str(b[2])
        print(line)
        f.write(line)
        f.write("\n")
    f.close()


def g_rot_matrix(degree="0.0", axis=np.asarray([1, 1, 1]), verb_lvl=0):
    """
       Return the rotation matrix associated with counterclockwise rotation about
       the given axis by theta radians.
    Parameters
    ----------
    axis
        Axis of the rotation. Array or list.
    degree
        Angle of the rotation in degrees.
    verb_lvl
        Verbosity level integer flag.
    Returns
    -------
    rot
        Rotation matrix to be used.
    """
    try:
        theta = degree * (np.pi / 180)
    except:
        degree = float(degree)
        theta = degree * (np.pi / 180)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    if verb_lvl > 1:
        print("Rotation matrix generated.")
    return rot


####making I2
vec_b_list = []
Hpos_list = []
atoms_list = []
for k, i in enumerate(structures):
    n_atoms = len(i.get_atomic_numbers())
    atoms = i.get_atomic_numbers()
    symbols = i.get_chemical_symbols()
    # print(symbols)
    coords_list = []
    for r, s in enumerate(atoms):
        coords = i.get_positions()[r]
        coords_list.append(coords)
    # print(atoms)
    # print(coords_list)
    cutoff = ase.neighborlist.natural_cutoffs(i, mult=1.0)
    nl = ase.neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(i)
    cm = nl.get_connectivity_matrix(sparse=False)
    # print(n_atoms)
    # print(i.get_chemical_symbols())
    # print(k, filenames[k])
    crd_N = i.get_positions()[npos[k]]
    # print(crd_N)
    subs_base = []
    for n, j in enumerate(cm[npos[k], :]):
        if j == 1:
            # print(i.get_chemical_symbols()[n], n, i.get_positions()[n])
            if i.get_chemical_symbols()[n] != "B":
                pos = i.get_positions()[n]
                pos = np.around(pos, 4)
                subs_base.append(pos)
    if len(subs_base) == 2:
        print("\nsp2 base requires perpendicular plane. Calculating planes.")

        subs_base.append(crd_N)
        points = Points(subs_base)
        try:
            plane = Plane.best_fit(points)
        except:
            try:
                plane = Plane.from_points(subs_base[0], subs_base[1], subs_base[2])
            except ValueError:
                angles_n.append(0.00)
                # status_base.append(1)
                continue
        # vec_b = np.array(crd_N) - plane.point
        vec_b = plane.point - np.array(crd_N)
        # vec_b = plane.normal
        # status_base.append(2)

    elif len(subs_base) == 3:
        print("\nsp3 base. Calculating planes.")
        points = Points(subs_base)
        try:
            plane = Plane.best_fit(points)
        except:
            try:
                plane = Plane.from_points(subs_base[0], subs_base[1], subs_base[2])
            except ValueError:
                angles_n.append(0.00)
                status_base.append(1)
                continue
        vec_b = plane.normal
    # vec_b = np.array(crd_N) - plane.point
    # status_base.append(3)

    else:
        print(
            f"Base substituents for {filenames[k]} are not 2 or 3 but rather {len(subs_base)} Angle set to 0!"
        )
        if len(subs_base) > 3:
            skip = True
            continue
        else:
            skip = True
            # angles_n.append(0.00)
            # status_base.append(1)
            continue
    for n, j in enumerate(i.get_chemical_symbols()):
        if j == "B":
            crd_acid = i.get_positions()[n]
            n_acid = n
            # print(crd_B)
            # d=i.get_distance(npos[k],n)
            # print(d)
            vec_bn = crd_acid - crd_N
            # print(vec)
    subs_acid = []
    # print("\nAtoms attached to acid:")
    for n, j in enumerate(cm[n_acid, :]):
        if j == 1:
            # print(i.get_chemical_symbols()[n], n, i.get_positions()[n])
            if (
                i.get_chemical_symbols()[n] != "H"
                and i.get_chemical_symbols()[n] != "N"
            ):
                pos = i.get_positions()[n]
                pos = np.around(pos, 4)
                subs_acid.append(pos)
                if n in range(len(i.get_positions()) - 11):
                    rotvec = crd_acid - i.get_positions()[n]
                    bhdist = np.linalg.norm(rotvec)
                    rotvec = unit_vector(rotvec)
                    print(
                        f"Found rotation vector {rotvec} which corresponded to norm {bhdist}"
                    )

    if len(subs_acid) == 3:
        print("\nsp3 acid. Calculating planes.")
        points = Points(subs_acid)
        try:
            plane = Plane.best_fit(points)
        except:
            plane = Plane.from_points(subs_acid[0], subs_acid[1], subs_acid[2])
        vec_a_1 = plane.normal
        vec_a_2 = -vec_a_1
        if angle_between(vec_a_1, vec_bn) > angle_between(vec_a_2, vec_bn):
            vec_a = vec_a_1
        else:
            vec_a = vec_a_2
    else:
        print("Acid substituents are not 3!!! Angle set to 0!")
        # vec_a = plane.normal
        angles_n.append(0.00)

    vec = unit_vector(vec_bn)
    vec_a = unit_vector(vec_a)
    vec_b = unit_vector(vec_b)

    if distance[k] < 3.5:
        Hpos_B = crd_acid + 1.4 * vec_a  ##attaching the proton
        new_atoms_BH = np.append(atoms, 1)
        new_symbols_BH = np.append(symbols, "H")
        new_coords_BH = coords_list.append(Hpos_B)

        Hpos_N = crd_N - 1.2 * vec_b  ##attaching the proton
        new_atoms_BH_NH = np.append(new_atoms_BH, 1)
        new_symbols_BH_NH = np.append(new_symbols_BH, "H")
        coords_list.append(Hpos_N)
        natoms = len(new_atoms_BH_NH)

        write_xyz(
            pathtoI2_selected, name_list[k], natoms, new_symbols_BH_NH, coords_list
        )
        coords_list = hh_min_rot(symbols, coords_list, rotvec)

        write_xyz(
            pathtoI2_selected,
            "rotated" + name_list[k],
            natoms,
            new_symbols_BH_NH,
            coords_list,
        )
