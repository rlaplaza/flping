#!/usr/bin/env python3
import ase
import sys
from ase.io import read
from ase.build import molecule
from ase.geometry.analysis import Analysis as ana
import ase.neighborlist
import numpy as np
from skspatial.objects import Plane, Points

filenames = []
structures = []
distances = []
angles = []
angles_a = []
angles_b = []
if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:]):
        filenames.append(arg)

for i in filenames:
    term = i.strip()[-3:]
    structure = read(i, format=term)
    structures.append(structure)


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


for i in structures:
    print("\nStarting distance and angle calculation:")
    analysis = ana(i)
    analysis.clear_cache()
    crd_acid = None
    crd_base = None
    for n, j in enumerate(i.get_chemical_symbols()):
        if j == "B" and crd_acid is None:
            crd_acid = i.get_positions()[n]
            n_acid = n
        elif j == "N" and crd_base is None:
            crd_base = i.get_positions()[n]
            n_base = n
    distance = np.linalg.norm(crd_acid - crd_base)
    distances.append(distance)
    print("B {0} {1}".format(n_acid, crd_acid))
    print("N {0} {1}".format(n_base, crd_base))
    print(f"Distance is {distance} angstrom.")
    cutoff = ase.neighborlist.natural_cutoffs(i, mult=1.2)
    nl = ase.neighborlist.NeighborList(cutoff, self_interaction=False, bothways=False)
    nl.update(i)
    cm = nl.get_connectivity_matrix(sparse=False)
    subs_acid = []
    subs_acid_potential = []
    subs_base = []
    print("\nAtoms attached to acid:")
    for n, j in enumerate(cm[n_acid, :]):
        if j == 1:
            print(i.get_chemical_symbols()[n], n, i.get_positions()[n])
            if i.get_chemical_symbols()[n] == "H":
                subs_acid_potential.append(i.get_positions()[n])
            elif i.get_chemical_symbols()[n] != "N":
                subs_acid.append(i.get_positions()[n])
    if len(subs_acid_potential) > 1:
        h_dists = []
        for hcoord in subs_acid_potential :
            h_dists.append(np.linalg.norm(hcoord - crd_base))
        acth = np.argmin(np.array(h_dists))
        subs_acid_potential.pop(acth)
        subs_acid.extend(subs_acid_potential)
    print("\nAtoms attached to base:")
    for n, j in enumerate(cm[n_base, :]):
        if j == 1:
            print(i.get_chemical_symbols()[n], n, i.get_positions()[n])
            if (
                i.get_chemical_symbols()[n] != "H"
                and i.get_chemical_symbols()[n] != "B"
            ):
                subs_base.append(i.get_positions()[n])
    if len(subs_base) < 3:
        print("\nsp2 base requires perpendicular plane. Calculating planes.")
        subs_base.append(crd_base)
        points = Points(subs_base)
        plane = Plane.best_fit(points)
        print(points, plane.point)
        vec_b = np.array(crd_base) - plane.point

    elif len(subs_base) == 3:
        print("\nsp3 base. Calculating planes.")
        points = Points(subs_base)
        plane = Plane.best_fit(points)
        print(points, plane.point)
        vec_b = plane.normal

    else:
        print("Base substituents are not 2 or 3!!! Angle set to 0!")
        angles.append(0.00)
        continue

    if len(subs_acid) == 3:
        print("\nsp3 acid. Calculating planes.")
        points = Points(subs_acid)
        plane = Plane.best_fit(points)
        print(points, plane.point)
        vec_a = plane.normal
    else:
        print("Acid substituents are not 3!!! Angle set to 0!")
        angles.append(0.00)
        continue

    angle = angle_between(vec_a, vec_b) * 180 / np.pi
    print(f"Angle is {angle} degrees.")
    angles.append(angle)

    angle_a = angle_between(crd_acid - crd_base, vec_a) * 180 / np.pi
    print(f"Acid angle is {angle_a} degrees.")
    angles_a.append(angle_a)

    angle_b = angle_between(crd_acid - crd_base, vec_b) * 180 / np.pi
    print(f"Base angle is {angle_b} degrees.")
    angles_b.append(angle_b)

f = open("anglesdistances.txt", "w+")
for n, i in enumerate(filenames):
    print(
        filenames[n],
        np.round(distances[n]),
        np.round(angles[n]),
        np.round(angles_a[n]),
        np.round(angles_b[n]),
        file=f,
    )
