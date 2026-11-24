#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.

import argparse
import os
import json
from functools import partial
from multiprocessing import Pool
import numpy as np
import pyscf.lib as lib

from pyscf import gto, dft, scf

BOHR2ANG = 0.52917721092
Ha_to_eV = 27.2114

def load_json(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} JSON '{path}' not found.")
    with open(path) as f:
        return json.load(f)

def read_xyz_named(path):
    """Read multi‐XYZ file, with molecule name on comment line."""
    atomic_number = {
        'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
        'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
        'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
        'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,
        'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
        'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,
        'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,
        'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,
        'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,
        'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,
        'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,
        'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,'Sg':106,
        'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,'Nh':113,'Fl':114,
        'Mc':115,'Lv':116,'Ts':117,'Og':118
    }
    entries = []
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].isdigit():
            nat  = int(lines[i])
            name = lines[i+1].strip()
            i   += 2
            atoms = []
            for _ in range(nat):
                parts = lines[i].split(); i += 1
                Z = atomic_number[parts[0]]
                xyz = tuple(map(float, parts[1:4]))
                atoms.append((Z, xyz))
            entries.append((name, atoms))
        else:
            i += 1
    return entries

def run_dft(name, atoms, args, spins, charges):
    """Build Mole, run RKS/UKS, optional DF, checkpoint, return mf."""
    mol = gto.Mole(unit='Angstrom')
    mol.atom    = atoms
    mol.basis   = args.basisset
    mol.charge  = charges.get(name, args.charge)
    mol.spin    = spins.get(name, 0)
    mol.verbose = args.verbose
    mol.build()

    # checkpoint file
    chk = os.path.join(args.save_path, f"{args.name_molecule}_{name}.chk")
    os.makedirs(args.save_path, exist_ok=True)
    mol.checkpoint = chk

    if mol.spin > 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc          = args.xc
    mf.grids.level = args.grid_level
    mf.verbose     = args.verbose
    mf.chkfile     = chk

    if args.density_fitting:
        mf = mf.density_fit()

    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel()

    return mf

def analytic_dft_forces(mf):
    """Analytic nuclear gradients → forces (eV/Å)."""
    g_au = mf.nuc_grad_method().kernel()
    return -g_au * (Ha_to_eV / BOHR2ANG)

def dft_energy(atoms, args, spin, charge):
    """Single-point DFT energy for FD."""
    mol = gto.Mole(unit='Angstrom')
    mol.atom    = atoms
    mol.basis   = args.basisset
    mol.charge  = charge
    mol.spin    = spin
    mol.verbose = 0
    mol.build()

    if spin > 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc          = args.xc
    mf.grids.level = args.grid_level
    mf.verbose     = 0

    if args.density_fitting:
        mf = mf.density_fit()

    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel()

    return mf.e_tot

def finite_diff_forces(mf, name, args, spins, charges):
    """Finite‐difference DFT forces."""
    mol         = mf.mol
    coords_bohr = mol.atom_coords()
    coords_ang  = coords_bohr * BOHR2ANG
    nat         = mol.natm
    spin        = mol.spin
    charge      = mol.charge

    forces = np.zeros((nat,3))
    δ      = args.fd_delta
    zs     = [mol.atom_charge(i) for i in range(nat)]

    for i in range(nat):
        for d in range(3):
            plus, minus = [], []
            for k, pos in enumerate(coords_ang):
                p = pos.copy(); m = pos.copy()
                if k == i:
                    p[d] += δ; m[d] -= δ
                plus.append((zs[k], p))
                minus.append((zs[k], m))

            Ep = dft_energy(plus,  args, spin, charge)
            Em = dft_energy(minus, args, spin, charge)
            forces[i,d] = -(Ep - Em)/(2*δ) * Ha_to_eV

    return forces

def save_data(mf, name, args, spins, charges):
    """Save NAO count, structure, energy, analytic & FD forces."""
    out    = args.save_path
    prefix = args.name_molecule
    coords = mf.mol.atom_coords() * BOHR2ANG

    # NAO count
    nao = mf.mol.nao_nr()
    with open(f"{out}/{prefix}_nao.dat","a") as f:
        f.write(f"{nao}\n")

    # structure
    with open(f"{out}/{prefix}_structure.dat","a") as f:
        f.write(" ".join(map(str,coords.flatten()))+"\n")

    # energy
    E_eV = mf.e_tot * Ha_to_eV
    with open(f"{out}/{prefix}_energy.dat","a") as f:
        f.write(f"{E_eV}\n")

    # analytic forces
    fa = analytic_dft_forces(mf)
    if fa is None:
        fa = np.zeros_like(coords)
    with open(f"{out}/{prefix}_force.dat","a") as f:
        f.write(" ".join(map(str,fa.flatten()))+"\n")

    # FD forces
    if args.use_fd:
        fd = finite_diff_forces(mf, name, args, spins, charges)
        if fd is None:
            fd = np.zeros_like(coords)
        with open(f"{out}/{prefix}_force_fd.dat","a") as f:
            f.write(" ".join(map(str,fd.flatten()))+"\n")

def process(entry, args, spins, charges):
    idx, (name, atoms) = entry
    mf = run_dft(name, atoms, args, spins, charges)
    if mf.converged:
        save_data(mf, name, args, spins, charges)
    else:
        print(f"[{idx}] {name}: DFT failed to converge")

def main(args):
    # how many cores
    threads = lib.num_threads()
    workers = args.n_workers
    print(f"Using {threads} threads × {workers} workers = {threads*workers} cores")

    spins   = load_json(args.spin_json,   "Spin")
    charges = load_json(args.charge_json, "Charge")
    mols    = read_xyz_named(args.xyz_path)
    data    = list(enumerate(mols))

    os.makedirs(args.save_path, exist_ok=True)
    with Pool(args.n_workers) as p:
        fn = partial(process, args=args, spins=spins, charges=charges)
        p.map(fn, data)

if __name__=="__main__":
    p = argparse.ArgumentParser(description="DFT driver with JSON spin/charge, DF, checkpoint, NAO & FD")
    p.add_argument("xyz_path",        help="Multi‐XYZ file")
    p.add_argument("--spin_json",     required=True, help="Molecule→spin JSON")
    p.add_argument("--charge_json",   required=True, help="Molecule→charge JSON")
    p.add_argument("--save_path",     default="./",    help="Output directory")
    p.add_argument("--name_molecule", default="molecule", help="File prefix")
    p.add_argument("--xc",            default="PBE",   help="XC functional")
    p.add_argument("--basisset",      default="ccpvdz",help="Basis set")
    p.add_argument("--grid_level",    type=int, default=3, help="DFT grid level")
    p.add_argument("--charge",        type=int, default=0, help="Default charge")
    p.add_argument("--density_fitting", action="store_true",
                   help="Enable density fitting")
    p.add_argument("--use_fd",        action="store_true", help="Compute FD forces")
    p.add_argument("--fd_delta",      type=float, default=0.01, help="FD Δ (Å)")
    p.add_argument("--n_workers",     type=int, default=1, help="Parallel workers")
    p.add_argument("--verbose",       type=int, default=0, help="PySCF verbosity")
    args = p.parse_args()
    main(args)
