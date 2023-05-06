# ==============
# This demo is for combining the usage of icet and vasp
# ==============

import numpy as np
from ase.db import connect
from ase.atoms import Atoms
from ase.cell import Cell

from autovasper.utils import to_primitive_structure
from autovasper.solve import StandardSolver

if __name__ == "__main__":

    # input parameters
    max_size = 5
    max_cal_sample = 20
    database_name = 'AgPddata.db'
    vasp_workdir = './icet_rundir'
    atoms = ['Ag', 'Pd', 'Pd']
    
    sites = np.array([
        [0.0, 0.0, 0.0],
        [0.33333333333333326, 0.3333333333333334, 0.6666666666666667],
        [0.6666666666666666, 0.6666666666666667, 0.3333333333333333]]
    )
    lattice = np.array([
        [0.0 ,2.045, 2.045], 
        [2.045, 0.0, 2.045], 
        [2.045, 2.045, 0.0]]
    )
    
    lattice, atoms, sites = to_primitive_structure(lattice, atoms, sites)

    print(sites @ lattice)

    print(" === experiment start === ")

    primitive_structure = Atoms(symbols=atoms,
                        positions=sites @ lattice,
                        cell=Cell(lattice), pbc=[True, True, True]
                    )

    chemical_symbols = list(set(atoms))

    db = connect(database_name)

    StandardSolver.vasp_loop(
            primitive_structure, 
            vasp_workdir, 
            max_size, 
            max_cal_sample, 
            db)
