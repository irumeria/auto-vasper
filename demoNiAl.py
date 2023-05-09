# ==============
# This demo is for handling the result from atat-vasp
# draw the phase diagram, and predict the ground state energy component by EA 
# ==============

import numpy as np
from ase.db import connect
from ase.atoms import Atoms
from ase.cell import Cell
from ase.build import make_supercell 
from icet import ClusterSpace, StructureContainer, ClusterExpansion
import os
import json
from sklearn.linear_model import LassoCV
from sklearn.metrics import *

from autovasper.utils import to_primitive_structure
from autovasper.vasp import VaspIO
from autovasper.gasearch.ground_state import get_ground_state

if __name__ == "__main__":

    database = connect("NiAl.db")

    if not os.path.exists("NiAl.db"):
        VaspIO.archive_atat_record(
            "vasp_rundir",
            database
        )

    _structure = database.get(id=1).toatoms()
    lattice, atoms, sites = to_primitive_structure( # primitive structure
                            _structure.cell[:],
                            _structure.get_chemical_symbols(),
                            _structure.get_scaled_positions()
                            )  

    primitive_structure = Atoms(symbols=atoms,
                        positions=sites @ lattice,
                        cell=Cell(lattice), pbc=[True, True, True]
                    )

    cs = ClusterSpace(structure=primitive_structure,
                cutoffs=[6.0, 6.0, 6.0],
                chemical_symbols=['Ni', 'Al'])
    
    sc = StructureContainer(cluster_space=cs)

    for row in database.select():
        try:
            sc.add_structure(structure=row.toatoms(),
                            properties={'dft_energy': row.dft_energy})
        except Exception as e: print(e)
    
    print(sc)

    model = LassoCV(fit_intercept=False)
    y = sc.get_fit_data(key='dft_energy')[1]
    X = sc.get_fit_data(key='dft_energy')[0]
    model.fit(X, y)
    print('R2 score: {:.3f}'.format(model.score(X, y)))
    preds = model.predict(X)
    print("MAE: {:.3f}".format(mean_absolute_error(preds, y)))
    ce = ClusterExpansion(cluster_space=cs, parameters=model.coef_)

    print(ce)

    # predict the ground state energy
    data = {'concentration': [], 'reference_energy': [], 'predicted_energy': []}
    for row in database.select():
        conc_dict = json.loads(row.concentration)
        if 'Ni' not in conc_dict:
            conc = 0.
        elif 'Al' not in conc_dict:
            conc = 1.
        else:
            conc = conc_dict['Ni'] / (conc_dict['Ni'] + conc_dict['Al'])
        data['concentration'].append(conc)
        data['reference_energy'].append(row.dft_energy)
        data['predicted_energy'].append(ce.predict(row.toatoms()))
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel(r'Ni concentration')
    ax.set_ylabel(r'Mixing energy (meV/atom)')
    ax.set_xlim([0, 1])
    ax.set_ylim([-69, 15])
    ax.scatter(data['concentration'], data['reference_energy'],
            marker='o', label='reference')
    ax.scatter(data['concentration'], data['predicted_energy'],
            marker='x', label='CE prediction')
    plt.savefig('NiAl.png')


    atom_type = list(set(primitive_structure.get_atomic_numbers()))
    chemical_symbols = list(set(primitive_structure.get_chemical_symbols()))
    
    supercell = make_supercell(cs.primitive_structure, 2*np.eye(3) )

    get_ground_state(cluster_space=cs,
                    cluster_expansion=ce,
                    atoms=supercell,
                    regression_model_weight=model.intercept_,
                    elements = atom_type,
                    elements_label = chemical_symbols,
                    generation=100,
                    population=100,
                    cross_element=False,
                    element_ratio=[5,3]
                    )
