from pymatgen.core import Lattice, Structure
import numpy as np
import itertools

def to_primitive_structure(lattice, atom_symbols, atom_posi):
    lat = Lattice(lattice)
    structure = Structure(lat, atom_symbols, atom_posi)
    primitive_structure = structure.get_primitive_structure()
    
    return primitive_structure.lattice.matrix,\
            np.array([str(element) for element in primitive_structure.species]),\
            primitive_structure.frac_coords

def get_conc_balls_in_boxes(balls, boxes):
    rng = list(range(balls + 1)) * boxes
    return set(i for i in itertools.permutations(rng, boxes) if sum(i) == balls)

def cal_conc_in_structure(structure):
    symbols = structure.get_chemical_symbols()
    return dict((x,symbols.count(x)) for x in set(symbols))