import os
import numpy as np
from ase.db import connect
import shutil
from ase.io import write, read
from ase.atoms import Atoms
from ase.cell import Cell
import subprocess
import json
from collections import OrderedDict 
from .utils import cal_conc_in_structure

class VaspIO():

    def write_to_POSCAR(poscar_path, atom_posi, mat, atom_symbols, type="cartesian"):

        atom_posi = np.array(atom_posi)

        if type == "cartesian":
            cell = Cell(mat)
            structure = Atoms(symbols=atom_symbols, positions=atom_posi,
                            cell=cell, pbc=[True, True, True])

            write(poscar_path, structure, format="vasp")

        elif type == "direct":
            of = open(poscar_path, "w+")

            frac_posi = atom_posi @ np.linalg.inv(mat)
            species_dict = {}
            for (i, item) in enumerate(atom_symbols):
                if item not in list(species_dict.keys()):
                    species_dict[item] = [i]
                else:
                    species_dict[item].append(i)

            # part 1
            for symbol in atom_symbols:
                of.write(symbol)
                of.write(" ")
            of.write("\n1.0\n")

            # part 2: lattice matrix
            for i in range(3):
                of.write("  ")
                for j in range(3):
                    of.write('%.8f' % mat[i,j])
                    of.write(" ")
                of.write("\n")
            
            # part 3: atom symbols
            for key in list(species_dict.keys()):
                of.write(key)
                of.write(" ")
            of.write("\n")
            for key in list(species_dict.keys()):
                i = 0
                while i < len(key) - 1:
                    of.write(" ")
                    i += 1
                of.write(str(len(species_dict[key])))
                of.write(" ")
            of.write("\n")

            # part 4: atom position
            indexs_list = []
            for key in list(species_dict.keys()):
                if len(indexs_list) == 0:
                    indexs_list = np.array(species_dict[key])
                else:
                    indexs_list = np.concatenate([indexs_list, species_dict[key]])
            of.write("Direct\n")
            for index in indexs_list:
                of.write(" ")
                for item in frac_posi[index]:
                    of.write(" ")
                    of.write('%.8f' % item)
                of.write("\n")

            of.flush()
            of.close()

        else:
            raise ValueError(
                "only 'cartesian' and 'direct' are supported in the option 'type' ")

    def archive_vasp_record(
        vasp_work_dir, 
        structure=None, 
        vasp_output="OUTCAR", 
        database=None):
        outcar_path = os.path.join(vasp_work_dir, vasp_output)
        f = open(outcar_path, "r")
        lines = f.readlines()
        energy = None
        
        for line in lines:
            if "energy  without entropy" in line:
                energy = eval(line.split("energy  without entropy=")[1].split("energy(sigma->0)")[0])
                break
        assert energy is not None, "no energy found in the output file in path "+outcar_path

        if database is not None:
            assert structure is not None, "please provide structure when inserting the record to database"
            conc = cal_conc_in_structure(structure)
            database.write(
                structure, 
                {"dft_energy": energy, "concentration": str(conc)}
            )
            
        return energy

    def generate_POTCAR(chemical_symbols, vasp_workdir):
        # obtain pseudopotential
        potcar = ""
        potential_path = 'template/Potentials'
        for chemical_symbol in chemical_symbols:
            entries = os.listdir(potential_path)
            assert chemical_symbol in entries, \
                "chemical symbol "+chemical_symbol+" cannot be found in potential library"
            for entry in entries:
                if entry == chemical_symbol: 
                    if chemical_symbol == "Li": # TODO : _sv ..
                        entry += "_sv"
                    f = open(os.path.join(potential_path, entry, "POTCAR"), "r")
                    potcar += f.read()
                    f.close()
                    break

        potcar_file = open(os.path.join(vasp_workdir, "POTCAR"), "w+")
        potcar_file.write(potcar)
        potcar_file.flush()
        potcar_file.close()

    def generate_KPOINTS(vasp_workdir, temp_path="./kpoints"):
        
        assert os.path.exists(os.path.join(vasp_workdir, "POSCAR")), \
                "'POSCAR' should be generate first in 'vasp_workdir', please call generate_POTCAR()"

        isExist = os.path.exists(temp_path)
        if not isExist:
            os.makedirs(temp_path)

        # generate KPOINTS
        shutil.copy(os.path.join(vasp_workdir, "POSCAR"), os.path.join(vasp_workdir, "KPOINTS"))
        poscar = open(os.path.join(vasp_workdir, "POSCAR"), "r")
        kpoints = open(os.path.join(temp_path, "KPOINTS"), "w")
        kcontent = poscar.read()
        kcontent = "[INCAR]\nKPPRA = 8000\n[POSCAR]\n" + kcontent
        kpoints.write(kcontent)
        kpoints.flush()
        kpoints.close()
        poscar.close()
        subprocess.call(["ezvasp","KPOINTS"], cwd=temp_path)
        shutil.move(os.path.join(temp_path, "KPOINTS"), os.path.join(vasp_workdir, "KPOINTS"))
    
    def generate_INCAR(chemical_symbols, vasp_workdir):
        # copy INCAR from template library
        shutil.copy("./template/INCAR", os.path.join(vasp_workdir, "INCAR"))

    def generate_POSCAR(primitive_structure, vasp_workdir):
        # generate POSCAR
        VaspIO.write_to_POSCAR(os.path.join(vasp_workdir, "POSCAR"),
            primitive_structure.get_positions(),
            primitive_structure.cell[:],
            primitive_structure.get_chemical_symbols(),
            type="direct")

        # write(os.path.join(vasp_workdir, "POSCAR"), primitive_structure, format="vasp")

    def generate_vasp_input(primitive_structure, vasp_workdir, k_point_temp_path="./kpoints"):

        isExist = os.path.exists(vasp_workdir)
        if not isExist:
            os.makedirs(vasp_workdir)
        
        chemical_symbols = list(OrderedDict.fromkeys(primitive_structure.get_chemical_symbols())) 

        VaspIO.generate_INCAR(chemical_symbols, vasp_workdir)

        VaspIO.generate_POTCAR(chemical_symbols, vasp_workdir)

        VaspIO.generate_POSCAR(primitive_structure, vasp_workdir=vasp_workdir)

        VaspIO.generate_KPOINTS(vasp_workdir, temp_path=k_point_temp_path)

    def vasp_run(vasp_workdir, vasp_commad):
        subprocess.call(["mpirun", vasp_commad], cwd=vasp_workdir)

    def read_vasp_in_Cartesian(fp):
        vaspin = open(fp)
        lines = vaspin.readlines()
        POSCAR_flag = -1
        lattice = []
        atoms = []
        sites = []
        for index, line in enumerate(lines):
            if "[POSCAR]" in line:
                POSCAR_flag = index
            if index > POSCAR_flag + 2 and index < POSCAR_flag + 6 and POSCAR_flag > 0:
                lattice.append([float(x) for x in line.split('\n')[0].split(' ')])
            if index > POSCAR_flag + 6 and POSCAR_flag > 0:
                sites.append([float(x) for x in line.split(' ')[:-1]])
                atoms.append(line.split('\n')[0].split(' ')[-1])
        
        return Atoms(symbols=atoms,
                        positions=sites,
                        cell=Cell(lattice), pbc=[True, True, True]
                    )



    def archive_atat_record(atat_workdir, database):
        subfolders = [ f.path for f in os.scandir(atat_workdir) if f.is_dir() ]

        for fld in subfolders:
            try:
                fld = os.path.abspath(fld)
                energy = open(os.path.join(fld, "energy" ), "r")\
                                                        .readlines()[0]\
                                                        .split('\n')[0]
                structure = VaspIO.read_vasp_in_Cartesian(os.path.join(fld, "vasp.in" ))
                conc = cal_conc_in_structure(structure)

                database.write(
                    structure,
                    {"dft_energy": float(energy), "concentration": json.dumps(conc)}
                )
            except:
                print("floder read faild", fld)

if __name__ == "__main__":
    from ase.db import connect
    db = connect('reference_data.db')
    primitive_structure = db.get(id=1).toatoms()  # primitive structure for test
    VaspIO.archive_vasp_record("./", structure=primitive_structure,database=db)
    
