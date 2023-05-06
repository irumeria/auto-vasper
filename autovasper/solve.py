from autovasper.vasp import VaspIO
from icet.tools import enumerate_structures 
import numpy as np
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from sklearn.linear_model import LassoCV

class StandardSolver():

    def vasp_loop(primitive_structure, 
                            vasp_workdir, 
                            max_size, 
                            max_cal_sample, 
                            database, 
                            min_size=1,
                            vasp_commad=None):
        
        if vasp_commad == None:
            vasp_commad = "vasp_std"

        chemical_symbols = primitive_structure.get_chemical_symbols()

        for size in range(min_size, max_size+1):

            print("the structure size now is " + str(size))
            

            structures = []
            for _structure in enumerate_structures(
                                        primitive_structure,
                                        sizes=[size], 
                                        chemical_symbols=chemical_symbols):
                structures.append(_structure)

            cal_sample = max_cal_sample
            if max_cal_sample > len(structures):
                cal_sample = len(structures)
            
            print(cal_sample)
            
            randChoice = np.random.choice(np.arange(len(structures)), cal_sample, replace=False)
            
            
            for choice in randChoice:
                structure = structures[choice]
                VaspIO.generate_vasp_input(structure, vasp_workdir, k_point_temp_path="./kpoints")
                VaspIO.vasp_run(vasp_workdir, vasp_commad)
                VaspIO.archive_vasp_record(vasp_workdir,
                                                structure=structure,
                                                vasp_output="OUTCAR", 
                                                database=database)

            # ==== sperate the CE part to a new process

            # =====

    def vasp_solve(structure, vasp_workdir, database=None, vasp_commad=None):

        if vasp_commad == None:
            vasp_commad = "vasp_std"

        VaspIO.generate_vasp_input(structure, vasp_workdir, k_point_temp_path="./kpoints")

        VaspIO.vasp_run(vasp_workdir, vasp_commad)

        return VaspIO.archive_vasp_record(vasp_workdir, 
                                    structure=structure,
                                    vasp_output="OUTCAR",
                                    database=database)
                                


