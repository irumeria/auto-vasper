# auto vasper
this project is for using vasp(Vienna Ab initio Simulation Package) easily with python. 

There are two aims of this project

## Analysis the results of vasp calculation

The [recent research](https://arxiv.org/abs/2205.09007) has shown the ability of optimal combination algorithm in searching the ground state energy of given components. We can use these algorithms to avoid enumerating all the structures of a given supercell.

**demoNiAl.py** is a demo script for searching ground state energy with the auto-vasper. The brief workflow is as follows:

1. read the results from [**atat-vasp** calculation](https://arxiv.org/pdf/1907.10151.pdf). This will write the record to a given [ASE atom database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html) file.

```py
database = connect("NiAl.db")

if not os.path.exists("NiAl.db"):
    VaspIO.archive_atat_record(
        "vasp_rundir", # demo results from atat-vasp
        database
    )
```

2. obtain the primitive structure

3. constuct the icet ClusterExpansion Object

```py
model = LassoCV(fit_intercept=False)
y = sc.get_fit_data(key='dft_energy')[1]
X = sc.get_fit_data(key='dft_energy')[0]
model.fit(X, y)
ce = ClusterExpansion(cluster_space=cs, parameters=model.coef_)
```

4. search the ground state by EA solver
```py
get_ground_state(cluster_space=cs,
                cluster_expansion=ce,
                atoms=supercell,
                regression_model_weight=model.intercept_,
                elements = atom_type,
                elements_label = chemical_symbols,
                generation=100,
                population=300)
```

## Use ICET with Vasp (testing)

ICET declares that their structure derivativing algorithm is purposed by [Gus L. W. Hart and Rodney W. Forcade](https://icet.materialsmodeling.org/moduleref_icet/tools.html#icet.tools.enumerate_structures). It could be faster than the FWZ algorithm using by atat accroding to this article : [*Algorithm for generating derivative structures*](http://dx.doi.org/10.1103/PhysRevB.77.224115).

So, I want to test if ICET can work with Vasp instead of ATAT. In **demoAgPd.py**. There are two step in the script as follows: 

0. Preparations. The POTCAR is search from 'template/Potentials/'. POSCAR and KPOINTS is generated automatically. The INCAR cannot be generated by now, the auto-vasper will copy it directly from 'template/INCAR'
```sh
mkdir template/Potentials
cp -r PATH_TO_PBW_POTENTIALS_IN_YOUR_COMPUTER/* template/Potentials/
vim template/INCAR
# set the content of the incar file
```

1. defining the primitive structure. (ASE Atom object)
```py
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
primitive_structure = Atoms(symbols=atoms,
                        positions=sites @ lattice,
                        cell=Cell(lattice), pbc=[True, True, True]
                    )
```

2. loop the vasp calculation from size of primitive structure to max_size. when the calculated samples for vasp is equal to max_cal_sample, the size will increase. 
   
```py
StandardSolver.vasp_loop(
        primitive_structure, 
        vasp_workdir, 
        max_size, 
        max_cal_sample, 
        db,
        cutoffs=cutoffs)
```

