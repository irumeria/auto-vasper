# Utilities for QCE
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
import numpy as np

def ICET2List(cs, ce, new_atom):
    """
    Converts all the information contained in ICET CE to nested lists of Python
    This enables much faster processing down-the-line than relying simply on ICET or Sympy
    
    cs: ClusterSpace object that was used to train CE model
    ce: ICET cluster expansion model that has been trained
    new_atom: instance of a decorated lattice on which we want to perform DFT
    """
    new_atom #= df.loc[0,'atoms']

    ClusterVector = [1.0] # first entry is for zerolet
    Species_map = ce._cluster_space.species_maps

    SuperOrbit = LocalOrbitListGenerator(cs.orbit_list, Structure.from_atoms(new_atom), 1e-5)
    orbitList = SuperOrbit.generate_full_orbit_list().get_orbit_list()

    record = {}
    record[0] = [ce.parameters[0],[]]
    counter = 0 # keeping track of zero and non-zero ECIs

    # first loop on OrbitList
    for orbit_idx,orbit in enumerate(orbitList):

        # Identify sites
        ClusterSites = []
        ClusterCounts = []
        indices = []
        size_of_each_cluster = orbit.order
        oc = orbit.clusters
        for j in range(len(oc)):
            site = tuple([oc[j].lattice_sites[i].index for i in range(size_of_each_cluster)])
            ClusterSites.append(site)
            ClusterCounts.append(tuple([new_atom.get_atomic_numbers()[j] for j in site]))

        # Allowed number of elements at a site [make this generalize better]
        AllowedOccupations = len(ce._cluster_space.species_maps[0].items())

        for site in orbit.representative_cluster.lattice_sites:
            indices.append(site.index)

        # Second loop on ClusterVectorElements
        for cve_index, cvelement in enumerate(orbit.cluster_vector_elements):

            ClusterVectorElement = 0.0
            Sum = [] # Sum(Product)
            counter += 1

            # Third loop on ClusterCounts
            for k,clustersite in zip(ClusterCounts,ClusterSites):

                # Fourth loop on permutations of multi-component vector
                for permutation in cvelement['site_permutations']:

                    Product = []
                    for i in range(len(k)):
                        single_basis_function = {}
                        index = permutation[i]
                        LocalSpeciesMap = ce._cluster_space.species_maps[0]
                        #LocalSpeciesMap = np.argwhere(Species_map_mapper == set(substitutions[indices[index]]))[0,0]
                        for ks,vs in LocalSpeciesMap.items():
                            single_basis_function[ks] = ce._cluster_space.evaluate_cluster_function(AllowedOccupations,
                                                    cvelement['multicomponent_vector'][index], vs)
                        Product.append([clustersite[i], single_basis_function])

                    Sum.append(Product)

            ClusterVector.append(ClusterVectorElement/cvelement['multiplicity'])
            record[len(ClusterVector)-1] = [ce.parameters[len(ClusterVector)-1], cvelement['multiplicity'], Sum]
            
    return record

def verify_record(record, ce, coeffs, intercept, new_atom):
    """
    Verify if the obtained record is correct
    record: extracted dictionary
    ce: ICET ce model
    coeffs: linear regression coefficients
    intercept: offset to the CE predictions
    new_atom: template atoms object
    """
    Zs = new_atom.get_atomic_numbers()
    calculate = []

    for k in record.keys():
        if k==0:
            continue
        sums = 0.0
        for entry in record[k][2]:
            product = 1.0
            for product_term in entry:
                product *= product_term[1][Zs[product_term[0]]]
            sums += product
        sums /= record[k][1] 
        calculate.append(sums)

    print('Calcuated via record: ',np.dot([1]+calculate, coeffs) + intercept)
    print('Calculated via ICET: ',ce.predict(new_atom) + intercept)
    
def ExtractDataFromLists(records, weights, elements):
    """
    Converts all data to an extracted form that can be used by any of the search processes [GA, BO, DA or Gurobi]
    
    records: List of record type data
    weights: list of weights associated with each of the records (in the same order)
    elements: subset of elements we want to search over
    """
    extracted_data = []
    
    for record,weight in zip(records,weights):

        for k in record.keys():
            if k != 0:
                data = record[k]
                outer_coeff = weight * data[0] / data[1]

                for entry in data[2]: #cluster_orbits
                    term = [outer_coeff, []]
                    for product_term in entry: #(cluster_alphas, cluster_atomic_idxs)
                        coef = []
                        for idx,i in enumerate(elements):
                            coef.append(product_term[1][i])
                        term[1].append([product_term[0],coef])    
                    extracted_data.append(term)
    return extracted_data

def Res2Atoms(state, elements, Allowed_Zs, site_num, template_atom, ce):
        """
        Converts results from DA output to an Atoms object
        state: bit vector from DA
        elements: subset of elements under consideration
        Allowed_Zs: list of atomic numbers that are allowed to occupy sites
        site_num: Number of sites
        template_atom: atom object used for decoration
        ce: list of ce-models

        returns:
            (new-atoms-object, properties)
        """
        shaped_state = state[:len(elements)*site_num].reshape(site_num,len(elements))
        element_indices = np.argmax(shaped_state,axis=1)
        outcome_Z = np.array(elements)[element_indices]
        at = template_atom.copy()

        final_Z, tracker = [], 0
        for Z in at.get_atomic_numbers():
            if Z not in Allowed_Zs:
                final_Z.append(Z)
            else:
                final_Z.append(outcome_Z[tracker])
                tracker += 1

        at.set_atomic_numbers(final_Z)
        return at, [model.predict(at) for model in ce]
