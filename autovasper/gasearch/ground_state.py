
# GA-based search
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import itertools
import pandas as pd
import numpy as np
from . import engines
from . import utils

def get_ground_state(
        cluster_space, 
        cluster_expansion, 
        atoms, 
        regression_model_weight, 
        elements, 
        elements_label, 
        generation=100, 
        population=100,
        fame_size=1,
        cross_element=True,
        element_ratio=[]):

    assert len(elements) == len(elements_label) , "length of elements should be equal to the length of elements_label"

    if not cross_element: 
        assert len(element_ratio) > 0
        assert len(atoms) % sum(element_ratio) == 0
        scale = int(len(atoms) / sum(element_ratio))
        element_ratio = np.array(element_ratio)
        element_ratio *= scale

    # Converting CE model to QCE
    record = utils.ICET2List(cluster_space, cluster_expansion, atoms)
    print(len(record))
    utils.verify_record(record, cluster_expansion, cluster_expansion.parameters, regression_model_weight, atoms) # last argument can be any valid atoms object
    extracted_data = utils.ExtractDataFromLists([record], [1.0], elements)

    vectorizedinteractions = engines.prepare_GA_BO(extracted_data)
    costfunctionGA = lambda x: engines.cost_function_GA(x, vectorizedinteractions)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    mate_rate = 0.3
    if cross_element: 
        toolbox.register("attr_bool", random.randint, 0, len(elements)-1) # Attribute generator 
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(atoms)) # Structure initializers
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    else:
        '''
        the origin paper&code implement the function for searching the lowest energy of the whole phase diagram
        here implement the part for searching ground state on each component ratio
        '''
        mate_rate = 0
        origin_array = np.zeros(len(atoms))
        current_index = 0
        for i, ratio in enumerate(element_ratio):
            current_index += ratio
            origin_array[ratio: current_index+element_ratio[i+1]] = i+1
            if i == len(element_ratio) - 2:
                break
        
        origin_array = list(origin_array.astype(int))
        def init_to_container(container, func):
            return container(func())

        toolbox.register("attr_bool", random.sample, origin_array, len(origin_array)) # Attribute generator 
        toolbox.register("individual", init_to_container, creator.Individual, toolbox.attr_bool) # Structure initializers
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", costfunctionGA)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=population)
    print(pop[:10])
    hof = tools.HallOfFame(fame_size)
    pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=mate_rate, mutpb=0.7, ngen=generation, stats=stats, halloffame=hof, verbose=True)

    atom_amount = len(hof[0])
    hof = np.array(hof[0])
    atom_count = np.zeros(len(elements_label))
    text = ""
    for i,el in enumerate(elements_label):
        atom_count[i] += np.count_nonzero(hof == i)
        text = text + str(el) + str(atom_count[i]/atom_amount) 
    print(text)
