
# GA-based search
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import pandas as pd
import numpy as np
from . import engines
from . import utils

def get_ground_state(cluster_space, cluster_expansion, atoms, regression_model_weight, elements, elements_label, generation=100, population=100):

    assert len(elements) == len(elements_label)

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
    toolbox.register("attr_bool", random.randint, 0, len(elements)-1) # Attribute generator 
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(atoms)) # Structure initializers
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", costfunctionGA)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.7, ngen=generation, stats=stats, halloffame=hof, verbose=True)

    print( len(atoms))
    atom_amount = len(hof[0])
    hof = np.array(hof[0])
    atom_count = np.zeros(len(elements_label))
    text = ""
    for i,el in enumerate(elements_label):
        atom_count[i] += np.count_nonzero(hof == i)
        text = text + str(el) + str(atom_count[i]/atom_amount) 
    print(text)
