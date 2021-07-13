import random
import operator
import math
import numpy as np
from deap import base, creator, tools, gp, algorithms
import pandas as pd


x_in = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5,  -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
y_out = [37, 24.16016, 15.0625, 8.91016, 5, 2.72266, 1.5625, 1.09766, 1, 1.03516, 1.0625, 1.03516, 1, 1.09766, 1.5625,
         2.72266, 5, 8.91016, 15.0625]




# against a zero division error
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# return the real output to compare
def output(x):
    index = x_in.index(x)
    return y_out[index]


# create function set with + - * /
function_set = gp.PrimitiveSet("MAIN", arity=1)
function_set.addPrimitive(operator.add, 2)
function_set.addPrimitive(operator.sub, 2)
function_set.addPrimitive(operator.mul, 2)
function_set.addPrimitive(protectedDiv, 2)
function_set.addPrimitive(operator.neg, 1)
function_set.addPrimitive(math.cos, 1)
function_set.addPrimitive(math.sin, 1)
function_set.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
function_set.renameArguments(ARG0='x')

# create fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=function_set)

# register all the objects
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=function_set, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=function_set)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real output
    sqerrors = ((func(x) - output(x))**2 for x in points)
    return math.fsum(sqerrors) / len(points),


toolbox.register("evaluate", evalSymbReg, points=x_in)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=function_set)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

if __name__ == "__main__":
    random.seed(318)

    # generate population
    N_POP = 100
    pop = toolbox.population(n=N_POP)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
    #print(gp.compile(hof, function_set))

    #tree = gp.PrimitiveTree(toolbox.individual)

