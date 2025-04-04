#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

# This is an implementation of the "Number IO" problem from the
# PSB1 suite of software synthesis benchmarks:
# https://dl.acm.org/doi/10.1145/2739480.2754769
#
# Here your program is given an integer and a floating point value
# and it needs to convert the integer to a float, add that to the floating
# point value, and return the sum. I'm implementing this as a typed problem,
# although one could obviously just add the two values in Python and be done.

import operator
import math
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions

# Here we're assuming that `i` is an integer
# value and converting it to a floating point
# value. Note that Python doesn't check type
# declarations at run-time, so we can call
# `int_to_float` with non-integral values
# (e.g., strings) and this will still "work".
def int_to_float(i: int):
    return float(i)

# See comments on `int_to_float`.
def float_to_int(x: float):
    return int(x)

# Arithmetic for integers

def addInt(left: int, right: int):
    return left + right

def subInt(left: int, right: int):
    return left - right

def mulInt(left: int, right: int):
    return left * right

def sqrInt(x: int):
    return x * x

def doubleInt(x: int):
    return x + x

def protectedDivInt(left: int, right: int):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Arithmetic for floats

def addFloat(left: float, right: float):
    return left + right

def subFloat(left: float, right: float):
    return left - right

def mulFloat(left: float, right: float):
    return left * right

def protectedDivFloat(left: float, right: float):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def sqrFloat(x: float):
    return x * x

def doubleFloat(x: float):
    return x + x

pset = gp.PrimitiveSetTyped("MAIN", [int, float], float)

pset.addPrimitive(addInt, [int, int], int)
pset.addPrimitive(subInt, [int, int], int)
pset.addPrimitive(mulInt, [int, int], int)
pset.addPrimitive(protectedDivInt, [int, int], int)

pset.addPrimitive(addFloat, [float, float], float)
pset.addPrimitive(subFloat, [float, float], float)
pset.addPrimitive(mulFloat, [float, float], float)
pset.addPrimitive(protectedDivFloat, [float, float], float)

pset.addPrimitive(int_to_float, [int], float)
pset.addPrimitive(float_to_int, [float], int)

# Add some more operators to make it more difficult.
pset.addPrimitive(sqrInt, [float], float)
pset.addPrimitive(doubleInt, [float], float)
pset.addPrimitive(sqrFloat, [float], float)
pset.addPrimitive(doubleFloat, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)

pset.renameArguments(ARG0='i')
pset.renameArguments(ARG1='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# `genHalfAndHalf` generates half the initial population using the `Full` method and half
# using the `Grow` method. `Full` picks a random depth (between min and max) and generates
# a _full_ tree of that depth. `Grow` picks a random depth, and grows a tree until, along
# each branch, either a leaf is chosen or the maximum depth is reached.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x^9 + 3x^6 + 3x^3 + 2
    # `sqerrors` is the square of all the errors.
    sqerrors = ((func(i, x) - (float(i) + x))**2 for (i, x) in points)

    # This computes the average of the squared errors, i.e., the mean squared error,
    # i.e., the MSE.
    return math.fsum(sqerrors) / len(points),

training_inputs = [(random.randint(-100, 100), random.random()) for index in range(0, 40)]

print(training_inputs)

# The training cases are from -4 (inclusive) to +4 (exclusive) in increments of 0.25.
toolbox.register("evaluate", evalSymbReg, points=training_inputs)

# Tournament selection with tournament size 3
toolbox.register("select", tools.selTournament, tournsize=3)

# One-point crossover, i.e., remove a randomly chosen subtree from the parent,
# and replace it with a randomly chosen subtree from a second parent.
toolbox.register("mate", gp.cxOnePoint)

# Remove a randomly chosen subtree and replace it with a "full" randomly generated
# tree whose depth is between min and max. `expr_mut` is specifying a way of
# generating new trees using `Full`. `mutUniform` below says that `mutate` will
# _uniformly_ choose a subtree to remove and replace it using `expr_mut`, i.e.,
# `Full`.
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Set the maximum height of a tree after either crossover or mutation to be 17.
# When an invalid (over the limit) child is generated, it is simply replaced
# by one of its parents, randomly selected. This replacement policy is a Real Bad Idea
# because it rewards parents who are likely to create offspring that are above the
# threshold.
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    # random.seed(318)

    # Sets the population size to 30.
    # (This problem is stupid easy, so we can get away with very small population sizes.)
    pop = toolbox.population(n=30)
    # Tracks the single best individual over the entire run.
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Does the run, going for 40 generations (the 5th argument to `eaSimple`).
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    # print log

    # Print the members of the hall of fame
    for winner in hof:
        print(str(winner))

    return pop, log, hof

if __name__ == "__main__":
    main()
