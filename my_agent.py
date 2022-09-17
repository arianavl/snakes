__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np
import random as rand
import math

agentName = "<my_agent>"
perceptFieldOfVision = 3   # Choose either 3,5,7 or 9
perceptFrames = 1          # Choose either 1,2,3 or 4
trainingSchedule = [("self", 300), ("random", 200)]

# git
# This is the class for your snake/agent
class Snake:
    def __init__(self, nPercepts, actions):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values)

        self.nPercepts = nPercepts
        # print("nPercepts: " + str(nPercepts))
        self.actions = actions
        chromosome = [0] * 3
        for i in range(3):
            one_chromosome = []
            for n in range(int(nPercepts/perceptFieldOfVision)):
                one_percept = []
                for y in range(perceptFieldOfVision):
                    one_percept.append(rand.random())
                one_chromosome.append(one_percept)
            chromosome[i] = one_chromosome

        self.chromosome = np.array(chromosome)
        # print("self.chromosome: " + str(self.chromosome))

    def AgentFunction(self, percepts):
        # print("Agent function")
        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.
        #
        # The 'actions' variable must be returned and it must be a 3-item list or 3-dim numpy vector

        #
        # The index of the largest numbers in the 'actions' vector/list is the action taken
        # with the following interpretation:
        # 0 - move left
        # 1 - move forward
        # 2 - move right
        #
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.
        # print("percepts: " + str(percepts))
        # print("percepts len: " + str(len(percepts)))
        # print("chromosome: " + str(np.array(self.chromosome)))
        weight_a = 0
        weight_b = 0
        weight_c = 0

        for x in range(len(percepts)):
            pre_index_a = self.chromosome[x][0] * percepts[x]
            # print("perceptsA: " + str(percepts))
            # print("chromosomeA: " + str(self.chromosome[0]))
            pre_index_b = self.chromosome[x][1] * percepts[x]
            # print("perceptsB: " + str(percepts))
            # print("chromosomeB: " + str(self.chromosome[1]))
            pre_index_c = self.chromosome[x][2] * percepts[x]
            # print("perceptsC: " + str(percepts))
            # print("chromosomeC: " + str(self.chromosome[2]))

            for a in pre_index_a:
                for y in a:
                    weight_a += y

            for b in pre_index_b:
                for y in b:
                    weight_b += y

            for c in pre_index_c:
                for y in c:
                    weight_c += y

            weight_a += rand.randint(0, 2)
            weight_b += rand.randint(0, 2)
            weight_c += rand.randint(0, 2)

        # print(self.chromosome)
        # print("preIndexA" + str(pre_index_a))
        # count = 0
        # for x in pre_index:
        #    count += x
        # print("weight_c: " + str(weight_c))
        # print("weight_a: " + str(weight_a))
        # print("weight_b: " + str(weight_b))

        if weight_c > weight_a and weight_c > weight_b:
            index = 2
        elif weight_b > weight_a and weight_b > weight_c:
            index = 1
        else:
            index = 0
        # print("index: " + str(index))
        # .
        # .
        # .
        # print(percepts) [[[0 0 0]
        #                   [0 1 0]
        #                   [1 1 0]]]

        # mindex = np.random.randint(low=0, high=len(self.actions))
        return self.actions[index]


def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros(N)

    # This loop iterates over your agents in the old population - the purpose of this boiler plate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, snake in enumerate(population):
        # snake is an instance of Snake class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, the object has the following attributes provided by the
        # game engine:
        #
        # snake.size - list of snake sizes over the game turns
        # .
        # .
        # .
        maxSize = np.max(snake.sizes)
        turnsAlive = np.sum(snake.sizes > 0)
        maxTurns = len(snake.sizes)

        # This fitness functions considers snake size plus the fraction of turns the snake
        # lasted for.  It should be a reasonable fitness function, though you're free
        # to augment it with information from other stats as well
        fitness[n] = maxSize + turnsAlive / maxTurns

    return fitness


def aSnakeFitness(s):
    maxSize = np.max(s.sizes)
    turnsAlive = np.sum(s.sizes > 0)
    maxTurns = len(s.sizes)

    # This fitness functions considers snake size plus the fraction of turns the snake
    # lasted for.  It should be a reasonable fitness function, though you're free
    # to augment it with information from other stats as well
    return maxSize + turnsAlive / maxTurns



def newGeneration(old_population):
    print("new Generation")

    # This function should return a tuple consisting of:
    # - a list of the new_population of snakes that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    nPercepts = old_population[0].nPercepts
    actions = old_population[0].actions

    fitness = evalFitness(old_population)

    max1 = 0
    index1 = 0
    max2 = 0
    index2 = 0
    for n, x in enumerate(fitness):
        if x > max1:
            max1 = x
            index1 = n
        elif x > max2:
            max2 = x
            index2 = n

    print("fitness: " + str(fitness))
    #print("old_population: " + str(old_population[0].chromosome))

    # Sort fitnesses, choose a fitness, find it in original unsorted fitness, then find in
    # old_population


    # At this point you should sort the old_population snakes according to fitness, setting it up for parent
    # selection.
    # .
    # .
    # .
    # print("parent1: " + str(old_population[0].chromosome))

    # Create new population list...
    new_population = list()

    new_population.append(old_population[index1])
    new_population.append(old_population[index2])
    # print("new population: " + str(fitness[index1]) + " 2: " + str(fitness[index2]))
    print("New fitness: " + str(evalFitness(new_population)))
    for n in range(N-2):

        # Create a new snake
        new_snake = Snake(nPercepts, actions)

        parent1fitness = tournament(old_population)
        parent2fitness = tournament(old_population)
        print("fitnesses: " + str(fitness))
        print("fitness1: " + str(aSnakeFitness(parent1fitness)) + " fitness2: " + str(aSnakeFitness(parent2fitness)))
        # parent1 = old_population[0]
        # parent2 = old_population[1]

        new_snake.chromosome = newChromosome(parent1fitness.chromosome, parent2fitness.chromosome)

        # Here you should modify the new snakes chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_snake.chromosome

        # Consider implementing elitism, mutation and various other
        # strategies for producing a new creature.

        # .
        # .
        # .

        # Add the new snake to the new population
        new_population.append(new_snake)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return new_population, avg_fitness


def roulette_wheel_selection(population, fitness):

    total_fit = fitness.sum()
    print("fitness" + str(fitness))
    prob_list = fitness / total_fit
    print("prob_list" + str(prob_list))

    # Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population))), p=prob_list, replace=True)

   # print("a: " + str(progenitor_list_a))

    return progenitor_list_a


def newChromosome(p1Chromo, p2Chromo):
    chromosome = p2Chromo
  #  print("p1Chromo: " + str(p1Chromo))
  #  print("p2Chromo: " + str(p2Chromo))
    for n, x in enumerate(p1Chromo):
        # print("chromosome[n]: " + str(chromosome[n]))
        # print("x: " + str(x))
        # for y, w in enumerate(x):
            # print("w: " + str(w))
        coinFlip = rand.randint(0, 1)
        if coinFlip == 0:
            chromosome[n] = x
       # print("changed chromo: " + str(chromosome))


    # mutate??
    mutate = rand.randint(0, 200)
    if mutate == 1:
        chromosome = mutateChromosome(chromosome)
    return chromosome


def mutateChromosome(chromosome):
    return chromosome


# Use many tournaments to get parents
def tournament(population):
    for i in range(len(population)):
        population_sample = rand.sample(population, 3)
        # print("candidates: " + str(candidates) + "num: " + str(len(population)))
        return max(population_sample, key=lambda x: aSnakeFitness(x))
