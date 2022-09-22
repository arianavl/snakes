__author__ = "<Ariana van Lith>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<vanar987@student.otago.ac.nz>"

import numpy as np
import random as rand
import math

agentName = "<my_agent>"
perceptFieldOfVision = 3  # Choose either 3,5,7 or 9
perceptFrames = 1  # Choose either 1,2,3 or 4
trainingSchedule = [("random", 500),  ("self", 0)]
file = open("sample.txt", "w")
hiddenFunctionSizeWeights = 12
bias = 8
low = -50
high = 50
tourSampleSize = 4
# trainingSchedule = None


# This is the class for your snake/agent
class Snake:
    def __init__(self, nPercepts, actions):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values)

        self.nPercepts = nPercepts
        self.actions = actions
        self.population_size = 3
        self.lastActions = [0, 1, 2]
        self.lastPercept = []
        self.lastAveFitness = 0
        # ----- Multi -----
        chromosome = \
            [np.random.uniform(low, high) for i in
             range((nPercepts * 3) + hiddenFunctionSizeWeights + bias)]

        self.chromosome = np.array(chromosome)

    def AgentFunction(self, percepts):
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

        # Change values on percepts to heavier weighted values
        flatPercepts = percepts.flatten()
        for i, n in enumerate(flatPercepts):
            if n == -1:
                flatPercepts[i] = -10
            elif n == 1:
                flatPercepts[i] = -10
            elif n == 2:
                flatPercepts[i] = 10

        # ------ Multilayer Perceptron -------
        weightsCount = 0

        # --- First hidden layer ---
        f1 = 0
        for i in range(len(flatPercepts)):
            f1 += flatPercepts[i] * self.chromosome[weightsCount]
            weightsCount += 1
        f1 += self.chromosome[weightsCount]
        weightsCount += 1

        f2 = 0
        for i in range(len(flatPercepts)):
            f2 += flatPercepts[i] * self.chromosome[weightsCount]
            weightsCount += 1
        f2 += self.chromosome[weightsCount]
        weightsCount += 1

        f3 = 0
        for i in range(len(flatPercepts)):
            f3 += flatPercepts[i] * self.chromosome[weightsCount]
            weightsCount += 1
        f3 += self.chromosome[weightsCount]
        weightsCount += 1

        # --- Second hidden layer ---
        f11 = (f1 * self.chromosome[weightsCount]) + (f2 * self.chromosome[weightsCount + 1]) + (
                    f3 * self.chromosome[weightsCount + 2]) + self.chromosome[weightsCount + 3]
        weightsCount += 4
        f12 = (f1 * self.chromosome[weightsCount]) + (f2 * self.chromosome[weightsCount + 1]) + (
                f3 * self.chromosome[weightsCount + 2]) + self.chromosome[weightsCount + 3]
        weightsCount += 4

        # --- Third hidden layer ---
        f21 = (f11 * self.chromosome[weightsCount]) + (f12 * self.chromosome[weightsCount + 1]) \
              + self.chromosome[weightsCount + 2]
        weightsCount += 3
        f22 = (f11 * self.chromosome[weightsCount]) + (f12 * self.chromosome[weightsCount + 1])\
              + self.chromosome[weightsCount + 2]
        weightsCount += 3
        f23 = (f11 * self.chromosome[weightsCount]) + (f12 * self.chromosome[weightsCount + 1])\
              + self.chromosome[weightsCount + 2]
        weightsCount += 3

        # stop lower fitness from going round in circles
        if self.lastAveFitness < 4:
            if np.all(self.lastPercept == percepts):
                randChoice = rand.choice([1, 2, 3])
                if randChoice == 1:
                    f21 += math.pow(high, high)
                elif randChoice == 2:
                    f22 += math.pow(high, high)
                else:
                    f23 += math.pow(high, high)

            if len(set(self.lastActions)) == 1:
                randChoice = rand.choice([1, 2, 3])
                if randChoice == 1:
                    f21 += math.pow(high, high)
                elif randChoice == 2:
                    f22 += math.pow(high, high)
                else:
                    f23 += math.pow(high, high)

        weightArray = np.array([f21, f22, f23])

        # Find max
        maxWeight = np.argmax(weightArray)

        # set previous values
        temp = self.lastActions[0]
        temp2 = self.lastActions[1]
        self.lastActions[0] = maxWeight
        self.lastActions[1] = temp
        self.lastActions[2] = temp2
        self.lastPercept = percepts

        index = maxWeight

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
    # This function should return a tuple consisting of:
    # - a list of the new_population of snakes that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    nPercepts = old_population[0].nPercepts
    actions = old_population[0].actions
    fitness = evalFitness(old_population)

    # Create new population list...
    new_population = list()

    # elitism
    avg_fitness = np.mean(fitness)
    amount = math.floor(avg_fitness)
    x = np.argsort(fitness)[::-1][:amount]
    for y in enumerate(x):
        new_population.append(old_population[y[1]])

    for n in range(N - amount):
        # Create a new snake
        new_snake = Snake(nPercepts, actions)

        # Get parents via tournament selection
        parents = tournament(old_population)
        new_snake.chromosome = newChromosome(parents[0], parents[1])

        new_snake.lastAveFitness = avg_fitness

        # Add the new snake to the new population
        new_population.append(new_snake)

    # At the end you need to compute the average fitness and return it along with your new population

    # write to file for graphing
    file.write(str(avg_fitness) + "\n")

    return new_population, avg_fitness


def newChromosome(p1Chromo, p2Chromo):
    # ------ Multi -------
    # cross over
    splitChromosome1 = np.array_split(p1Chromo.chromosome, p1Chromo.nPercepts)
    splitChromosome2 = np.array_split(p2Chromo.chromosome, p1Chromo.nPercepts)
    for i, n in enumerate(splitChromosome1):
        coinFlip = rand.randint(0, 1)
        if coinFlip == 0:
            splitChromosome2[i] = n
    chromosome = np.concatenate(splitChromosome2)

    # mutation
    mutate = rand.random()
    if mutate < 0.01:
        chromosome = mutateChromosome(chromosome)
    return chromosome


def mutateChromosome(chromosome):
    # Mutate chromosome
    index1 = rand.randint(0, len(chromosome) - 1)
    mutation = (np.random.uniform(low, high))
    chromosome[index1] = mutation
    return chromosome


def tournament(population):
    # Choose random sample
    population_sample = rand.sample(population, tourSampleSize)
    # Choose first parent
    parents = [max(population_sample, key=lambda x: aSnakeFitness(x))]
    # remove first parent from sample
    population_sample.remove(max(population_sample, key=lambda x: aSnakeFitness(x)))
    # choose second parent
    parents.append(max(population_sample, key=lambda x: aSnakeFitness(x)))
    return parents
