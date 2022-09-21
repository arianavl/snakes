__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy
import numpy as np
import random as rand
import math

# if they are all zeros assign a bias to a random action to stop it from going in circles

# np.dot(chromosome, percepts)

# I think the problem is that its random but eventually the more
# common outputs are reproduced cause slightly worse fitnesses
# changing all ones to -1
# changing fitness to not care about amount of turns as much

agentName = "<my_agent>"
perceptFieldOfVision = 3   # Choose either 3,5,7 or 9
perceptFrames = 1           # Choose either 1,2,3 or 4
trainingSchedule = [("random", 300), ("self", 0), ("random", 0)]
file = open("sample.txt", "w")
hiddenFunctionSize = 8
hiddenFunctionSizeWeights = 12

# population_size = 10
# trainingSchedule = None

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
        self.population_size = 3
        self.lastPercepts = []
        chromosome = \
            [np.random.uniform(-50, 50) for i in range(hiddenFunctionSizeWeights + (nPercepts*3) + hiddenFunctionSize)]
        self.chromosome = np.array(chromosome)

        #
        # self.chromosome = np.array(one_chromosome)
        # print("self.chromosome(original random chromosome): \n" + str(self.chromosome))

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

        # print(percepts)
        # print(percepts.flatten())

        flatPercepts = percepts.flatten()
        for i, n in enumerate(flatPercepts):
            if n == -1:
                flatPercepts[i] = -10
            elif n == 1:
                flatPercepts[i] = -10
            elif n == 2:
                flatPercepts[i] = 10
        # print(percepts)
        # print(flatPercepts)

        weigthsCount = 0

        #### First hidden layer ####
        f1 = 0
        for i in range(len(flatPercepts)):
            f1 += flatPercepts[i] * self.chromosome[weigthsCount]
            weigthsCount += 1
        ## Increase to pass bias value
        # f1 += self.chromosome[weigthsCount]
        f1 += np.random.uniform(-1, 1)
        weigthsCount += 1
        # print(f1)

        f2 = 0
        for i in range(len(flatPercepts)):
            f2 += flatPercepts[i] * self.chromosome[i + self.nPercepts + 1]
            weigthsCount += 1
        ## Increase to pass bias value
        # f2 += self.chromosome[weigthsCount]
        f2 += np.random.uniform(-1, 1)
        weigthsCount += 1

        # print(f2)

        f3 = 0
        for i in range(len(flatPercepts)):
            f3 += flatPercepts[i] * self.chromosome[i + self.nPercepts*2 + 2]
            weigthsCount += 1
        ## Increase to pass bias value
        # f3 += self.chromosome[weigthsCount]
        f3 += np.random.uniform(-1, 1)
        weigthsCount += 1
        # print(f3)
        ###### evaluation function #####

        #### Second hidden layer ####
        f11 = (f1*self.chromosome[weigthsCount]) + (f2*self.chromosome[weigthsCount+1]) + (f3*self.chromosome[weigthsCount+2]) + np.random.uniform(-1, 1)
        weigthsCount += 4
        f12 = (f1 * self.chromosome[weigthsCount]) + (f2 * self.chromosome[weigthsCount + 1]) + (
                    f3 * self.chromosome[weigthsCount + 2]) + np.random.uniform(-1, 1)
        weigthsCount += 4

        #### Third hidden layer ####
        f21 = (f11 * self.chromosome[weigthsCount]) + (f12 * self.chromosome[weigthsCount+1]) + np.random.uniform(-1, 1)
        weigthsCount += 3
        f22 = (f11 * self.chromosome[weigthsCount]) + (f12 * self.chromosome[weigthsCount + 1]) + np.random.uniform(-1, 1)
        weigthsCount += 3
        f23 = (f11 * self.chromosome[weigthsCount]) + (f12 * self.chromosome[weigthsCount + 1]) + np.random.uniform(-1, 1)
        weigthsCount += 3



        weightArray = np.array([f21, f22, f23])
        # print("\nweightArray: " + str(weightArray))
        maxWeight = np.argmax(weightArray)
        # print("\nmaxWeight: " + str(maxWeight))
        index = maxWeight
        # print(index)

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
        # fitness[n] = maxSize + maxTurns

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
    # population_size = old_population[0].population_size
    # population_size = population_size * 0.9
    fitness = evalFitness(old_population)



    # Create new population list...
    new_population = list()
    print("fitness: " + str(fitness))


    x = np.argsort(fitness)[::-1][:5]
    for y in enumerate(x):
        new_population.append(old_population[y[1]])
        print(y[1])
    print("Indices:", x)
    print("fitnesses: " + str(fitness))
    print("population_size: " + str(old_population[0].population_size))

    # new_population.append(old_population[index1])
    # new_population.append(old_population[index2])
    # print("\nElitism new population: \n1: " + str(aSnakeFitness(new_population[0])) + "\n 2: " + str(aSnakeFitness(new_population[1])))
    for n in range(N-5):

        # Create a new snake
        new_snake = Snake(nPercepts, actions)
        # if old_population[0].population_size <= 9:
        #     new_snake.population_size = old_population[0].population_size + 0.02
        # else:
        #     new_snake.population_size = 9
        # print("population_size: " + str(new_snake.population_size))

        parents = tournament(old_population)

        # parent1fitness = roulette_wheel_selection(old_population, fitness)
        # parent2fitness = roulette_wheel_selection(old_population, fitness)
        # parent1 = old_population[parent1fitness]
        # parent2 = old_population[parent2fitness]

        new_snake.chromosome = newChromosome(parents[0], parents[1])

        # Add the new snake to the new population
        new_population.append(new_snake)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    file.write(str(avg_fitness) + "\n")

    return new_population, avg_fitness


def roulette_wheel_selection(population, fitness):

    total_fit = fitness.sum()
    # print("fitness" + str(fitness))
    prob_list = fitness / total_fit
    # print("prob_list" + str(prob_list))

    p = fitness/np.sum(fitness)
    # print("p" + str(p))

    # Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population))), p=prob_list, replace=True)

   # print("a: " + str(progenitor_list_a))

    return progenitor_list_a


def newChromosome(p1Chromo, p2Chromo):

    splitChromosome1 = np.array_split(p1Chromo.chromosome, p1Chromo.nPercepts)
    splitChromosome2 = np.array_split(p2Chromo.chromosome, p1Chromo.nPercepts)
    for i, n in enumerate(splitChromosome1):
        coinFlip = rand.randint(0, 1)
        if coinFlip == 0:
            splitChromosome2[i] = n
    print(splitChromosome1)
    print(splitChromosome2)
    chromosome = np.concatenate(splitChromosome2)
    print(chromosome)

    # chromosome = []
    # for x in range(p1Chromo.nPercepts + 1):
    #
    #
    # while(x < len(chromosome)):
    #     chromosome = np.concatenate([chromosome, p1Chromo.chromosome[x : x+])
    # # one
    # for n, x in enumerate(p1Chromo.chromosome):
    #     coinFlip = rand.randint(0, 1)
    #     if coinFlip == 0:
    #         chromosome[n] = x

    # x = 0
    # y = 0
    # while(x < len(chromosome)):
    #     x += np.randint(1, perceptFieldOfVision*perceptFieldOfVision)

    #
    # # print("\nchromosome after (child chromosome):\n " + str(chromosome))

    # mutate??
    mutate = rand.random()
    # print("mutate: " + str(mutate))
    if mutate < 0.001:
        print("mutated")
        chromosome = mutateChromosome(chromosome)
    # #     print("\nchanged chromo (shoudl be same as after mutation chromosome):\n " + str(chromosome))

    return chromosome


def mutateChromosome(chromosome):
    index1 = rand.randint(0, len(chromosome)-1)

    mutation = []
    # for x in range(3):
    #     one_chromosome = []
    #     for y in range(perceptFieldOfVision):
    #         one_chromosome.append((rand.random()))
    #     mutation.append(one_chromosome)
    #
    # for y in range(perceptFieldOfVision):
    #     mutation.append((np.random.uniform(-1, 1)))

    mutation = (np.random.uniform(-1, 1))

        # mutation.append(rand.randint(1, 11))
    # chromosome[index1][index2] = [rand.random(), rand.random(), rand.random()]
    # print("mutation: " + str(mutation))
    # print("\nBefore mutation chromosome: \n" + str(chromosome))
    # chromosome[index1] = mutation
    # chromosome[index1][index2] = mutation
    chromosome[index1] = mutation

    # print("\n after mutated: \n" + str(chromosome))
    return chromosome


# Use many tournaments to get parents
def tournament(population):
    # Choose random sample
    population_sample = rand.sample(population, 3)
    # Choose first parent
    parents = [max(population_sample, key=lambda x: aSnakeFitness(x))]
    population_sample.remove(max(population_sample, key=lambda x: aSnakeFitness(x)))
    # choose second parent
    parents.append(max(population_sample, key=lambda x: aSnakeFitness(x)))
    print("Parent fitness: " + str(aSnakeFitness(parents[0])) + " Second parent: " + str(aSnakeFitness(parents[1])))
    return parents

