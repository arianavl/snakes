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
# population_size = 10
trainingSchedule = None

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
            [[[np.random.uniform(-1, 1) for i in range(perceptFieldOfVision)]
              for i in range(perceptFieldOfVision)] for i in range(3)]
        # for i in range(3):
        #     one_chromosome = []
        #     for n in range(int(nPercepts/perceptFieldOfVision)):
        #         one_percept = []
        #         for y in range(perceptFieldOfVision):
        #             one_percept.append((np.random.uniform(-1, 1)))
        #         one_chromosome.append(one_percept)
        #     chromosome[i] = one_chromosome
        self.chromosome = np.array(chromosome)
        # # self.biases = [rand.uniform(0, 0.5), rand.uniform(0, 0.5), rand.uniform(0, 0.5)]
        # self.biases = [rand.random(), rand.random(), rand.random()]

        # one_chromosome = []
        # for n in range(int(nPercepts / perceptFieldOfVision)):
        #     one_percept = []
        #     for y in range(perceptFieldOfVision):
        #         one_percept.append(rand.random())
        #     one_chromosome.append(one_percept)
        #
        # self.chromosome = np.array(one_chromosome)
        # print("self.chromosome(original random chromosome): \n" + str(self.chromosome))
        # print("self.chromosome bais: " + str(self.chromosome[3]))

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
        # bias = 0
        # if self.lastPercepts != []:
        #
        #     comparison = self.lastPercepts == percepts
        #     # print(comparison)
        #     if comparison.all():
        #         bias = np.random.uniform(-1.5, 1.5)
        #         print("true")
        # self.lastPercepts = percepts

        # for x in range(len(percepts)):
        #     pre_index_a = self.chromosome[0] * percepts[x]
        #     # print("perceptsA: " + str(percepts[x]))
        #     # print("chromosomeA: " + str(self.chromosome[0]))
        #     pre_index_b = self.chromosome[1] * percepts[x]
        #     # print("perceptsB: " + str(percepts[x]))
        #     # print("chromosomeB: " + str(self.chromosome[1]))
        #     pre_index_c = self.chromosome[2] * percepts[x]
        #     # print("perceptsC: " + str(percepts[x]))
        #     # print("chromosomeC: " + str(self.chromosome[2]))


        pre_index_a = self.chromosome[0] * percepts
        # print("perceptsA: " + str(percepts))
        # print("chromosomeA: " + str(self.chromosome[0]))
        # print("pre_index_a: " + str(pre_index_a))
        pre_index_b = self.chromosome[1] * percepts
        # print("perceptsB: " + str(percepts[x]))
        # print("chromosomeB: " + str(self.chromosome[1]))
        # print("pre_index_b: " + str(pre_index_a))
        pre_index_c = self.chromosome[2] * percepts

        # pre_index_a = pre_index_a * percepts
        # pre_index_b = pre_index_b * percepts
        # pre_index_c = pre_index_c * percepts

        # print("perceptsC: " + str(percepts[x]))
        # print("chromosomeC: " + str(self.chromosome[2]))
        # print("pre_index_c: " + str(pre_index_a))

        # print(sum(pre_index_c))
        # print(percepts[0][1][1])

        # print(percepts)

        # count = 0
        # for n, x in enumerate(pre_index_a):
        #     # print("x: " + str(x))
        #     for y, i in enumerate(x):
        #         # print("i " + str(i))
        #         if i > 0:
        #             count += 1
        ######################
        # weight_a = (np.sum(pre_index_a) + (rand.random()))
        # weight_b = (np.sum(pre_index_b) + (rand.random()))
        # weight_c = (np.sum(pre_index_c) + (rand.random()))

        # weight_a = ((pre_index_a.sum()).sum() + (rand.random()))
        # weight_b = ((pre_index_b.sum()).sum() + (rand.random()))
        # weight_c = ((pre_index_c.sum()).sum() + (rand.random()))

        # weight_a = (pre_index_a.sum()).sum() + np.random.uniform(-1, 1)
        # weight_b = (pre_index_b.sum()).sum() + np.random.uniform(-1, 1)
        # weight_c = (pre_index_c.sum()).sum() + np.random.uniform(-1, 1)


        ############################
        weight_a = (pre_index_a.sum()).sum()
        weight_b = (pre_index_b.sum()).sum()
        weight_c = (pre_index_c.sum()).sum()

        weight_a = (pre_index_a.sum()).sum() + (rand.uniform(-1, 1))
        weight_b = (pre_index_b.sum()).sum() + (rand.uniform(-1, 1))
        weight_c = (pre_index_c.sum()).sum() + (rand.uniform(-1, 1))

        # weight_a = (pre_index_a.sum()).sum() + bias
        # weight_b = (pre_index_b.sum()).sum() + bias
        # weight_c = (pre_index_c.sum()).sum() + bias
        # print(weight_a)
        weightArray = np.array([weight_a, weight_b, weight_c])
        # print("\nweightArray: " + str(weightArray))
        maxWeight = np.argmax(weightArray)
        # print("\nmaxWeight: " + str(maxWeight))
        index = maxWeight
        # print(index)
        # print("index 2 (should be same): " + str(index))

        # weight_a = (pre_index_a.sum() + (rand.uniform(0, 0.5)))
        # weight_b = (pre_index_b.sum() + (rand.uniform(0, 0.5)))
        # weight_c = (pre_index_c.sum() + (rand.uniform(0, 0.5)))

        # weight_a = (np.sum(pre_index_a) + (rand.uniform(0, 0.5)))
        # weight_b = (np.sum(pre_index_b) + (rand.uniform(0, 0.5)))
        # weight_c = (np.sum(pre_index_c) + (rand.uniform(0, 0.5)))

        # weight_a = weight_a / perceptFieldOfVision
        # weight_a = weight_a / count
        # weight_b = weight_b / perceptFieldOfVision
        # weight_b = weight_b / count
        # weight_c = weight_c / perceptFieldOfVision
        # weight_c = weight_c / count
            # weight_a += rand.randint(0, 2)
            # weight_b += rand.randint(0, 2)
            # weight_c += rand.randint(0, 2)




        # # print(self.chromosome)
        # print("preIndexA" + str(pre_index_a))
        # count = 0
        # # for x in pre_index:
        # #    count += x
        # print("weight_c: " + str(weight_c))
        # print("weight_a: " + str(weight_a))
        # print("weight_b: " + str(weight_b))
        #
        # if weight_c > weight_a and weight_c > weight_b:
        #     index = 0
        # elif weight_b > weight_a and weight_b > weight_c:
        #     index = 1
        # else:
        #     index = 2

        # if index == 0 and self.habit_0 > self.habit_1/2 or self.habit_0 > self.habit_2/2:
        #     index = rand.randint(0, 2)
        #     print(index)
        # elif index == 1 and self.habit_1 > self.habit_0/2 or self.habit_1 > self.habit_2/2:
        #     index = rand.randint(1, 2)
        #     print(index)
        # elif index == 2 and self.habit_2 > self.habit_1/2 or self.habit_2 > self.habit_0/2:
        #     index = rand.randint(0, 2)
        #     print(index)

        # print(percepts) [[[0 0 0]
        #                   [0 1 0]
        #                   [1 1 0]]]

        # if len(percepts) == 1:
        #     pre_index_a = self.chromosome * percepts
        # else:
        # for x in range(len(percepts)):
        #     pre_index_a = self.chromosome * percepts[x]
        #     # pre_index_a = self.chromosome[x] * percepts[x]
        #
        # # print("perceptsA: " + str(percepts))
        # # print("pre_index_a: " + str(pre_index_a))
        # # print("chromosomeA: " + str(self.chromosome[0]))
        #
        # weight_a = np.sum(pre_index_a)

        # count = 0
        # for n, x in enumerate(pre_index_a):
        #     # print("x: " + str(x))
        #     for y, i in enumerate(x):
        #         # print("i " + str(i))
        #         if i > 0:
        #             count += 1
        # weight_a = weight_a/perceptFieldOfVision
        # weight_a = weight_a/count

            # weight_a += rand.rand(0, 0.5)

            # weight_a += rand.random()

        # print(self.chromosome)
        # print("preIndexA" + str(pre_index_a))
        # count = 0
        # for x in pre_index:
        #    count += x





        # mindex = np.random.randint(low=0, high=len(self.actions))
        # print("\nActions: " + str(self.actions))
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
    chromosome = p2Chromo.chromosome

    # one
    for n, x in enumerate(p1Chromo.chromosome):
        coinFlip = rand.randint(0, 1)
        if coinFlip == 0:
            chromosome[n] = x

    #two
    # for n, x in enumerate(p1Chromo.chromosome):
    #     # print("chromosome[n]: " + str(chromosome[n]))
    #     # print("x: " + str(x))
    #     for y, w in enumerate(x):
    #     # print("w: " + str(w))
    #         coinFlip = rand.randint(0, 1)
    #         if coinFlip == 0:
    #             # print("p1Chromo.chromosome[n][y]: " + str(p1Chromo.chromosome[n][y]))
    #             chromosome[n][y] = w
    #             # print("p1Chromo.chromosome[n][y] after chance of changing: " + str(p1Chromo.chromosome[n][y]))

    #Three
    # # print("\nchromosome before child creation: \n" + str(chromosome))
    # # pop = pop + (pop * 0.3)
    # for n, x in enumerate(p1Chromo.chromosome):
    #     # print("\nchromosome[n] before child creation: " + str(chromosome[n]))
    #     # print("x: " + str(x))
    #     for y, w in enumerate(x):
    #         for z, k in enumerate(w):
    #     # print("w: " + str(w))
    #             coinFlip = rand.randint(0, 1)
    #             # print("p1Chromo.chromosome[n][y]: " + str(chromosome[n][y][z]))
    #             if coinFlip == 0:
    #                 # print("K: " + str(k))
    #                 # print("Explicate K: " + str(p1Chromo.chromosome[n][y][z]))
    #                 # print("Explicate K p2Chromo: " + str(p2Chromo.chromosome[n][y][z]))
    #                 chromosome[n][y][z] = k
    #                 # print("p1Chromo.chromosome[n][y] after chance of changing: " + str(chromosome[n][y][z]))
    #
    #     # print("chromosome[n] after: " + str(chromosome[n]))
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
    index1 = rand.randint(0, 2)
    index2 = rand.randint(0, 2)
    index3 = rand.randint(0, 2)

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
    chromosome[index1][index2][index3] = mutation

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

