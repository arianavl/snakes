# Snakes
## A genetic algorithm to develope a population of snakes over many generations to intellgently play a survival game
The game is played on a two-dimensional plane where the snakes, opponent snakes from a different population, and red squares representing food, are generated randomly at random positions. To begin with, all the snakes are only two lengths long and they can either move to their left, right or straight ahead. If the snake’s head runs into another snake or itself it dies and if the snake eats a piece of food its size will increase by one.
Each snake is created with its own chromosome that is responsible for deciding what action to take in relation to its perceived environment. Therefore, the task was to create a genetic algorithm that over many new generations, would create a population with chromosomes that choose intelligent actions and, therefore, play better games.

my_agent.py

