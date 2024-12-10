import math
import numpy as np

def fitness(x, y, z):
    result = (2 * x * z * math.exp(-x) -2 * y**3 + y**2 - 3 * z ** 3)
    return result

def generate_pop(min, max, pop_size):
    return np.random.uniform(min, max, (pop_size, 3))

def tournament_selection(pop, fitness_values, tournament_size=10):
    selected = np.random.choice(len(pop), tournament_size, replace=False)
    selected_fitness = fitness_values[selected]
    winner_idx = selected[np.argmax(selected_fitness)]
    return pop[winner_idx]

def crossover(parrent1, parrent2, crossover_rate=0.8):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parrent1))
        child1 = np.concatenate([parrent1[:crossover_point], parrent2[crossover_point:]])
        child2 = np.concatenate([parrent2[:crossover_point], parrent1[crossover_point:]])
        return child1, child2
    else:
        return parrent1, parrent2
    
def mutant(indevisual, mutent_rate=0.1):
    if np.random.rand() < mutent_rate:
        mutent_idx = np.random.randint(len(indevisual))
        mutent_val = np.random.uniform(0, 100)
        indevisual[mutent_idx]=mutent_val
    return indevisual

population = generate_pop(0, 100, 100)
generation_epoc = 5000
best_solution = None
best_fitness = -np.inf
for _ in range(generation_epoc):
    fitness_values = np.array([fitness(p[0], p[1], p[2]) for p in population])
    if fitness_values[np.argmax(fitness_values)] > best_fitness:
        best_fitness = fitness_values[np.argmax(fitness_values)]
        best_solution = population[np.argmax(fitness_values)]
        
    new_popultion = []
    
    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, fitness_values)
        parent2 = tournament_selection(population, fitness_values)
        
        child1, child2 = crossover(parent1, parent2)
        
        child1 = mutant(child1)
        child2 = mutant(child2)
        
        new_popultion.append(child1)
        new_popultion.append(child2)
        
    population = np.array(new_popultion)
print(best_solution, best_fitness)