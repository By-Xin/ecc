import os
import gym
import neat
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configuration
GAME = 'CartPole-v1'
CONFIG_PATH = './config'
GENERATIONS = 50

CHECKPOINT_INTERVAL = 5
CHECKPOINT_DIR = './checkpoints'

# Lists to record fitness statistics
best_fitness_history = []
mean_fitness_history = []
std_fitness_history = []

# Fitness evaluation function for NEAT
def eval_genomes(genomes, config):
    env = gym.make(GAME)
    global best_fitness_history, mean_fitness_history, std_fitness_history
    fitness_values = []

    for genome_id, genome in genomes:
        # Create neural network from genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset()[0]
        genome.fitness = 0.0
        done = False

        # Run one episode and accumulate reward
        while not done:
            action_values = net.activate(observation)
            action = int(np.argmax(action_values))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            genome.fitness += reward

        fitness_values.append(genome.fitness)

    env.close()

    # Record generation-wise statistics
    best_fitness_history.append(np.max(fitness_values))
    mean_fitness_history.append(np.mean(fitness_values))
    std_fitness_history.append(np.std(fitness_values))


if __name__ == '__main__':
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    # Disable early stopping by overriding fitness threshold
    config.fitness_threshold = float('inf')

    # Create the population and add reporters
    p = neat.Population(config)
    stats_reporter = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats_reporter)
    p.add_reporter(neat.Checkpointer(CHECKPOINT_INTERVAL, filename_prefix=f'{CHECKPOINT_DIR}/neat-checkpoint-'))

    # Run NEAT training
    winner = p.run(eval_genomes, GENERATIONS)

    # Save the best genome
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print("Training complete. Best genome saved to 'best_genome.pkl'")

    # Determine number of recorded generations
    recorded_gens = len(best_fitness_history)
    if recorded_gens == 0:
        print("No fitness data recorded. Check that eval_genomes was called correctly.")
    else:
        generations = np.arange(1, recorded_gens + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(generations, best_fitness_history, marker='o', label='Best Fitness')
        plt.plot(generations, mean_fitness_history, marker='s', label='Mean Fitness')
        plt.fill_between(
            generations,
            np.array(mean_fitness_history) - np.array(std_fitness_history),
            np.array(mean_fitness_history) + np.array(std_fitness_history),
            alpha=0.2, label='Fitness StdDev'
        )
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("NEAT Fitness over Generations")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neat_fitness_plot.png")
        plt.show()
