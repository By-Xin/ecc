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
C_UCB = 0.1  # UCB exploration parameter

# Lists to record fitness statistics
best_fitness_history = []
mean_fitness_history = []
std_fitness_history = []

# Fitness evaluation with UCB-enhanced action selection
def eval_genomes(genomes, config):
    env = gym.make(GAME)
    global best_fitness_history, mean_fitness_history, std_fitness_history
    fitness_values = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, _ = env.reset()
        genome.fitness = 0.0
        done = False
        t = 0
        n_actions = env.action_space.n
        # UCB statistics per episode
        action_counts = np.zeros(n_actions, dtype=np.int32)
        action_values = np.zeros(n_actions, dtype=np.float32)

        while not done:
            t += 1
            # get network preferences
            prefs = net.activate(obs)
            # compute UCB score: network preference + exploration bonus
            bonus = C_UCB * np.sqrt(np.log(t + 1) / (action_counts + 1))
            ucb_scores = np.array(prefs) + bonus
            # select action
            action = int(np.argmax(ucb_scores))
            # step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # update genome fitness
            genome.fitness += reward
            # update UCB stats
            action_counts[action] += 1
            # incremental update of action value estimate
            prev_value = action_values[action]
            action_values[action] += (reward - prev_value) / action_counts[action]
            obs = next_obs

        fitness_values.append(genome.fitness)

    env.close()

    # Record statistics
    best_fitness_history.append(np.max(fitness_values))
    mean_fitness_history.append(np.mean(fitness_values))
    std_fitness_history.append(np.std(fitness_values))


if __name__ == '__main__':
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    # disable early stopping
    config.fitness_threshold = float('inf')

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    p.add_reporter(stats_reporter)
    p.add_reporter(neat.Checkpointer(CHECKPOINT_INTERVAL,
                                     filename_prefix=f'{CHECKPOINT_DIR}/neat-checkpoint-'))

    # run training
    winner = p.run(eval_genomes, GENERATIONS)

    # save best genome
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print("Training complete. Best genome saved to 'best_genome.pkl'")

    # plot fitness over generations
    recorded = len(best_fitness_history)
    gens = np.arange(1, recorded + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(gens, best_fitness_history, marker='o', label='Best Fitness')
    plt.plot(gens, mean_fitness_history, marker='s', label='Mean Fitness')
    plt.fill_between(
        gens,
        np.array(mean_fitness_history) - np.array(std_fitness_history),
        np.array(mean_fitness_history) + np.array(std_fitness_history),
        alpha=0.2, label='Fitness StdDev'
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("NEAT Fitness with UCB over Generations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("neat_fitness_plot_ucb.png")
    plt.show()
