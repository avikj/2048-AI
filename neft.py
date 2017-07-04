from sklearn import datasets
from sklearn.metrics import log_loss
import numpy as np
import random
from lib.player import Player
from lib.game import GameBoard
from lib.play_game import play_and_get_score
from datetime import datetime
import pickle 

def main():
  evolved_network = evolve([random_network_params() for i in range(200)], 200, 1000)


##########################################
# Implementation of Neuroevolution Steps #
##########################################
def evolve(initial_networks, population_size, num_generations):
  current_generation_networks = initial_networks
  for generation in range(num_generations):
    start_time = datetime.now()
    network_fitnesses = [fitness(network_params, nrof_simulations=6) for network_params in current_generation_networks]
    output = open('evolved_network.pkl', 'wb')
    pickle.dump(current_generation_networks[np.argmax(network_fitnesses)], output)
    output.close()
    print 'generation: %d, mean fitness %.3f, time: %s' % (generation, np.mean(network_fitnesses), datetime.now()-start_time)
    selected_network_indices = select(network_fitnesses, population_size / 4, 4)
    survivors = [current_generation_networks[i] for i in selected_network_indices]
    current_generation_networks = next_generation(survivors)
    
  return current_generation_networks[np.argmax([accuracy(network) for network in current_generation_networks])]


# apply pairing, crossover, and mutation to compute next generation of networks
def next_generation(survivors):
  next_generation_networks = []
  for i in range(8):
    for mommy, daddy in random_pairs(survivors):
      child = crossover_networks(mommy, daddy)
      next_generation_networks.append(child)
  return next_generation_networks


# return one child, determined by randomly swapping a neuron in parents and choosing more fit result
def crossover_networks(mommy_params, daddy_params):
  child_m = {'W_1': np.copy(mommy_params['W_1']), 'b_1': np.copy(mommy_params['b_1']), 'W_2': np.copy(mommy_params['W_2']), 'b_2': np.copy(mommy_params['b_2'])}
  child_d = {'W_1': np.copy(daddy_params['W_1']), 'b_1': np.copy(daddy_params['b_1']), 'W_2': np.copy(daddy_params['W_2']), 'b_2': np.copy(daddy_params['b_2'])}
  rand = np.random.rand()
  # randomly select layer to perform crossover on
  if rand < 0.7:  # crossover hidden layer
    child_m['W_1'], child_d['W_1'] = crossover_layer_weights(mommy_params['W_1'], daddy_params['W_1'])
  else:  # crossover output layer
    child_m['W_2'], child_d['W_2'] = crossover_layer_weights(mommy_params['W_2'], daddy_params['W_2'])

  if fitness(child_d) > fitness(child_m):
    return mutate_network(child_d)
  else:
    return mutate_network(child_m)


# return two children layers from randomly swapping a neuron/weight in parent layers
def crossover_layer_weights(mommy_W, daddy_W, crossover_type="neuron"):
  if crossover_type == 'neuron':
    # swap weights for random neuron in layer (matrix column)
    rand_column = np.random.randint(mommy_W.shape[1])
    mommy_column = np.copy(mommy_W[:, rand_column])
    daddy_column = np.copy(daddy_W[:, rand_column])
    child_m_W = np.copy(mommy_W)
    child_d_W = np.copy(daddy_W)
    child_m_W[:, rand_column] = daddy_column
    child_d_W[:, rand_column] = mommy_column
    return (child_m_W, child_d_W)


# randomly mutate network weights in each layer
def mutate_network(network_params):
  network_params['W_1'] = mutate_weights(network_params['W_1'])
  network_params['W_2'] = mutate_weights(network_params['W_2'])
  return network_params


# randomly mutate weights in a layer
def mutate_weights(W):
  if np.random.rand() < 0.2:
    rand = np.random.rand()
    i = np.random.randint(W.shape[0])
    j = np.random.randint(W.shape[1])
    if rand < 0.5:
      W[i][j] *= np.random.uniform(0.5, 1.5)
    elif rand < 0.9:
      W[i][j] += np.random.uniform(-1, 1)
    else:
      W[i][j] *= -1
  return W


# select n networks to be used as parents in the next generation, using tournament selection algorithm
def select(network_fitnesses, n, k=10):
  selected_network_indices = []
  for i in range(n):
    competitors_indices = np.random.choice(len(network_fitnesses), k, replace=False)
    winner_index = competitors_indices[np.argmax(np.array(network_fitnesses)[competitors_indices])]
    selected_network_indices.append(winner_index)
  return np.array(selected_network_indices)


# compute fitness of network as average score after 10 simulations
def fitness(network_params, nrof_simulations=1):
  scores = [play_and_get_score(NNPlayer(network_params)) for i in range(nrof_simulations)]
  return np.mean(scores)

# randomly pair networks in list of tuples
def random_pairs(networks):
  random.shuffle(networks)
  result = []
  for i in range(0, len(networks), 2):
    result.append((networks[i], networks[i + 1]))
  return result


#########################################
# Neural Network Definition and Metrics #
#########################################

def sigmoid(X):
  return 1 / (1 + np.exp(-X))


def softmax(X):
  return np.divide(np.exp(X), np.sum(np.exp(X)))

def random_network_params():
  INPUT_LAYER_SIZE = 16
  HIDDEN_LAYER_SIZE = 40
  OUTPUT_LAYER_SIZE = 4
  stddev = 0.1
  W_1 = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * stddev
  b_1 = np.random.randn(HIDDEN_LAYER_SIZE) * stddev
  W_2 = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * stddev
  b_2 = np.random.randn(OUTPUT_LAYER_SIZE) * stddev
  return {'W_1': W_1, 'b_1': b_1, 'W_2': W_2, 'b_2': b_2}

def predict_move_proba(board_flat, network_params):
  h = sigmoid(np.matmul(board_flat, network_params['W_1']) + network_params['b_1'])
  y = softmax(np.matmul(h, network_params['W_2']) + network_params['b_2'])
  return y

# 0: w, 1: a, 2: s, 3: d
def predict_move(board_flat, network_params):
  return np.argmax(predict_move_proba(board_flat, network_params))

##########################################
# Neural Network Player Class Definition #
##########################################
class NNPlayer(Player):
  moves = [GameBoard.MOVE_UP, GameBoard.MOVE_LEFT, GameBoard.MOVE_DOWN, GameBoard.MOVE_RIGHT,]
  def __init__(self, network_params):
    self.params = network_params
  def get_move(self, board):
    move_proba = predict_move_proba(board.get_state().flatten(), self.params)
    for i in np.argsort(move_proba)[::-1]:
      if board.can_move(self.moves[i]):
        return self.moves[i]

if __name__ == '__main__':
  main()
