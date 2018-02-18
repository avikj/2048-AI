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
  child_m = []
  child_d = []
  for i in range(len(mommy_params)):
    child_m.append({
      'W': np.copy(mommy_params[i]['W']),
      'b': np.copy(mommy_params[i]['b'])
    })
    child_d.append({
      'W': np.copy(daddy_params[i]['W']),
      'b': np.copy(daddy_params[i]['b'])
    })
  layer_to_crossover = np.random.choice(range(len(mommy_params)))
  # randomly select layer to perform crossover on
  child_m[layer_to_crossover], child_d[layer_to_crossover] = crossover_layer_weights_bias(mommy_params[layer_to_crossover], daddy_params[layer_to_crossover])

  if fitness(child_d) > fitness(child_m):
    return mutate_network(child_d)
  else:
    return mutate_network(child_m)


# return two children layers from randomly swapping a neuron in parent layers
def crossover_layer_weights_bias(mommy_layer, daddy_layer, crossover_type="neuron"):
  if crossover_type == 'neuron':
    # swap weights for random neuron in layer (matrix column)
    rand_column = np.random.randint(mommy_layer['W'].shape[1])
    mommy_column_W = np.copy(mommy_layer['W'][:, rand_column])
    daddy_column_W = np.copy(daddy_layer['W'][:, rand_column])
    child_m_W = np.copy(mommy_layer['W'])
    child_d_W = np.copy(daddy_layer['W'])
    child_m_W[:, rand_column] = daddy_column_W
    child_d_W[:, rand_column] = mommy_column_W

    mommy_column_b = np.copy(mommy_layer['b'][rand_column])
    daddy_column_b = np.copy(daddy_layer['b'][rand_column])
    child_m_b = np.copy(mommy_layer['b'])
    child_d_b = np.copy(daddy_layer['b'])
    child_m_b[rand_column] = daddy_column_b
    child_d_b[rand_column] = mommy_column_b
    return ({'W':child_m_W, 'b':child_m_b}, {'W':child_d_W, 'b':child_d_b})


# randomly mutate network weights in each layer
def mutate_network(network_params):
  for i in range(len(network_params)):
    network_params[i]['W'] = mutate_weights(network_params[i]['W'])
    network_params[i]['b'] = mutate_bias(network_params[i]['b'])
  return network_params

# randomly mutate weights in a layer
def mutate_weights(W):
  if np.random.rand() < 0.1:
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

def mutate_bias(b):
  if np.random.rand() < 0.1:
    rand = np.random.rand()
    i = np.random.randint(b.size)
    if rand < 0.5:
      b[i] *= np.random.uniform(0.5, 1.5)
    elif rand < 0.9:
      b[i] += np.random.uniform(-1, 1)
    else:
      b[i] *= -1
  return b
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
  NUM_HIDDEN_LAYERS = 3
  INPUT_LAYER_SIZE = 16
  HIDDEN_LAYER_SIZE = 40
  OUTPUT_LAYER_SIZE = 4
  stddev = 0.1
  params = []
  W_i = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * stddev
  b_i = np.random.randn(HIDDEN_LAYER_SIZE) * stddev
  params.append({'W': W_i, 'b': b_i})

  for i in range(NUM_HIDDEN_LAYERS-1):
    W_h = np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * stddev
    b_h = np.random.randn(HIDDEN_LAYER_SIZE) * stddev
    params.append({'W': W_h, 'b': b_h})

  W_o = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * stddev
  b_o = np.random.randn(OUTPUT_LAYER_SIZE) * stddev
  params.append({'W': W_o, 'b': b_o})
  return params

def predict_move_proba(board_flat, network_params):
  current_layer = board_flat
  for i in range(len(network_params)-1):
    next_layer = sigmoid(np.matmul(current_layer, network_params[i]['W']) + network_params[i]['b'])
    current_layer = next_layer
  y = softmax(np.matmul(current_layer, network_params[-1]['W']) + network_params[-1]['b'])
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
