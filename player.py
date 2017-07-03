from abc import ABCMeta, abstractmethod
import random


# Abstract player which returns move (WASD) given board
class Player:
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_move(self, board):
    pass


# Human player class which gets move as user input from stdin
class HumanPlayer(Player):
  def get_move(self, board):
    move = ''
    while len(move) != 1 or move not in 'wasd':
      move = raw_input('Enter move (WASD)>').lower()
    return move


# Random player class which randomly selects a move
class RandomPlayer(Player):
  def get_move(self, board):
    return random.choice('wasd')
