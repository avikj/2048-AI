from abc import ABCMeta, abstractmethod
from game import GameBoard
import random


# Abstract player which returns move given board
class Player:
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_move(self, board):
    pass


# Human player class which gets move as user input from stdin
class HumanPlayer(Player):
  moves = {
      'w': GameBoard.MOVE_UP,
      'a': GameBoard.MOVE_LEFT,
      's': GameBoard.MOVE_DOWN,
      'd': GameBoard.MOVE_RIGHT,
  }
  def get_move(self, board):
    while True:
      move_letter = ''
      while len(move_letter) != 1 or move_letter not in 'wasd':
        move_letter = raw_input('Enter move (WASD)>').lower()
      move = moves[move_letter]
      if board.can_move(move):
        return move


# Random player class which randomly selects a move
class RandomPlayer(Player):
  def get_move(self, board):
    while True:
      move =  random.choice([GameBoard.MOVE_UP,GameBoard.MOVE_LEFT,GameBoard.MOVE_DOWN,GameBoard.MOVE_RIGHT])
      if board.can_move(move):
        return move