import numpy as np


# Represents a 2048 game board. Provides functions for manipulating and playing on board.
class GameBoard:
  # Standard game parameters
  NROF_START_TILES = 2
  BOARD_SIZE = 4

  # Moves are represented as [di, dj]
  MOVE_UP = np.array([-1, 0])
  MOVE_RIGHT = np.array([0, 1])
  MOVE_DOWN = np.array([1, 0])
  MOVE_LEFT = np.array([0, -1])

  def __init__(self):
    self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
    self.score = 0
    for i in range(self.NROF_START_TILES):
      self.add_tile()

  # Randomly adds a 2 or 4 tile to the board, if possible.
  # True if tile was added to board.
  def add_tile(self):
    if self.is_game_over():
      raise RuntimeError('Cannot add tile to full board.')
      return False
    else:
      tile_value = 1 if np.random.rand() < 0.9 else 2
      cells_i, cells_j = self._get_unoccupied_cells()
      choice = np.random.randint(cells_i.size)
      self.board[cells_i[choice], cells_j[choice]] = tile_value
      return True

  def move(self, direction):
    traversals_i, traversals_j = self._build_traversals(direction)
    did_merge = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=bool)
    moved = False
    for tile_i in traversals_i:
      for tile_j in traversals_j:
        tile_value = self.board[tile_i, tile_j]
        if tile_value != 0:
          farthest_open_cell, next_occupied_cell = self._find_farthest_position((tile_i, tile_j), direction)

          # Merge equal tiles; only one merger per row traversal?
          if self.is_within_bounds(next_occupied_cell) and self.board[next_occupied_cell] == tile_value and not did_merge[next_occupied_cell]:
            self.board[next_occupied_cell] = tile_value + 1
            did_merge[next_occupied_cell] = True
            self.board[tile_i, tile_j] = 0

            # Update the score
            self.score += np.power(2, self.board[next_occupied_cell])
            moved = True
          # If can't merge, move the tile to farthest open cell
          else:
            self.board[tile_i, tile_j] = 0
            self.board[farthest_open_cell] = tile_value
            if (tile_i, tile_j) != farthest_open_cell:
              moved = True

    return moved

  def can_move(self, direction):
    return self.copy().move(direction)

  def is_within_bounds(self, cell):
    return cell[0] < self.BOARD_SIZE and cell[1] < self.BOARD_SIZE and cell[0] >= 0 and cell[1] >= 0

  def is_board_full(self):
    return np.count_nonzero(self.board) == self.board.size

  def is_game_over(self):
    if not self.is_board_full():
      return False
    return not any([self.can_move(direction) for direction in [self.MOVE_UP, self.MOVE_RIGHT, self.MOVE_DOWN, self.MOVE_LEFT]])

  def current_score(self):
    return int(self.score)

  def get_state(self):
    return self.board

  def copy(self):
    ret = GameBoard()
    ret.load_state(self.board)
    return ret

  def load_state(self, board):
    if board.shape != (self.BOARD_SIZE, self.BOARD_SIZE):
      raise ValueError('Board must have shape %s' % (self.BOARD_SIZE, self.BOARD_SIZE))
    else:
      self.board = board.copy()

  #####################
  # Utility Functions #
  #####################
  # Returns value to display (power of 2, blank for ones)
  def _display_value_at(self, cell):
    return str(int(np.power(2, self.board[cell]))) if self.board[cell] != 0 else ' '

  # Returns unoccupied cells as tuple (i values, j values).
  def _get_unoccupied_cells(self):
    return np.where(self.board == 0)

  # Returns list of positions to traverse in right order
  def _build_traversals(self, direction):
    traversals = [[], []]
    for pos in range(self.BOARD_SIZE):
      traversals[0].append(pos)
      traversals[1].append(pos)

    # Always traverse from farthest cell in chosen direction
    if direction[0] == 1:
      traversals[0] = traversals[0][::-1]
    if direction[1] == 1:
      traversals[1] = traversals[1][::-1]
    return traversals

  # Returns farthest open cell in direction from cell and next cell in direction to check for merge
  def _find_farthest_position(self, cell, direction):
    previous = None
    # Progress towards the vector direction until an obstacle is found
    while True:
      previous = cell
      cell = (previous[0] + direction[0], previous[1] + direction[1])
      if not (self.is_within_bounds(cell) and self.board[cell] == 0):
        return (previous, cell)

  def __str__(self):
    result = '.___________________________________.\n'
    for i in range(self.BOARD_SIZE):
      result += '|        |        |        |        |\n'
      for j in range(self.BOARD_SIZE):
        result += '|%8s' % (self._display_value_at((i, j)) + ' ' * (5 - len(self._display_value_at((i, j)))))
      result += '|\n|________|________|________|________|\n'
    return result[:-1]
