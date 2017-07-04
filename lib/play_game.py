from game import GameBoard

def play_and_get_score(player):
  board = GameBoard()
  while True:
    move = player.get_move(board)
    moved = board.move(move)
    board.add_tile()
    if board.is_game_over():    
      return board.current_score()
      print('game over')