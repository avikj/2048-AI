import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from game import GameBoard


def main(args):
  moves = {
      'w': GameBoard.MOVE_UP,
      'a': GameBoard.MOVE_LEFT,
      's': GameBoard.MOVE_DOWN,
      'd': GameBoard.MOVE_RIGHT,
  }
  scores = []
  for n in range(args.nrof_games):
    if args.nrof_games != 1:
      if not args.quiet:
        print '\n\n\n'
      print '\nGame #%s' % (n + 1)
    board = GameBoard()
    while not board.is_game_over():
      if not args.quiet:
        print 'Score: %d' % board.current_score()
      if not args.quiet:
        print board
      move_letter = ' '
      while len(move_letter) > 1 or move_letter not in 'wasd':
        if args.random_moves:
          move_letter = random.choice('wasd')
          if not args.quiet:
            print 'Randomly chose> %s\n' % move_letter
        else:
          move_letter = raw_input('Enter move (WASD)>').lower()

      moved = board.move(moves[move_letter])
      if not moved:
        if not args.quiet:
          print 'No tiles were moved.'
      else:
        board.add_tile()

    if not args.quiet:
      print '======================================='
      print board
      print 'Game Over.'
    print 'Score: %d' % board.current_score()
    scores.append(board.current_score())
  if args.show_stats:
    print '\n\n\nScore Stats'
    print '==========='
    print 'Mean score: %.2f' % np.mean(scores)
    print 'High score: %d' % np.amax(scores)
    print 'Low score: %d' % np.amin(scores)
    print 'Standard deviation: %.2f' % np.std(scores)
    plt.hist(scores, bins='auto')
    plt.title('2048 Score Distribution')
    plt.show()


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--random_moves', help='Switch to run random simulation', action='store_true')
  parser.add_argument('-n', '--nrof_games', help='Number of games to play.', type=int, default=1)
  parser.add_argument('-q', '--quiet', help='Switch to prevent printing boards.', action='store_true')
  parser.add_argument('-s', '--show_stats', help='Switch to show score stats.', action='store_true')
  return parser.parse_args(argv)


main(parse_arguments(sys.argv[1:]))
