## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).
See [LICENSE](./LICENSE).

## Stockfish

This repo includes a Windows version of [Stockfish](https://stockfishchess.org) in `stockfish/`.
Stockfish is licensed under the GNU GPL v3 (see `third_party/stockfish/Copying.txt`).
If you are not on Windows, define a path to the stockfish executable in “env_variables.py.”

## Known Issue
The chess pieces may look like rectangles if the required font is not installed.  Either install the "segoeuisymbol" font or define your own in "env_variables.py"

## How to Use
The main script is "main.py."  It takes two arguments in this order:
- `white_player`: The player who controls the white chess pieces.  Options: human, minimax, new_minimax, mcts, ann_minimax, stockfish  
- `black_player`: The player who controls the black chess pieces.  Options: human, minimax, new_minimax, mcts, ann_minimax, stockfish