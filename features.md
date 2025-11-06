
# Feature Frequency Per Game Stage
| **Feature**               | **Early (1–12)** | **Mid (13–35)** | **Late (36+)** | **Notes**                              |
|---------------------------|------------------|-----------------|----------------|----------------------------------------|
| Material Imbalance        | ~30%             | ~65%            | ~85%           | Starts small, grows with captures      |
| King in Center            | ~90%             | ~20%            | <5%            | Almost all early, rare late            |
| Pawn Shield (3+ pawns)    | ~70%             | ~50%            | ~10%           | Erodes as pawns advance/capture        |
| Open File Near King       | ~10%             | ~40%            | ~60%           | Opens up in middlegame                 |
| Castled                   | ~20%             | ~80%            | ~90%           | Most castle by move 12                 |
| Knight Outpost            | <5%              | ~20%            | ~10%           | Peaks in middlegame                    |
| Bishop Pair               | ~50%             | ~40%            | ~30%           | One bishop often traded                |
| Rook on Open File         | <5%              | ~35%            | ~50%           | Rooks activate later                   |
| Rook on 7th Rank          | <1%              | ~15%            | ~40%           | Classic endgame motif                  |
| Piece Mobility            | ~60%             | ~85%            | ~70%           | Peaks mid, drops late                  |
| Doubled Pawns             | ~20%             | ~50%            | ~60%           | Accumulates over time                  |
| Isolated Pawns            | ~10%             | ~25%            | ~35%           | More common late                       |
| Passed Pawns              | <5%              | ~15%            | ~50%           | Endgame gold                           |
| Backward Pawns            | ~5%              | ~20%            | ~15%           | Middlegame weakness                    |
| Center Control            | ~70%             | ~60%            | ~40%           | Fades in endgame                       |
| Space Advantage           | ~40%             | ~55%            | ~50%           | One side usually ahead                 |
| Pieces Developed          | ~50%             | ~90%            | ~95%           | Complete by move 15                    |
| Rooks Connected           | ~20%             | ~60%            | ~70%           | After castling                         |
| King Castled Early        | ~60%             | ~20%            | —              | Only relevant early                    |


# Features, how to compute and weights
| **Feature**              | **How to Compute (Python)**                                   | **Weight (centipawns)** | **Notes**                                      |
|--------------------------|---------------------------------------------------------------|--------------------------|------------------------------------------------|
| **Material Imbalance**   | `sum(piece_value(p) for p in board.pieces) * (1 if white else -1)` | **+100 per pawn**       | pawn=100, knight=320, bishop=330, rook=510, queen=1000 |
| **King in Center**       | `if file in [3,4] and rank in [3,4]: -50`                     | **−50**                 | Penalty for king on d4/d5/e4/e5                |
| **Pawn Shield (per pawn)** | `count pawns on f2/g2/h2 (white) or f7/g7/h7 (black)`        | **+30**                 | Max +90 for full shield                        |
| **Open File Near King**  | `if no pawns on f/g/h file: -80`                             | **−80**                 | Per open file                                  |
| **Castled**              | `if board.has_castled(white/black): +50`                     | **+50**                 | Bonus for castling                             |
| **Knight Outpost**       | `knight on d5/e5 (supported by pawn)`                        | **+40**                 | Central, protected                             |
| **Bishop Pair**          | `if both bishops alive: +50`                                 | **+50**                 | Classic bonus                                  |
| **Rook on Open File**    | `rook on file with no pawns`                                 | **+40**                 | Per rook                                       |
| **Rook on 7th Rank**     | `rook on rank 7 (white) or 2 (black)`                        | **+50**                 | Attacks pawns/kings                            |
| **Piece Mobility**       | `len(list(board.legal_moves))`                               | **+5 per move**         | Total legal moves                              |
| **Doubled Pawns**        | `count files with 2+ pawns`                                  | **−30**                 | Per doubled pawn                               |
| **Isolated Pawns**       | `pawn with no friendly pawn on adjacent files`               | **−40**                 | Per isolated pawn                              |
| **Passed Pawns**         | `no enemy pawn in front or adjacent files`                   | **+50 to +150**         | +50 base, +20 per rank advanced                |
| **Backward Pawns**       | `pawn can't advance without capture`                         | **−20**                 | Per backward pawn                              |
| **Center Control**       | `pieces/pawns on d4,d5,e4,e5`                                | **+20**                 | Per controlled square                          |
| **Space Advantage**      | `pawns in opponent's half`                                   | **+10**                 | Per pawn                                       |
| **Pieces Developed**     | `knights/bishops off back rank`                              | **+30**                 | Per developed piece                            |
| **Rooks Connected**      | `rooks on same rank, no pieces between`                      | **+20**                 | Per connected pair                             |
| **King Castled Early**   | `castled by move 10`                                         | **+40**                 | Opening bonus                                  |

# Time penalty and bonus

$ \text{score} = \begin{cases} +0.5 \times \frac{40 - \text{move\_count}}{30} & \text{if moves} < 40\\ -0.5 \times \frac{\text{move\_count} - 40}{40} & \text{if moves} \geq 40 \end{cases} $



