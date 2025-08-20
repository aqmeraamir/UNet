# U-Net with Self-Attention for MRI Segmentation

This project contains the implementation of a 3D U-Net model with self-attention (SA) designed for MRI brain tumor segmentation. The project was developed as part of my Extended Project Qualification (EPQ) and uses the BraTS 2020 dataset.

---

## Features

- 3D U-Net using PyTorch for volumetric segmentation
- Move generation for all piece types
- Minimax AI with a simple evaluation function
- Legal move filtering and check detection
- Command-line input with standard chess algebraic notation
- ❌ No castling, en passant, or promotion support (yet)

---

## Setup

### Download executable
1. Download the latest executable file [chess-bot-mac](https://github.com/aqmeraamir/chess-bot/releases/download/1.0/chess-bot-mac) (only works on mac/unix)

2. Run the executable 

OR

### Build from source (will work on windows)

1. Clone the repo:
   ```bash
   git clone https://github.com/aqmeraamir/chess-bot.git
   cd chess-bot
   ```

2. Build the project using Make
    ```
    make
    ```


3. Run the game
    ```
    .bin/ChessBot
    ```

## How to Play
Once the board prints in the terminal, enter your moves using standard algebraic notation:

| Example | Meaning                 |
| ------- | ----------------------- |
| `e2e4`  | Move pawn from e2 to e4 |
| `Nf3`   | Move knight to f3       |
| `q`     | Quit the game           |

The AI will play as Black and make its move after yours.

## Sample Output

CLI:

```
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
6
5
4
3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
  a b c d e f g h
```

Enter your move (e.g., e2e4) or 'q' to quit:

## License
MIT — feel free to use or modify this for your own chess projects.
