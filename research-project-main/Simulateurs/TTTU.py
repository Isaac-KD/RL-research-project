import numpy as np
from typing import Optional

class TTTU:
    
    def __init__(self):
        self.big_board = np.zeros(9, dtype= int)
        self.small_boards = np.zeros((9, 3, 3), dtype= int)

    def put(self, player: int, board_index: int, row: int, col: int) -> bool:
        """
        Place le jeton du joueur `player` dans la sous-grille `board_index`
        à la position (`row`, `col`). Retourne True si le coup est valide, False sinon.
        """
        board = (self.small_boards)[board_index, :, :]
        
        if board[row, col] != 0:
            return False  # Case déjà prise

        board[row, col] = player  # Placer le coup

        # Vérifier si le joueur a gagné cette sous-grille
        if self.win(board):  
            self.big_board[board_index] = player
        elif self.is_full(board):
            self.big_board[board_index] = 10  # Match nul dans cette sous-grille

        return True

    @staticmethod
    def win(board: np.array) -> bool:
        """
        Vérifie si une grille 3x3 est gagnée par un joueur.
        """
        sum_rows = np.sum(board, axis=1)
        sum_cols = np.sum(board, axis=0)
        sum_main_diag = np.trace(board)
        sum_secondary_diag = np.fliplr(board).diagonal().sum()

        return np.any(sum_rows == 3) or np.any(sum_rows == -3) or \
               np.any(sum_cols == 3) or np.any(sum_cols == -3) or \
               sum_main_diag in (3, -3) or sum_secondary_diag in (3, -3)

    @staticmethod
    def is_full(board: np.array) -> bool:
        """
        Vérifie si une grille 3x3 est entièrement remplie (aucune case vide).
        """
        return np.all(board != 0)

    def next_allowed_board_index(self, last_move: tuple[int, int]) -> Optional[int]:
        """
        Calcule l'indice de la sous-grille où le prochain joueur peut jouer
        en fonction du dernier coup joué.
        Retourne None si la sous-grille est déjà pleine.
        """
        x = 3 * last_move[0] + last_move[1]
        return x if not self.is_full(self.small_boards[x]) else None
    
    def reset(self):
        """
        Réinitialise le plateau de jeu.
        """
        self.big_board.fill(0)
        self.small_boards.fill(0)
