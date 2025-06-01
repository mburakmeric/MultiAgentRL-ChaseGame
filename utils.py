"""
Utility functions for the chase game.
Includes board rendering and helper functions.
"""

import numpy as np
from typing import Dict, Any, Optional


def render_board(board_state: 'BoardState', clear_screen: bool = True) -> str:
    """
    Render the board as ASCII art.
    
    Args:
        board_state: The current board state
        clear_screen: Whether to clear screen before rendering
        
    Returns:
        String representation of the board
    """
    board = board_state.board.copy()
    
    # Handle chaser overlap
    chaser1_pos = board_state.chaser1_pos
    chaser2_pos = board_state.chaser2_pos
    
    if (chaser1_pos.x == chaser2_pos.x and chaser1_pos.y == chaser2_pos.y):
        # Mark overlap position
        board[chaser1_pos.x, chaser1_pos.y] = 5  # Special value for overlap
    
    # Build the board string
    output = []
    
    if clear_screen:
        output.append("\033[2J\033[H")  # Clear screen and move cursor to top
    
    # Column labels
    output.append("   ")
    for col in range(board_state.board_size):
        output.append(f" {chr(ord('a') + col)}")
    output.append("\n")
    
    # Rows with row numbers
    for row in range(board_state.board_size):
        # Row number (1-indexed, right-aligned)
        output.append(f"{row + 1:2} ")
        
        # Cell contents
        for col in range(board_state.board_size):
            cell_value = board[row, col]
            
            if cell_value == board_state.EMPTY:
                output.append(" .")
            elif cell_value == board_state.OBSTACLE:
                output.append(" █")
            elif cell_value == board_state.RUNNER:
                output.append(" R")
            elif cell_value == board_state.CHASER1:
                output.append(" C1")  # Two chars, but visually OK
            elif cell_value == board_state.CHASER2:
                output.append(" C2")
            elif cell_value == 5:  # Chaser overlap
                output.append(" C")
        
        output.append("\n")
    
    return "".join(output)


def format_game_status(board_state: 'BoardState') -> str:
    """
    Format the current game status information.
    
    Args:
        board_state: The current board state
        
    Returns:
        Formatted status string
    """
    status_lines = []
    
    # Turn and agent info
    status_lines.append(f"\nTurn: {board_state.turn_count + 1}/{board_state.max_turns}")
    status_lines.append(f"Current agent: {board_state.get_current_agent()}")
    
    # Agent positions
    runner_pos = board_state.runner_pos
    chaser1_pos = board_state.chaser1_pos
    chaser2_pos = board_state.chaser2_pos
    
    status_lines.append(f"\nPositions:")
    status_lines.append(f"  Runner: {chr(ord('a') + runner_pos.y)}{runner_pos.x + 1}")
    status_lines.append(f"  Chaser1: {chr(ord('a') + chaser1_pos.y)}{chaser1_pos.x + 1}")
    status_lines.append(f"  Chaser2: {chr(ord('a') + chaser2_pos.y)}{chaser2_pos.x + 1}")
    
    # Game over status
    if board_state.game_over:
        status_lines.append(f"\nGAME OVER!")
        if board_state.winner == "runner":
            status_lines.append("The Runner wins!")
        else:
            status_lines.append("The Chasers win!")
    
    return "\n".join(status_lines)


def position_to_notation(x: int, y: int) -> str:
    """
    Convert board position to chess-like notation.
    
    Args:
        x: Row index (0-based)
        y: Column index (0-based)
        
    Returns:
        Position in notation like "c7"
    """
    return f"{chr(ord('a') + y)}{x + 1}"


def notation_to_position(notation: str) -> tuple:
    """
    Convert chess-like notation to board position.
    
    Args:
        notation: Position like "c7"
        
    Returns:
        Tuple of (x, y) indices
    """
    if len(notation) < 2:
        raise ValueError("Invalid notation")
    
    col = ord(notation[0].lower()) - ord('a')
    row = int(notation[1:]) - 1
    
    return (row, col)