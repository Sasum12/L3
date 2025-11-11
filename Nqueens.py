def isSafe(board, row, col, n):
    # Check if there's a queen in the same row
    for i in range(n):
        if board[row][i] == 1:
            return False

    # Check if there's a queen in the same column
    for i in range(n):
        if board[i][col] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

def backtrack(board, row, n):
    if row == n:
        return True
    
    for col in range(n):
        if isSafe(board, row, col, n):
            board[row][col] = 1
            if backtrack(board, row + 1, n):
                return True
            board[row][col] = 0  # Backtrack if no solution found
    
    return False

def printSolution(board, n):
    for i in range(n):
        for j in range(n):
            if board[i][j] == 1:
                print("Q ", end="")
            else:
                print("_ ", end="")
        print()

def solveNQueens():
    n = int(input("Enter the size of the chessboard (N): "))
    
    # Initialize the board with all 0's
    board = [[0 for _ in range(n)] for _ in range(n)]
    
    # Manually place the first queen (for example at (0, 0))
    first_queen_row = 0
    first_queen_col = 0
    board[first_queen_row][first_queen_col] = 1
    
    print("First Queen placed at (0, 0):")
    printSolution(board, n)
    
    # Now use backtracking to place the remaining queens
    if backtrack(board, 1, n):
        print("\nFinal N-Queens Solution:")
        printSolution(board, n)
    else:
        print("No solution found.")

if __name__ == "__main__":
    solveNQueens()
