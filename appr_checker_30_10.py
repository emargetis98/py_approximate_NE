import numpy as np

def compute_epsilon_approx_ne(R, C, x, y):
    """
    Computes the value ε for an approximate Nash Equilibrium (ε_approx).
    Explicitly handles pure Nash Equilibria by returning ε_approx = 0.

    Parameters:
        R (numpy.ndarray): Payoff matrix for the row player.
        C (numpy.ndarray): Payoff matrix for the column player.
        x (numpy.ndarray): Strategy of the row player (1D array, mixed strategy).
        y (numpy.ndarray): Strategy of the column player (1D array, mixed strategy).

    Returns:
        float: The ε value for the approximate Nash Equilibrium.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    y = np.array(y) if not isinstance(y, np.ndarray) else y

    # Check for pure strategies (i.e., exactly one element is 1)
    if np.sum(x == 1) == 1 and np.sum(y == 1) == 1:
        return 0.0  # A pure Nash Equilibrium is a valid approximate NE with ε_approx = 0

    # Compute payoffs
    row_payoff = np.dot(x.T, np.dot(R, y))  # x^T R y
    col_payoff = np.dot(x.T, np.dot(C, y))  # x^T C^T y

    # Compute maximum payoffs
    max_row = np.max(np.dot(R, y))  # max_i { (R y)_i }
    max_col = np.max(np.dot(C.T, x))  # max_j { (C^T x)_j }

    # Compute ε for the approximate NE
    eps_approx = max(max_row - row_payoff, max_col - col_payoff)
    return eps_approx

def compute_epsilon_supp_ne(R, C, x, y):
    """
    Computes the value ε for a well-supported approximate Nash Equilibrium (ε-WSNE).
    
    Parameters:
        R (numpy.ndarray): Payoff matrix for the row player.
        C (numpy.ndarray): Payoff matrix for the column player.
        x (numpy.ndarray): Strategy of the row player (1D array, mixed strategy).
        y (numpy.ndarray): Strategy of the column player (1D array, mixed strategy).
    
    Returns:
        float: The ε value for the well-supported approximate Nash Equilibrium.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    y = np.array(y) if not isinstance(y, np.ndarray) else y

    # Check for pure strategies (i.e., exactly one element is 1)
    if np.sum(x == 1) == 1 and np.sum(y == 1) == 1:
        return 0.0 
    
    # Compute the maximum payoff for each player
    max_row_payoff = np.max(np.dot(R, y))  # max_i { (R y)_i }
    max_col_payoff = np.max(np.dot(C.T, x))  # max_j { (C^T x)_j }

    # Compute ε for the row player (any strategy played with positive probability must be an ε-best response)
    epsilon_row = 0
    for i in range(len(x)):
        if x[i] > 0.0001:  # Only for strategies played with positive probability
            payoff = np.dot(R[i], y)
            epsilon_row = max(epsilon_row, max_row_payoff - payoff)

    # Compute ε for the column player
    epsilon_col = 0
    for j in range(len(y)):
        if y[j] > 0.0001:  # Only for strategies played with positive probability
            payoff = np.dot(C[:, j], x)
            epsilon_col = max(epsilon_col, max_col_payoff - payoff)

    # The total ε is the maximum of the two
    epsilon_wsne = max(epsilon_row, epsilon_col)
    return epsilon_wsne
