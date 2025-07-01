# py_approximate_NE
Python 3.12.5 source code for two functions that compute the approximation guarantee (e) for approximate Nash equilibrium and well-supported approximate Nash equilibrium.
A function has been designed in Python to compute the approximation values of a strategy profile as an ε-approximate / well-supported Nash equilibrium in a bimatrix game.
The ε-approximate Nash equilibrium (ε-NE) measures how far the payoff of the given strategy for the row player/column player deviates from the maximum possible payoff they could achieve against the opponent’s fixed strategy.
In contrast, the ε-well-supported Nash equilibrium (ε-WSNE) considers only those actions that are played with positive probability to compute the maximum deviation in payoff from the best possible response against the opponent’s strategy.
Both functions make extensive use of NumPy’s np.dot() for matrix multiplication, enabling efficient computation of expected payoff vectors for each player. This is critical because nested loops for multiplying large matrices would be computationally expensive.
To avoid errors in algebraic computations related to floating-point precision (as defined by the IEEE 754 standard for double-precision numbers), it is common not to compare two numbers directly (e.g., checking for strict equality). Instead, the comparison is based on whether the absolute value of their difference is smaller than a small threshold, say e. If this difference is smaller than e, the numbers are treated as equal; otherwise, they are considered different.
The input to the functions is a set of NumPy array objects (R, C, x, y), representing the payoff matrices for the row and column players and their respective mixed strategies.
At the end, the function returns a floating-point value, which represents the approximation guarantee.
For ε-NE, this is the difference between the maximum possible payoffs and the actual payoffs.
For ε-WSNE, the function checks every action played with positive probability (i.e., when x[i] > 0.0001 and y[j] > 0.0001). For these actions, it computes the difference between the maximum and actual payoffs.
