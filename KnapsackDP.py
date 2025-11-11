def knapSack(W, wt, val, n):
    dp = [0 for i in range(W + 1)]
    
    # Iterate through all items
    for i in range(1, n + 1):
        # Traverse the weights backwards
        for w in range(W, 0, -1):
            if wt[i - 1] <= w:
                dp[w] = max(dp[w], dp[w - wt[i - 1]] + val[i - 1])
    
    return dp[W]

# Input section
W = []
V = []

# Number of items
num = int(input("Enter the number of weights or values required: \n"))

# Input the weights
for i in range(num):
    n = int(input(f"Enter weight {i+1} : "))
    W.append(n)

print("\n")

# Input the values
for i in range(num):
    n = int(input(f"Enter value {i+1} : "))
    V.append(n)

print("\n")

# Capacity of the knapsack
M = int(input("Enter the cost: "))
n = len(V)

print("\n")

# Solve the knapsack problem and print the result
print(knapSack(M, W, V, n))
