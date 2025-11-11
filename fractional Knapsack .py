# Time Complexity:
#- **Sorting the items based on value-to-weight ratio**: \(O(n \log n)\), where \(n\) is the number of items.
#- **Filling the knapsack**: \(O(n)\), since we iterate through the sorted list of items and possibly take fractions.

#Thus, the overall time complexity is:\[ O(n \log n) \]

# Space Complexity:
#- The space complexity is \(O(n)\) because we store the list of \(n\) items and their value-to-weight ratios.

#So, the space complexity is:\[ O(n) \]
#Program:

class KnapsackPackage(object):
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.cost = value / weight

    def __lt__(self, other):
        return self.cost < other.cost

class FractionalKnapsack(object):
    def knapsackGreProc(self, W, V, M, n):
        packs = []
        
        # Create the KnapsackPackage objects for each item
        for i in range(n):
            packs.append(KnapsackPackage(W[i], V[i]))
        
        # Sort the packages based on the cost (value/weight) in descending order
        packs.sort(reverse=True)

        remain = M  # Remaining capacity of the knapsack
        result = 0  # Maximum value obtained
        i = 0       # Index of the current item

        # Loop through the items and pick them greedily
        while remain > 0 and i < n:
            if packs[i].weight <= remain:  # If the entire item can fit
                remain -= packs[i].weight
                result += packs[i].value
                print(f"Pack {i} - Weight {packs[i].weight} - Value {packs[i].value}")
            else:
                # Take the fractional part of the item that fits
                fraction = remain / packs[i].weight
                result += packs[i].value * fraction
                print(f"Pack {i} (fraction) - Weight {remain} - Value {packs[i].value * fraction}")
                remain = 0  # Knapsack is full after taking the fraction

            # Move to the next item
            i += 1

        print("Max Value:\t", result)

if __name__ == "__main__":
    W = []
    V = []
    
    # Input for the number of items
    num = int(input("Enter the number of weights or values required:\n"))
    
    # Input the weights of the items
    for i in range(num):
        n = int(input(f"Enter weight {i + 1} : "))
        W.append(n)

    print("\n")
    
    # Input the values of the items
    for i in range(num):
        n = int(input(f"Enter value {i + 1} : "))
        V.append(n)

    print("\n")
    
    # Input the maximum weight capacity of the knapsack
    M = int(input("Enter the cost: "))
    n = len(V)
    print("\n")
    
    # Process the knapsack problem using the greedy approach
    proc = FractionalKnapsack()
    proc.knapsackGreProc(W, V, M, n)
