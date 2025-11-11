import heapq

# Define Node class for the Huffman tree
class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    # Comparison operator for priority queue
    def __lt__(self, other):
        return self.freq < other.freq

# Function to print Huffman codes
def printNodes(node, val=''):
    newVal = val + str(node.huff)
    if (node.left):
        printNodes(node.left, newVal)
    if (node.right):
        printNodes(node.right, newVal)
    if (not node.left and not node.right):
        print(f"{node.symbol} -> {newVal}")

# Manually input characters and frequencies
chars = []
freq = []

# Input number of characters
n = int(input("Enter the number of characters: "))

# Input each character and its frequency
for i in range(n):
    char = input(f"Enter character {i + 1}: ")
    chars.append(char)
    frequency = int(input(f"Enter frequency of {char}: "))
    freq.append(frequency)

# Create a priority queue (min-heap)
nodes = []
for x in range(len(chars)):
    heapq.heappush(nodes, Node(freq[x], chars[x]))

# Build the Huffman tree
while len(nodes) > 1:
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)
    left.huff = 0
    right.huff = 1
    newNode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
    heapq.heappush(nodes, newNode)

# Print the Huffman codes
print("Huffman Codes are:")
printNodes(heapq.heappop(nodes))
