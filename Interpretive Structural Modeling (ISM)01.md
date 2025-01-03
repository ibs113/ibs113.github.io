Step I: D--> A

```py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Generate a random 10x10 Direct Influence Matrix (D)
np.random.seed(42)
D = np.random.rand(10, 10) * 10  # Random values between 0 and 10
np.fill_diagonal(D, 0)  # No self-influence
print(D)

# Normalize the Direct Influence Matrix
max_row_sum = np.max(np.sum(D, axis=1))
N = D / max_row_sum

# Compute Total Relation Matrix (T)
I = np.eye(D.shape[0])
T = np.dot(N, np.linalg.inv(I - N))
print(T)

# Define a threshold (alpha) and compute Reachability Matrix (R)
alpha = np.mean(T) + np.std(T) # Use mean of T as threshold
A = (T >= alpha).astype(int)
print(A)
```

A--> O--> R

```PY
# # Method II
import numpy as np

def calculate_reachable_matrix_fast(A):
    """
    Efficiently calculates the reachable matrix R from the adjacency matrix A.
    Parameters:
        A (np.ndarray): Adjacency matrix (square matrix).
    Returns:
        R (np.ndarray): Reachable matrix.
    """
    # Step 1: Add identity matrix (O = A + I)
    n = A.shape[0]
    I = np.eye(n, dtype=int)
    O = A + I

    # Step 2: Use matrix exponentiation to calculate O^k until stabilization
    R = O.copy()
    prev_R = np.zeros_like(R)

    while not np.array_equal(R, prev_R):
        prev_R = R.copy()
        # Logical OR (boolean addition)
        R = (np.dot(R, O) > 0).astype(int)

    return R

# Calculate reachable matrix R
R = calculate_reachable_matrix_fast(A)

# Print results
print("Adjacency Matrix A:")
print(A)
print("\nReachable Matrix R:")
print(R)

# Method III
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm

def calculate_reachable_matrix_sparse(A):
    """
    Efficient calculation of reachable matrix R using sparse matrices.
    Parameters:
        A (np.ndarray): Adjacency matrix (square matrix).
    Returns:
        R (np.ndarray): Reachable matrix.
    """
    # Convert A to a sparse matrix
    A_sparse = csr_matrix(A)
    I_sparse = eye(A.shape[0], format='csr', dtype=int)

    # Step 1: Compute O = A + I
    O = A_sparse + I_sparse

    # Step 2: Compute reachable matrix R using exponentiation
    R = O
    prev_R = csr_matrix(A.shape)

    while not (R != prev_R).nnz == 0:  # Check if R has stabilized
        prev_R = R.copy()
        R = (R @ O > 0).astype(int)

    return R.toarray()
```

R--> reduce R and skeketon Matrix S

```py
import numpy as np

def calculate_skeleton_matrix(R):
    """
    Calculate the Skeleton Matrix (S) from the Reachable Matrix (R).
    Parameters:
        R (np.ndarray): Reachable Matrix (binary matrix with 0s and 1s).
    Returns:
        S (np.ndarray): Skeleton Matrix (binary matrix with 0s and 1s).
    """
    # Step 1: Transpose the Reachable Matrix
    R0 = R.copy()
    np.fill_diagonal(R0, 0)
    R_transposed = R0.T
    
    # Step 2: Identity Matrix
    I = np.eye(R0.shape[0], dtype=int)
    
    # Step 3: Compute S using the formula S = R' - (R' - I)^2 - I
    intermediate = R_transposed - I
    S = R_transposed - np.dot(intermediate, intermediate) - I
    
    # Step 4: Ensure S is binary (convert all non-zero values to 1)
    S = (S > 0).astype(int)
    np.fill_diagonal(S, 0)
    
    return S

print("\nReduced Reachable Matrix R0:")
R0=np.fill_diagonal(R, 0)
print(R0)

# Calculate Skeleton Matrix
S = calculate_skeleton_matrix(R)
print("Skeleton Matrix (S):")
print(S)
```

We got the R and S

1. Drive power and Dependent power

```py
# Compute Driving Power and Dependence Power
driving_power = np.sum(R, axis=1)  # Row sums
dependence_power = np.sum(R, axis=0)  # Column sums

# Visualize Driving vs Dependence Power
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(driving_power, dependence_power, color='b', s=100)
for i, txt in enumerate(range(len(driving_power))):
    plt.annotate(txt, (driving_power[i], dependence_power[i]), fontsize=12, ha='right')

plt.axhline(np.mean(dependence_power), color='gray', linestyle='--', linewidth=1.5)
plt.axvline(np.mean(driving_power), color='gray', linestyle='--', linewidth=1.5)
plt.title("Driving Power vs Dependence Power", fontsize=14)
plt.xlabel("Driving Power", fontsize=12)
plt.ylabel("Dependence Power", fontsize=12)
plt.grid(alpha=0.5)
plt.show()
```



```py
# driving_power = np.sum(R, axis=1)  # Row sums
# dependence_power = np.sum(R, axis=0)  # Column sums

x = np.sum(R, axis=1)  # Row sums
y = np.sum(R, axis=0)  # Column sums

#因子名
factors_name = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$',
        r'$x_6$', r'$x_7$', r'$x_8$', r'$x_9$', r'$x_{10}$'] # 10 factors

#画散点图，并增加相应名称、线段和调整大小和位置
plt.scatter(x, y, s=3, c='k')

plt.xlabel("Driving Power", fontsize=12)
plt.ylabel("Dependence Power", fontsize=12)
for i in range(len(x)):
    if i == 0:
        plt.text(x[i]+0.025, y[i]-0.085, factors_name[i], fontsize=17)
    elif i == 3:
        plt.text(x[i]+0.025, y[i]-0.1, factors_name[i], fontsize=17)
    elif i == 6:
        plt.text(x[i]-0.18, y[i]-0.1, factors_name[i], fontsize=17)
    elif i == 7:
        plt.text(x[i]-0.15, y[i]-0.12, factors_name[i], fontsize=17)
    else:
        plt.text(x[i]+0.025, y[i]+0.025, factors_name[i], fontsize=17)

plt.vlines(sum(x)/len(x), -1.55, 1.55, colors='k', linestyles='dashed')
plt.hlines(0, sum(x)/len(x)-2, sum(x)/len(x)+2, colors='k', linestyles='dashed')

plt.xlim(sum(x)/len(x)-2, sum(x)/len(x)+2)
plt.ylim(-1.55, 1.55) # check!!!!!!

plt.text(sum(x)/len(x)+2-0.2, 1.4-0.1, 'Ⅰ', fontsize=17)
plt.text(sum(x)/len(x)-2+0.1, 1.4-0.1, 'Ⅱ', fontsize=17)
plt.text(sum(x)/len(x)-2+0.1, -1.55+0.1, 'Ⅲ', fontsize=17)
plt.text(sum(x)/len(x)+2-0.2, -1.55+0.1, 'Ⅳ', fontsize=17)

plt.title("Driving Power vs Dependence Power", fontsize=14)
plt.grid(alpha=0.5)
plt.show()
```

```py
def xandy(final):
    Driving_power = []
    Dependence_power = []

    for i in range(n):
        countx=0
        county=0
        for j in range(n):
            if(final[i][j]==1):
                countx = countx + 1
            if(final[j][i]==1):
                county = county + 1
        Driving_power.append(countx)
        Dependence_power.append(county)
    return Driving_power, Dependence_power
    
def plot_it(Driving_power, Dependence_power):
    plt.scatter(Dependence_power, Driving_power)
    pts = dict() #pts is dictionary mapping from tuple of points to list of index corresponding to that
    for i, txt in enumerate(range(n)):
    	t = (Dependence_power[i], Driving_power[i]) #t is placeholder variable for coordinate
    	if t in pts:
    		pts[t].append(txt+1)
    	else:
    		pts[t]=[txt+1]

    for i, txt in enumerate(range(n)):
       	t = (Dependence_power[i], Driving_power[i])
        plt.annotate(pts[t],t)


    x1, y1 = [-1, n+1], [n/2, n/2]
    x2, y2 = [n/2, n/2], [-1, n+1]
    plt.plot(x1, y1, x2, y2)

    plt.xlim(0,n+1)
    plt.ylim(0,n+1)
    plt.xlabel('Dependence')
    plt.ylabel('Driving Power')
    plt.title('Micmac Analysis')
    plt.grid()
    plt.show()
    
Driving_power, Dependence_power = xandy(R)
plot_it(Driving_power, Dependence_power)
```



2. Hierarchy and Table for ISM only

```py
# Function to build hierarchy levels based on intersection sets
def build_hierarchy_with_sets(R):
    hierarchy_levels = [-1] * len(R)  # Initialize with -1
    for i, intersection in enumerate(R):
        if intersection:
            # Only calculate the level if the intersection is not empty
            non_empty_levels = [hierarchy_levels[elem] for elem in intersection if hierarchy_levels[elem] != -1]
            if non_empty_levels:  # Check if there are any levels to process
                hierarchy_levels[i] = max(non_empty_levels) + 1
            else:
                hierarchy_levels[i] = 0
    return hierarchy_levels

# Convert reachability sets to a 10x10 matrix
reachability_set_matrix = R

# Compute antecedent sets from the reachability set
antecedent_set = []
for i in range(reachability_set_matrix.shape[0]):
    antecedent = set(np.where(reachability_set_matrix[:, i] == 1)[0])
    antecedent_set.append(antecedent)

# Compute intersection sets
intersection_set = []
for i in range(reachability_set_matrix.shape[0]):
    intersection = set(np.where((reachability_set_matrix[:, i] == 1) & (reachability_set_matrix[i, :] == 1))[0])
    intersection_set.append(intersection)

# Build hierarchy levels
R1 = intersection_set
hierarchy_levels = build_hierarchy_with_sets(R1)
print(hierarchy_levels)

# Ensure all levels start from 0
for i in range(len(hierarchy_levels)):
    if hierarchy_levels[i] == -1:
        hierarchy_levels[i] = 0
print(hierarchy_levels)

# Generate a table for Reachability, Antecedent, Intersection Sets, and Level
def generate_sets_table_with_levels(reachability_set, antecedent_set, intersection_set, hierarchy_levels):
    data = {
        "Factor": list(range(len(reachability_set))),
        "Reachability Set (R)": [sorted(list(r)) for r in reachability_set],
        "Antecedent Set (A)": [sorted(list(a)) for a in antecedent_set],
        "Intersection Set (R ∩ A)": [sorted(list(i)) for i in intersection_set],
        "Level": [level+1 for level in hierarchy_levels]
    }
    df = pd.DataFrame(data)
    return df

# Create the table
sets_table = generate_sets_table_with_levels(reachability_set_matrix, antecedent_set, intersection_set, hierarchy_levels)

# Display the table
print("\nReachability, Antecedent, Intersection Sets, and Level Table:\n")
print(sets_table.to_string(index=False))
```



```py
def Level_Partioning(final):
    common_mat = []
    for i in range(n):
        temp_reach = []
        temp_antec = []
        for j in range(n):
            if(final[i][j]==1):
                temp_reach.append(j)
            if(final[j][i]==1):
                temp_antec.append(j)
        common_mat.append(temp_reach)
        common_mat.append(temp_antec)
    return common_mat
    
def find_level(intersection_set, common_mat):
    levels = np.zeros(n, dtype=int)
    count = 1

    while(stop_crit(levels)):
        store = []
        for i in range(n):
            if(len(intersection_set[i])!=0 and 
               set(intersection_set[i]) == set(common_mat[2*i])):
                levels[i] = count
                store.append(i)
        count = count + 1
        for x in store:
            for i in common_mat:
                if x in i: i.remove(x)
            for i in intersection_set:
                if x in i: i.remove(x)
    return levels
    
common_mat = Level_Partioning(R)  
n = 10 #check!!!!!
intersection_set = []
for i in range(n):
    intersection_set.append(list(set(common_mat[2*i]) & set(common_mat[2*i + 1])))
    
levels = find_level(intersection_set, common_mat)
levels   

n = 10 # check!!!!!
for i in range(n):
    print('Level in ISM for E%d is %d'%(i+1,levels[i]))
```

```py
# Hierarchical partitioning (simplified)
def partition_hierarchy(reachability):
    levels = []
    remaining_elements = set(range(len(reachability)))
    while remaining_elements:
        current_level = []
        for i in remaining_elements:
            if all(reachability[i, j] == 0 for j in remaining_elements if i != j):
                current_level.append(i)
        levels.append(current_level)
        remaining_elements -= set(current_level)
    return levels
    
# Partition hierarchy
hierarchy_levels = partition_hierarchy(R)
print("Hierarchical Levels:\n", hierarchy_levels)    
```

##### up_type and down_type

- from  R or S
- from reachability set, antecedent set, intersection set

```py
# from the R or S
def up_type_extraction(S):
    n = S.shape[0]
    nodes = set(range(n))
    levels = {}
    current_level = 0

    # Start with nodes with no incoming edges (in-degree = 0)
    while nodes:
        current_level_nodes = {node for node in nodes if np.sum(S[:, node]) == 0}
        if not current_level_nodes:
            break

        levels[current_level] = list(current_level_nodes)
        nodes -= current_level_nodes

        # Remove current level nodes from S
        for node in current_level_nodes:
            S[node, :] = 0  # Remove outgoing edges

        current_level += 1

    return levels

def down_type_extraction(S):
    n = S.shape[0]
    nodes = set(range(n))
    levels = {}
    current_level = 0

    # Start with nodes with no outgoing edges (out-degree = 0, np.sum(S[node, :]) == 0)
    while nodes:
        current_level_nodes = {node for node in nodes if np.sum(S[node, :]) == 0}
        if not current_level_nodes:
            break

        levels[current_level] = list(current_level_nodes)
        nodes -= current_level_nodes

        # Remove current level nodes from S
        for node in current_level_nodes:
            S[:, node] = 0  # Remove incoming edges

        current_level += 1

    return levels

# Perform UP-type extraction
up_hierarchy = up_type_extraction(S.copy())
print("UP-type Hierarchical Levels:")
for level, nodes in up_hierarchy.items():
    print(f"Level {level}: Nodes {nodes}")


# Perform DOWN-type extraction
down_hierarchy = down_type_extraction(S.copy())
print("\nDOWN-type Hierarchical Levels:")
for level, nodes in down_hierarchy.items():
    print(f"Level {level}: Nodes {nodes}")
```

R to three sets

```py
def find_all_sets(matrix):
    """
    Find Reachability Set, Antecedent Set, and Intersection Set for each element
    Returns dictionaries containing each set for all elements
    """
    n = len(matrix)
    reachability_sets = {}
    antecedent_sets = {}
    intersection_sets = {}
    
    # Find Reachability and Antecedent sets for each element
    for i in range(n):
        # Reachability Set (rows) - elements that can be reached from i
        reachability_sets[i] = set([j for j in range(n) if matrix[i][j] == 1])
        
        # Antecedent Set (columns) - elements that can reach i
        antecedent_sets[i] = set([j for j in range(n) if matrix[j][i] == 1])
        
        # Intersection Set
        intersection_sets[i] = reachability_sets[i].intersection(antecedent_sets[i])
    
    return reachability_sets, antecedent_sets, intersection_sets
    

# Step 1: Find all sets
reachability_sets, antecedent_sets, intersection_sets = find_all_sets(R)
    
# Print all sets for verification
print("Reachability Sets:")
for i in range(len(R)):
    print(f"Element {i}: {reachability_sets[i]}")
    
print("\nAntecedent Sets:")
for i in range(len(R)):
    print(f"Element {i}: {antecedent_sets[i]}")
    
print("\nIntersection Sets:")
for i in range(len(R)):
    print(f"Element {i}: {intersection_sets[i]}") 
```







```py
# we need the reachability set , anteceddent set and intersection set
def up_type_extraction(reachability_sets, intersection_sets):
    """
    Bottom-up hierarchy extraction using Reachability Set and Intersection Set
    """
    n = len(reachability_sets)
    unassigned = set(range(n))
    levels = []
    
    while unassigned:
        current_level = set()
        for element in unassigned:
            # Compare Reachability Set with Intersection Set for up-type
            if reachability_sets[element].intersection(unassigned) == \
               intersection_sets[element].intersection(unassigned):
                current_level.add(element)
        
        if not current_level:  # Prevent infinite loop
            break
            
        levels.append(sorted(list(current_level)))
        unassigned -= current_level
    
    return levels
    
def down_type_extraction(antecedent_sets, intersection_sets):
    """
    Top-down hierarchy extraction using Antecedent Set and Intersection Set
    """
    n = len(antecedent_sets)
    unassigned = set(range(n))
    levels = []
    
    while unassigned:
        current_level = set()
        for element in unassigned:
            # Compare Antecedent Set with Intersection Set for down-type
            if antecedent_sets[element].intersection(unassigned) == \
               intersection_sets[element].intersection(unassigned):
                current_level.add(element)
        
        if not current_level:  # Prevent infinite loop
            break
            
        levels.append(sorted(list(current_level)))
        unassigned -= current_level
    
    return levels
    
# Step 2: Extract hierarchies
print("\nBottom-up hierarchy:")
bottom_up = up_type_extraction(reachability_set, intersection_set)
for level, elements in enumerate(bottom_up, 1):
    print(f"Level {level}: {elements}")
    
print("\nTop-down hierarchy:")
top_down = down_type_extraction(antecedent_set, intersection_set)
for level, elements in enumerate(top_down, 1):
    print(f"Level {level}: {elements}")    
```











```py
def build_topological_hierarchy(S):
    """
    Build a topological hierarchy using the Skeleton Matrix S.
    Parameters:
        S (np.ndarray): Skeleton Matrix (binary matrix).
    Returns:
        hierarchy_levels (dict): Levels assigned to each node.
    """
    n = S.shape[0]
    G = nx.DiGraph()
    
    # Add edges based on Skeleton Matrix S
    for i in range(n):
        for j in range(n):
            if S[i, j] != 0:  # If there's a relationship
                if S[i, j] > 0:  # Positive relationship (direct)
                    G.add_edge(i, j)
                elif S[i, j] < 0:  # Mutual relationship (loop)
                    G.add_edge(i, j)  # One direction
                    G.add_edge(j, i)  # Opposite direction
    
    # Assign levels using DFS
    hierarchy_levels = {}
    def dfs(node, level):
        hierarchy_levels[node] = level
        for neighbor in G.neighbors(node):
            if neighbor not in hierarchy_levels:
                dfs(neighbor, level + 1)
    
    # Start DFS from the root node (node 0)
    dfs(0, 0)
    
    return G, hierarchy_levels

# Example Skeleton Matrix S
S = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
], dtype=int)

# Build Topological Hierarchy
G, hierarchy_levels = build_topological_hierarchy(S)

print(hierarchy_levels)
```

$$
R=\left[\begin{array}{cccc}
1 & 1 & 0 & 0\\
0 & 1 & 1 & 1\\
0 & 0 & 1 & 1\\
0 & 0 & 0 & 1
\end{array}\right]
$$

The **UP Type of Hierarchical Extraction** organizes nodes into levels starting from the **root causes** (lowest level) and progresses **upwards** in hierarchy. This approach relies on the **Reachable Set** and defines the **Common Set** as equal to the **Reachable Set**.

#### Extraction Process:

1. **Step 1 (Initial Nodes)**:
   - All nodes: {0,1,2,3}
2. **Level 0**:
   - Nodes at this level have reachable sets that do not include already assigned nodes.
   - R0={0,1}: Valid.
   - Extract Node 0: Level 0.
3. **Level 1**:
   - Remaining nodes: {1,2,3}
   - R1={1,2,3}: Valid.
   - Extract Node 1: Level 1.
4. **Level 2**:
   - Remaining nodes: {2,3}.
   - R2={2,3}: Valid.
   - Extract Node 2: Level 2.
5. **Level 3**:
   - Remaining node: {3}\.
   - R3={3}: Valid.
   - Extract Node 3: Level 3.

#### Result: 

UP Type Hierarchical Levels: [[0],[1],[2],[3]]



The **DOWN Type of Hierarchical Extraction** is the reverse of the UP type, where the hierarchy is organized starting from the **direct effects (top-level)** and progresses **downwards** to the root causes. This method relies on the **Antecedent Set** and defines the **Common Set** as equal to the **Antecedent Set**.



#### Extraction Process:

1. **Step 1 (Initial Nodes)**:
   - All nodes: {0,1,2,3}.
2. **Level 0** (Top-Level Effects):
   - Nodes at this level have antecedent sets that do not include already assigned nodes.
   - A3={1,2,3}: Valid.
   - Extract Node 3: Level 0.
3. **Level 1**:
   - Remaining nodes: {0,1,2}.
   - A2={1,2}: Valid.
   - Extract Node 2: Level 1.
4. **Level 2**:
   - Remaining nodes: {0,1}.
   - A1={0,1}: Valid.
   - Extract Node 1: Level 2.
5. **Level 3**:
   - Remaining node: {0}.
   - A0={0}: Valid.
   - Extract Node 0: Level 3.

#### Result:

DOWN Type Hierarchical Levels: [[3],[2],[1],[0]]





----

層級區分以及點點相連是兩個不同的觀念, 

The **Daisy Chain Topology** is derived from the **General Skeleton Matrix (S)**, where S[i,j]=1 indicates a direct relationship from factor i to factor j Key points for correctness:

- **Chains and Loops**:
  - A Daisy Chain topology is essentially a sequential chain where nodes are connected linearly or form loops.
  - If S[i,j]=1, there should be a directed edge i→j in the graph.
  - Self-loops (S[i,i]=1) and backward connections (e.g., S[n−1,0]) can exist in S, and these should be correctly represented in the topology.
- **Correct Implementation**:
  - Ensure that all non-zero entries in S are translated into graph edges.
  - Cyclic dependencies (loops) should be identified and visualized appropriately.

**Validation**: If your code generates a graph that reflects the structure of S, including all chains, loops, and self-loops, it is correct.



A **Daisy Chain Topology** in the context of network structures refers to a linear sequence of nodes, where each node is connected to its successor and, optionally, back to its predecessor, forming loops or linear chains.

**Loops (Cycles)**:

- Represent mutual dependencies between nodes.
- Indicate circular causality in a system where the factors affect each other in a cycle.

**Self-Loops**:

- Represent self-reinforcing effects or feedback mechanisms within a single node.
- Highlight intrinsic properties of a node affecting itself.





#### **Hierarchical Extractions (UP-type and DOWN-type)** Based on R

The **Reachability Matrix (R)** provides the foundation for **hierarchical extractions**. Key aspects include:

#### **UP-type Hierarchical Extraction**

- **Definition**:
  - Factors are grouped by their **reachability sets**: Common Set=Reachability Set∩Antecedent Set.
  - Start from the factors with the `smallest` **reachability sets** that include themselves.
- **Process**:
  1. Identify factors where the reachability set equals the common set.
  2. Extract these factors as the highest level and remove them from consideration.
  3. Repeat until all factors are assigned to levels.

#### **DOWN-type Hierarchical Extraction**

- **Definition**:
  - Factors are grouped by their **antecedent sets**: Common Set=Reachability Set∩Antecedent Set.
  - Start from the factors with the `smallest` **antecedent sets** that include themselves.
- **Process**:
  1. Identify factors where the antecedent set equals the common set.
  2. Extract these factors as the lowest level and remove them from consideration.
  3. Repeat until all factors are assigned to levels.

**Validation**:

- Ensure that RRR is properly computed (e.g., R=O^k where O=A+I).
- Check the grouping process for accuracy:
  - **UP-type**: Progression from higher levels (root causes) downward.
  - **DOWN-type**: Progression from lower levels (direct effects) upward.



### **Key Differences Between S and R:**

1. **General Skeleton Matrix (S)**:
   - Derived from R or other transformations.
   - Represents reduced connections or key structural features (e.g., loops, cycles).
2. **Reachability Matrix (R)**:
   - Captures full reachability relationships (transitive closure) based on the adjacency matrix.
   - Forms the basis for hierarchical extractions.



S retains only **direct relationships** and removes **transitive redundancies**.

This matrix serves as the basis for analyzing the **essential structure** of the system.



The **Skeleton Matrix** is a **reduced version** of the **Reachable Matrix (R)**. It is derived by considering the relationships between factors and reducing or eliminating certain connections based on specific rules or conditions. The main focus of the Skeleton Matrix is to capture the **essential or core relationships** between factors, leaving out less significant or indirect relationships.

The **General Skeleton Matrix** is a broader form of the Skeleton Matrix, typically used when dealing with complex networks, systems, or when you want to preserve more information about the relationships among the factors.

In the **Skeleton Matrix**, we removed indirect relationships or reduced them, while in the **General Skeleton Matrix**, we may have kept a more detailed version of relationships, potentially accounting for indirect effects.

**Skeleton Matrix (S)**: Only keeps direct relationships from the Reachable Matrix (R).

**General Skeleton Matrix (G)**: Captures both direct and indirect relationships by computing higher powers of RRR.



```py
from graphviz import Digraph

def draw_hierarchy_with_dashed_lines(hierarchy, title, filename):
    """
    Draw a hierarchy diagram using Graphviz with horizontal dashed lines between levels.
    :param hierarchy: List of sets or lists representing hierarchy levels.
    :param title: Title of the diagram.
    :param filename: Filename to save the diagram.
    """
    dot = Digraph(format="png")
    dot.attr(rankdir="TB")  # Top to bottom hierarchy
    
    # Add nodes with level annotations
    for level_index, level in enumerate(hierarchy):
        with dot.subgraph() as sub:
            sub.attr(rank="same")  # Group nodes on the same level
            for node in level:
                sub.node(str(node), f"Node {node+1} ")  # Level starts from 1
    
    # Add edges between levels
    for level_index, level in enumerate(hierarchy[:-1]):
        next_level = hierarchy[level_index + 1]
        for node in level:
            for next_node in next_level:
                dot.edge(str(node), str(next_node))
    
    # Add horizontal dashed lines between different levels
    for level_index in range(len(hierarchy) - 1):
        # Draw a horizontal dashed line from one level to the next level
        dot.attr(style="dashed", color="gray")
        dot.edge(f"Level_{level_index + 1}", f"Level_{level_index + 2}")  # Horizontal dashed line between levels
        
    dot.attr(label=title, fontsize="16", labelloc="top", labeljust="center")
    dot.render(filename, cleanup=True)
    print(f"Diagram saved as {filename}.png")

# Corrected Example hierarchies with explicit level assignments:
up_type_hierarchy = [{0}, {1, 2}, {3, 4, 5}]  # Level 1: [0], Level 2: [1, 2], Level 3: [3, 4, 5]
down_type_hierarchy = [{5, 6, 7}, {3, 4}, {1}, {0}]  # Level 1: [5, 6, 7], Level 2: [3, 4], Level 3: [1], Level 4: [0]

# Draw UP-Type Hierarchy with horizontal dashed lines
draw_hierarchy_with_dashed_lines(up_type_hierarchy, "UP-Type Hierarchy", "up_type_hierarchy")

# Draw DOWN-Type Hierarchy with horizontal dashed lines
draw_hierarchy_with_dashed_lines(down_type_hierarchy, "DOWN-Type Hierarchy", "down_type_hierarchy")
```

## Graphviz 套件

graphviz provides two classes: Graph無向圖 and Digraph有向圖. 

```py
pip install graphviz
from graphviz import Digraph
dot = Digraph(comment='The Round Table')
#新增一個點 A，顯示名稱為 QQ
dot.node('A', label = 'QQ') 
#新增一個點 B，顯示名稱為 www
dot.node('B', label = 'www')

#新增一個從點 A 到點 B 的邊，顯示名稱為 Like
dot.edge("A", "B", label = "Like")
```

```py
from graphviz import Digraph
dot = Digraph(comment='The Round Table')

dot.node('A', label = 'QQ', color='green')
dot.node('B', label = 'www')
dot.node('L', label = 'PP')

dot.edges(['AB', 'AL'])
dot.edge("B", "L", constraint='false')
```

we can use the graph_attrr, node_attr, and edge_attr

```py
from graphviz import Digraph
from graphviz import Graph
ni = Graph('ni', format='jpg')

ni.attr('node', shape='rarrow')
ni.node('1', 'Ni!')
ni.node('2', 'Ni!')

ni.node('3', 'Ni!', shape='egg')

ni.attr('node', shape='stat')
ni.node('4', 'Ni!')
ni.node('5', 'Ni!')
ni.attr(rankdir='LR')

ni.edges(['12', '23', '34', '45'])
ni.view()
```

Subgraph and Cluster

```py
from graphviz import Digraph
from graphviz import Graph
p = Graph(name='parent', node_attr={'shape': 'plaintext'}, format='png')
p.edge('spam', 'eggs')

c=Graph('child', note_attr={'shape': 'box'})
c.edge('foo', 'bar')

p.subgraph(c)
p.view()
```

```py
from graphviz import Digraph

# 创建一个有向图
dot = Digraph()

# 设置图形的属性
dot.attr('graph', forcelabels='true', rankdir='LR', ranksep='1', nodesep='0.5')

# 设置所有节点的默认样式
dot.attr('node', shape='box')

# 添加具有特定属性的节点
dot.node('start', xlabel='start', xlp='0,0', shape='doublecircle', label=' ')
dot.node('fault', xlabel='fault', shape='doublecircle', label=' ')
dot.node('complete', xlabel='complete', shape='doublecircle', label=' ')

# 添加边
dot.edge('requested', 'fault')
dot.edge('requested', 'progress')
dot.edge('start', 'requested')
dot.edge('progress', 'fault')
dot.edge('progress', 'progress')
dot.edge('progress', 'complete')

# 保存和/或显示图形
dot.render('my_graph', format='png', view=True)
```

