"""
create_additional_datasets.py — Generate additional ASAG evaluation datasets.

PURPOSE:
  Creates 2 new datasets for Table 3 cross-domain evaluation:
    1. Vietnamese STEM dataset (Math/Physics/Chemistry) — 200 samples
    2. CS Domain dataset (Data Structures/Algorithms) — 150 samples

  Both datasets use realistic variations with:
    - Multiple answer quality levels (not just 3 templates)
    - Realistic typos and incomplete answers
    - Unbalanced label distribution (mimics real-world data)
    - Compatible with ASAGSample format

USAGE:
  python research/create_additional_datasets.py
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_cs_domain_dataset(n_samples: int = 150, seed: int = 42) -> List[Dict]:
    """
    Create CS Domain dataset (Data Structures & Algorithms).
    
    Based on Mohler-style questions about fundamental CS topics.
    More diverse than the original synthetic dataset.
    """
    rng = np.random.default_rng(seed)
    
    topics = [
        {
            "question": "What is a stack data structure and how does it work?",
            "reference": "A stack is a linear data structure that follows the Last-In-First-Out (LIFO) principle. Elements can only be added (push) or removed (pop) from the top. Common operations include push, pop, peek, and isEmpty, all running in O(1) time.",
            "correct_variants": [
                "A stack is a LIFO data structure where elements are added and removed from the top only. Push adds, pop removes, and both are O(1) operations.",
                "Stack follows Last-In-First-Out principle. You push elements onto the top and pop them from the top. Operations like push, pop, peek are all constant time O(1).",
                "A stack works like a pile of plates — you can only add or remove from the top. This LIFO behavior makes push and pop O(1) operations. It also supports peek to view the top element.",
            ],
            "partial_variants": [
                "A stack is where you add and remove elements from one end.",
                "Stack is LIFO. You push and pop things.",
                "It's a data structure that stores elements in order, and you take the last one added first.",
                "A stack is like a list but you can only access the top element.",
            ],
            "incorrect_variants": [
                "A stack is a FIFO data structure where elements are processed in order of arrival.",
                "Stack allows random access to any element like an array.",
                "A stack is a tree structure where each node has two children.",
                "Stack works like a queue — first in, first out.",
            ],
        },
        {
            "question": "Explain the difference between an array and a linked list.",
            "reference": "An array stores elements in contiguous memory locations allowing O(1) random access by index, but insertion/deletion in the middle requires O(n) shifting. A linked list stores elements in nodes connected by pointers, allowing O(1) insertion/deletion at known positions but requiring O(n) for access since elements must be traversed sequentially.",
            "correct_variants": [
                "Arrays use contiguous memory for O(1) index access but O(n) insertion. Linked lists use nodes with pointers for O(1) insertion at known positions but O(n) access since you must traverse sequentially.",
                "Array elements are stored consecutively in memory, giving constant-time random access. Linked list nodes are scattered in memory, connected by pointers. Arrays are better for access, linked lists for frequent insertions and deletions.",
                "The key difference is memory layout: arrays are contiguous (fast access O(1), slow insert O(n)), while linked lists are non-contiguous nodes connected by pointers (slow access O(n), fast insert O(1) at known position).",
            ],
            "partial_variants": [
                "Arrays have fixed size and linked lists can grow dynamically.",
                "Array is faster for accessing elements, linked list is better for inserting.",
                "Arrays store data next to each other, linked lists use pointers.",
                "Linked lists don't need contiguous memory unlike arrays.",
            ],
            "incorrect_variants": [
                "Arrays and linked lists are the same thing but with different syntax.",
                "Linked lists are always faster than arrays for all operations.",
                "Arrays use pointers to connect elements while linked lists store data contiguously.",
                "Both have O(1) access time but arrays use more memory.",
            ],
        },
        {
            "question": "What is the time complexity of binary search and why?",
            "reference": "Binary search has O(log n) time complexity because it divides the search space in half with each comparison. Starting from the middle element, it eliminates half of the remaining elements by comparing with the target, resulting in at most log2(n) comparisons for n elements.",
            "correct_variants": [
                "Binary search is O(log n) because it halves the search space each step. After each comparison with the middle element, half the remaining elements are eliminated, so it takes at most log2(n) steps.",
                "The time complexity is O(log n). Binary search works by comparing the target with the middle element and discarding half the array each time. This halving process means we need log2(n) comparisons in the worst case.",
                "O(log n) — each iteration divides the sorted array in half by checking the middle element. Since the problem size reduces by half each step, the number of steps is logarithmic in n.",
            ],
            "partial_variants": [
                "Binary search is O(log n) because it splits the data.",
                "It's logarithmic time because it doesn't check every element.",
                "Binary search is fast because it goes to the middle first.",
                "The complexity is log n. It divides the array each time.",
            ],
            "incorrect_variants": [
                "Binary search has O(n) time complexity because it checks each element.",
                "The time complexity is O(n log n) because it sorts and then searches.",
                "Binary search is O(1) because it always finds the element immediately.",
                "It's O(n²) because it compares every pair of elements.",
            ],
        },
        {
            "question": "What is a hash table and how does it handle collisions?",
            "reference": "A hash table is a data structure that maps keys to values using a hash function. The hash function converts keys into array indices for O(1) average-case lookup. Collisions occur when two keys hash to the same index. Common collision resolution methods include chaining (storing multiple elements in a linked list at each index) and open addressing (probing for the next available slot).",
            "correct_variants": [
                "A hash table maps keys to values via a hash function that computes array indices, giving O(1) average lookup. When two keys map to the same index (collision), it uses chaining (linked lists at each slot) or open addressing (linear/quadratic probing).",
                "Hash tables use a hash function to convert keys into indices for near-constant-time access. Collisions happen when different keys produce the same index. They're resolved by chaining (each bucket holds a linked list) or open addressing (finding the next empty slot).",
                "It's a key-value data structure using hashing for O(1) average operations. A hash function maps keys to array positions. Collision resolution uses either chaining with linked lists or open addressing with probing techniques.",
            ],
            "partial_variants": [
                "A hash table stores key-value pairs and uses a function to find where to store them.",
                "Hash tables give fast lookup using a hash function. Collisions are handled somehow.",
                "It uses hashing to store data. When two things go to the same place, it chains them.",
                "A hash table is like a dictionary. The hash function is used to find the index.",
            ],
            "incorrect_variants": [
                "A hash table sorts elements in order and uses binary search to find them.",
                "Hash tables are the same as arrays but with string keys.",
                "Collisions never happen if you use a good hash function.",
                "A hash table is a balanced binary tree that organizes data by hash values.",
            ],
        },
        {
            "question": "Explain how BFS (Breadth-First Search) traverses a graph.",
            "reference": "BFS explores a graph level by level starting from a source vertex. It uses a queue data structure: the source is enqueued first, then while the queue is not empty, it dequeues a vertex, marks it as visited, and enqueues all unvisited adjacent vertices. This ensures vertices are visited in order of their distance from the source. BFS has O(V+E) time complexity.",
            "correct_variants": [
                "BFS starts at the source vertex and explores all neighbors before moving to the next level. It uses a queue: enqueue the source, then repeatedly dequeue a vertex, visit it, and enqueue its unvisited neighbors. Time complexity is O(V+E).",
                "BFS traverses a graph level-by-level using a FIFO queue. Starting from the source, it visits all vertices at distance 1, then distance 2, etc. Each vertex is dequeued, marked visited, and its unvisited neighbors are enqueued. Runs in O(V+E).",
                "Breadth-First Search explores the graph layer by layer. Using a queue, it starts at the source node, processes each vertex by visiting its adjacent unvisited nodes and adding them to the queue. This gives O(V+E) complexity where V is vertices and E is edges.",
            ],
            "partial_variants": [
                "BFS visits all neighbors first before going deeper. It uses a queue.",
                "BFS goes level by level through the graph starting from a node.",
                "It explores the graph breadth-first using a queue data structure.",
                "BFS checks all nodes at the same distance before moving further away.",
            ],
            "incorrect_variants": [
                "BFS uses a stack to go as deep as possible before backtracking.",
                "BFS randomly selects the next vertex to visit from the graph.",
                "BFS sorts the vertices first and then visits them in order.",
                "BFS uses recursion to explore each branch completely before trying another.",
            ],
        },
        {
            "question": "What is recursion and what are its essential components?",
            "reference": "Recursion is a programming technique where a function calls itself to solve smaller subproblems. Every recursive function must have two essential components: a base case (termination condition that stops recursion) and a recursive case (the function calling itself with a smaller/simpler input that progresses toward the base case).",
            "correct_variants": [
                "Recursion is when a function calls itself to break a problem into smaller parts. It needs a base case to stop the recursion and a recursive case where the function calls itself with reduced input moving toward the base case.",
                "A recursive function solves problems by calling itself with smaller inputs. Two essential components: the base case (stopping condition) and the recursive case (self-call with simpler input). Without a base case, it causes infinite recursion and stack overflow.",
                "Recursion means a function invokes itself to solve subproblems. The base case defines when to stop, and the recursive case reduces the problem size with each call. Both are required — missing the base case leads to infinite recursion.",
            ],
            "partial_variants": [
                "Recursion is when a function calls itself repeatedly.",
                "It's a technique where functions call themselves. You need a base case.",
                "Recursion breaks big problems into smaller ones by self-calling.",
                "A recursive function keeps calling itself until some condition is met.",
            ],
            "incorrect_variants": [
                "Recursion is a loop that runs a fixed number of times.",
                "Recursive functions don't need a base case if the input is always valid.",
                "Recursion is the same as iteration but always uses more memory.",
                "Recursion means running the same function on different threads simultaneously.",
            ],
        },
        {
            "question": "Describe how quicksort works and its average time complexity.",
            "reference": "Quicksort is a divide-and-conquer sorting algorithm. It works by selecting a pivot element, partitioning the array into elements less than and greater than the pivot, then recursively sorting the two partitions. The average time complexity is O(n log n) because each partition step takes O(n) and the recursion depth is O(log n) on average.",
            "correct_variants": [
                "Quicksort picks a pivot, partitions elements into those smaller and larger than the pivot, then recursively sorts both partitions. Average case is O(n log n) since partitioning is O(n) and recursion depth averages O(log n).",
                "It's a divide-and-conquer algorithm: choose a pivot element, partition the array so all smaller elements are left and larger are right, then recursively sort each side. Average time complexity is O(n log n), worst case O(n²) with bad pivot choices.",
                "Quicksort selects a pivot and rearranges the array into two groups — less than pivot and greater than pivot. Each group is sorted recursively. On average this takes O(n log n) time because the array roughly halves each level.",
            ],
            "partial_variants": [
                "Quicksort uses a pivot to split the array and sort each half. It's O(n log n).",
                "It picks an element and puts smaller ones before it and larger ones after, then repeats.",
                "Quicksort is a fast sorting algorithm that divides and conquers. Average is n log n.",
                "It partitions the array around a pivot element recursively.",
            ],
            "incorrect_variants": [
                "Quicksort compares every pair of elements and swaps them if needed, like bubble sort but faster.",
                "Quicksort always runs in O(n log n) regardless of the input.",
                "Quicksort works by merging sorted subarrays together after splitting them.",
                "The average complexity is O(n²) because it needs to compare all elements.",
            ],
        },
        {
            "question": "What is a binary search tree (BST) and its key property?",
            "reference": "A binary search tree is a binary tree where each node has at most two children, and for every node, all values in the left subtree are less than the node's value, and all values in the right subtree are greater. This ordering property enables efficient O(log n) average-case search, insertion, and deletion operations.",
            "correct_variants": [
                "A BST is a binary tree where left children are smaller and right children are larger than the parent. This ordering allows O(log n) average search, insert, and delete by eliminating half the tree at each step.",
                "Binary search tree maintains the invariant: left subtree values < node value < right subtree values. This property enables efficient O(log n) operations on average because each comparison eliminates half the remaining nodes.",
                "A BST is a tree where each node has up to 2 children, with left subtree values less than the node and right subtree values greater. Average operations are O(log n) but degrade to O(n) if the tree becomes unbalanced.",
            ],
            "partial_variants": [
                "A BST organizes data so left is smaller and right is bigger.",
                "It's a tree where you can search efficiently because data is sorted.",
                "Binary search tree stores data in a tree with an ordering rule.",
                "BST has the property that left children are less than the parent.",
            ],
            "incorrect_variants": [
                "A BST is a tree where every node has exactly two children.",
                "BST guarantees O(log n) operations in all cases without balancing.",
                "In a BST, the left subtree contains values greater than the root.",
                "A BST is the same as a heap data structure.",
            ],
        },
    ]
    
    samples = []
    label_weights = [0.35, 0.40, 0.25]  # correct, partial, incorrect
    labels = ["correct", "partially_correct_incomplete", "contradictory"]
    
    for i in range(n_samples):
        topic = topics[i % len(topics)]
        label = rng.choice(labels, p=label_weights)
        
        if label == "correct":
            answer = rng.choice(topic["correct_variants"])
            score = round(rng.uniform(4.0, 5.0), 1)
        elif label == "partially_correct_incomplete":
            answer = rng.choice(topic["partial_variants"])
            score = round(rng.uniform(2.0, 3.5), 1)
        else:
            answer = rng.choice(topic["incorrect_variants"])
            score = round(rng.uniform(0.0, 1.5), 1)
        
        # Add realistic noise (typos, truncation)
        if rng.random() < 0.10:
            words = answer.split()
            if len(words) > 4:
                idx = rng.integers(1, len(words) - 1)
                w = words[idx]
                if len(w) > 3:
                    words[idx] = w[:len(w)-1]  # truncate one char
                answer = " ".join(words)
        
        if rng.random() < 0.05:
            # Truncate answer (simulates incomplete typing)
            words = answer.split()
            cut = max(3, len(words) // 2)
            answer = " ".join(words[:cut]) + "..."
        
        samples.append({
            "id": f"cs_{i:04d}",
            "question": topic["question"],
            "reference": topic["reference"],
            "student": answer,
            "label": label,
            "score": score,
            "context": "Computer Science — Data Structures & Algorithms",
        })
    
    rng.shuffle(samples)
    return samples


def create_vietnamese_stem_dataset(n_samples: int = 200, seed: int = 123) -> List[Dict]:
    """
    Create Vietnamese STEM dataset (Science topics, English language).
    
    Uses broader science topics (Biology, Physics, Chemistry, Earth Science)
    with more variation per topic than the original 10-topic synthetic set.
    """
    rng = np.random.default_rng(seed)
    
    topics = [
        {
            "context": "Biology — Cell Division",
            "question": "Explain the difference between mitosis and meiosis.",
            "reference": "Mitosis produces two identical diploid daughter cells for growth and repair, while meiosis produces four genetically unique haploid cells (gametes) for sexual reproduction. Meiosis involves two divisions and crossing over, resulting in genetic diversity.",
            "correct_variants": [
                "Mitosis creates 2 identical diploid cells for growth/repair. Meiosis creates 4 unique haploid gametes through two divisions with crossing over, enabling genetic diversity in sexual reproduction.",
                "Mitosis = 2 identical diploid daughter cells (growth, repair). Meiosis = 4 genetically diverse haploid cells (gametes). Meiosis has two rounds of division and crossing over to increase variation.",
                "The key difference is output: mitosis yields 2 diploid clones for somatic growth, meiosis yields 4 haploid gametes with genetic recombination through crossing over during two division phases.",
            ],
            "partial_variants": [
                "Mitosis makes 2 cells and meiosis makes 4 cells.",
                "Mitosis is for growth and meiosis is for reproduction.",
                "In meiosis the cells have half the chromosomes, in mitosis they have the same number.",
                "They are both cell division but meiosis has two stages.",
            ],
            "incorrect_variants": [
                "Mitosis produces haploid cells while meiosis produces diploid cells.",
                "Both mitosis and meiosis produce identical cells but meiosis is faster.",
                "Mitosis is only in plants and meiosis is only in animals.",
                "Meiosis produces 2 cells and mitosis produces 4 cells.",
            ],
        },
        {
            "context": "Physics — Thermodynamics",
            "question": "State the first law of thermodynamics and give an example.",
            "reference": "The first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another. The total energy of an isolated system remains constant. For example, in a car engine, chemical energy from fuel is converted to thermal energy and then to mechanical energy (kinetic energy of the car).",
            "correct_variants": [
                "Energy cannot be created or destroyed, only converted between forms (conservation of energy). In a car engine, chemical energy in fuel becomes heat, then mechanical energy to move the car.",
                "The first law says total energy in an isolated system is constant — energy only changes forms. Example: a pendulum converts potential energy to kinetic energy and back repeatedly.",
                "Energy is conserved: it transforms but the total stays the same. For instance, an electric heater converts electrical energy to thermal energy. No energy is lost, only transformed.",
            ],
            "partial_variants": [
                "Energy can't be created or destroyed.",
                "The first law is about conservation of energy.",
                "Energy changes form but the total stays the same. Like when you burn wood.",
                "It says energy is conserved in a closed system.",
            ],
            "incorrect_variants": [
                "The first law says energy always increases in a system over time.",
                "Energy can be created from nothing if enough force is applied.",
                "The first law of thermodynamics is about entropy always increasing.",
                "Heat always flows from cold to hot objects according to the first law.",
            ],
        },
        {
            "context": "Chemistry — Chemical Bonding",
            "question": "Explain the difference between ionic and covalent bonds.",
            "reference": "Ionic bonds form when one atom transfers electrons to another, creating oppositely charged ions that attract each other (typically between metals and nonmetals). Covalent bonds form when atoms share electron pairs (typically between nonmetals). Ionic compounds have high melting points and conduct electricity when dissolved, while covalent compounds generally have lower melting points.",
            "correct_variants": [
                "Ionic bonds involve electron transfer between a metal and nonmetal, creating charged ions that attract. Covalent bonds involve electron sharing between nonmetals. Ionic compounds have higher melting points and conduct electricity in solution.",
                "In ionic bonding, electrons are transferred from one atom to another (metal to nonmetal), forming ions. In covalent bonding, atoms share electrons (nonmetal to nonmetal). Ionic compounds are typically crystalline with high melting points.",
                "Ionic bonds = electron transfer → ions with electrostatic attraction (e.g., NaCl). Covalent bonds = shared electron pairs (e.g., H₂O). Ionic compounds conduct electricity when dissolved; covalent ones generally don't.",
            ],
            "partial_variants": [
                "Ionic bonds transfer electrons, covalent bonds share them.",
                "Ionic is between metals and nonmetals, covalent is between nonmetals.",
                "One involves sharing and the other involves giving electrons.",
                "Ionic bonds make ions and covalent bonds make molecules.",
            ],
            "incorrect_variants": [
                "Ionic bonds are stronger than covalent bonds in all cases.",
                "Covalent bonds transfer electrons while ionic bonds share them.",
                "Both types of bonds involve sharing electrons equally.",
                "Ionic bonds only exist between two nonmetal atoms.",
            ],
        },
        {
            "context": "Physics — Mechanics",
            "question": "Explain Newton's third law of motion with an example.",
            "reference": "Newton's third law states that for every action, there is an equal and opposite reaction. When object A exerts a force on object B, object B simultaneously exerts an equal force in the opposite direction on object A. For example, when you push against a wall, the wall pushes back on you with equal force.",
            "correct_variants": [
                "For every action there is an equal and opposite reaction. Forces come in pairs: when you push a wall, the wall pushes back equally. The action force on the wall equals the reaction force on your hand.",
                "Newton's third law: if object A pushes on B, then B pushes back on A with equal magnitude but opposite direction. Example: a rocket expels gas downward (action), and the gas pushes the rocket upward (reaction).",
                "Every force has an equal and opposite reaction force. When you stand on the floor, you push down with your weight and the floor pushes up equally. These action-reaction pairs act on different objects.",
            ],
            "partial_variants": [
                "Every action has an equal and opposite reaction.",
                "When you push something, it pushes back.",
                "Newton's third law is about action and reaction forces.",
                "Forces come in pairs. Like when you jump, the ground pushes you up.",
            ],
            "incorrect_variants": [
                "The third law says the reaction force is always smaller than the action force.",
                "Action and reaction forces cancel each other out so nothing moves.",
                "Newton's third law only applies to stationary objects.",
                "The reaction force acts on the same object as the action force.",
            ],
        },
        {
            "context": "Biology — Ecology",
            "question": "What is an ecosystem and what are its main components?",
            "reference": "An ecosystem is a community of living organisms interacting with each other and their non-living environment. Its main components are biotic factors (living things: producers, consumers, decomposers) and abiotic factors (non-living: sunlight, water, temperature, soil, air). Energy flows through food chains/webs and nutrients are recycled.",
            "correct_variants": [
                "An ecosystem includes all living organisms and non-living factors in an area interacting together. Biotic components are producers, consumers, and decomposers. Abiotic components include sunlight, water, soil, and temperature. Energy flows through food chains.",
                "An ecosystem is the interaction between living (biotic) and non-living (abiotic) components. Biotic: plants (producers), animals (consumers), fungi/bacteria (decomposers). Abiotic: light, water, temperature, minerals. Energy flows and nutrients cycle.",
                "It's a system of living organisms and their physical environment. Components include biotic factors (producers like plants, consumers like animals, decomposers like bacteria) and abiotic factors (water, sunlight, soil, air, temperature).",
            ],
            "partial_variants": [
                "An ecosystem is where living things interact with their environment.",
                "It includes plants, animals, and the environment they live in.",
                "Ecosystems have living and non-living parts.",
                "An ecosystem is a community of organisms and their surroundings.",
            ],
            "incorrect_variants": [
                "An ecosystem only includes living organisms, not non-living things.",
                "Ecosystems are the same as habitats — just the physical place where organisms live.",
                "In an ecosystem, energy is created by decomposers and consumed by producers.",
                "Ecosystems don't include microorganisms, only plants and animals.",
            ],
        },
        {
            "context": "Chemistry — Acids and Bases",
            "question": "What is pH and how does it relate to acidity?",
            "reference": "pH is a logarithmic scale from 0 to 14 that measures the concentration of hydrogen ions (H+) in a solution. pH 7 is neutral, below 7 is acidic (more H+ ions), and above 7 is basic/alkaline (fewer H+ ions). Each unit change represents a tenfold change in H+ concentration.",
            "correct_variants": [
                "pH measures H+ ion concentration on a 0-14 logarithmic scale. pH 7 is neutral, below 7 is acidic (high H+), above 7 is basic (low H+). Each pH unit represents a 10x change in H+ concentration.",
                "It's a 0-14 scale measuring how acidic or basic a solution is based on hydrogen ion concentration. Acidic solutions have pH < 7, neutral is pH 7, basic is pH > 7. The scale is logarithmic — pH 3 is 10 times more acidic than pH 4.",
                "pH is the negative logarithm of hydrogen ion concentration, ranging from 0 to 14. Lower pH means more H+ ions (more acidic), higher pH means fewer H+ ions (more basic). Each unit is a 10x difference.",
            ],
            "partial_variants": [
                "pH tells you if something is acidic or basic. 7 is neutral.",
                "pH goes from 0 to 14, with acids being lower numbers.",
                "It measures how acidic a solution is.",
                "Low pH is acidic, high pH is basic.",
            ],
            "incorrect_variants": [
                "pH measures the temperature of a chemical solution.",
                "pH 0 is neutral and pH 14 is the most acidic.",
                "pH is a linear scale where each unit is the same change.",
                "Bases have a pH below 7 and acids have a pH above 7.",
            ],
        },
        {
            "context": "Physics — Waves",
            "question": "What is the difference between longitudinal and transverse waves?",
            "reference": "In transverse waves, particle displacement is perpendicular to the direction of wave propagation (e.g., light waves, water surface waves). In longitudinal waves, particle displacement is parallel to the wave direction, creating compressions and rarefactions (e.g., sound waves). Both types transfer energy without transferring matter.",
            "correct_variants": [
                "Transverse waves have particle motion perpendicular to wave direction (like light, ripples). Longitudinal waves have parallel particle motion with compressions and rarefactions (like sound). Both transfer energy, not matter.",
                "In transverse waves, particles move up/down while the wave travels horizontally (e.g., electromagnetic waves). In longitudinal waves, particles move back/forth in the same direction as the wave (e.g., sound), creating compression zones.",
                "The difference is in particle motion: transverse = perpendicular to propagation direction (light, water waves), longitudinal = parallel to propagation with compressions/rarefactions (sound waves). Neither type moves matter, only energy.",
            ],
            "partial_variants": [
                "Transverse waves go up and down, longitudinal waves go back and forth.",
                "Sound is longitudinal and light is transverse.",
                "They differ in the direction particles move compared to the wave.",
                "One moves perpendicular and the other parallel to the wave direction.",
            ],
            "incorrect_variants": [
                "Longitudinal waves move perpendicular to the direction of energy flow.",
                "Transverse waves can only travel through solids, not liquids or gases.",
                "There is no real difference — all waves behave the same way.",
                "Sound waves are transverse because they make the air vibrate up and down.",
            ],
        },
        {
            "context": "Biology — Genetics",
            "question": "What is a gene and how does it determine traits?",
            "reference": "A gene is a segment of DNA that contains instructions for making a specific protein. Genes are located on chromosomes and come in different versions called alleles. Proteins produced by genes control cell functions and determine observable traits (phenotype). The combination of alleles (genotype) inherited from both parents determines which traits are expressed.",
            "correct_variants": [
                "A gene is a DNA segment coding for a protein. Located on chromosomes, genes have variants called alleles. The genotype (allele combination from parents) determines which proteins are made, which controls the phenotype (observable traits).",
                "Genes are sections of DNA on chromosomes that encode proteins. Different alleles of a gene produce different proteins, leading to trait variation. Your genotype (inherited allele pairs) determines your phenotype (expressed traits).",
                "A gene is a functional unit of DNA that provides instructions for protein synthesis. Alleles are gene variants inherited from parents. The specific allele combination (genotype) determines which proteins are produced, controlling traits (phenotype).",
            ],
            "partial_variants": [
                "Genes are parts of DNA that control traits.",
                "A gene tells the body how to make proteins which affect traits.",
                "Genes are on chromosomes and you get them from your parents.",
                "Genes determine what you look like. They're made of DNA.",
            ],
            "incorrect_variants": [
                "Genes are made of proteins and they directly create traits without DNA.",
                "Every trait is controlled by exactly one gene with no variation.",
                "Genes can only be inherited from the mother, not the father.",
                "A gene is a type of cell that reproduces to create traits.",
            ],
        },
        {
            "context": "Chemistry — Reactions",
            "question": "What is the law of conservation of mass in chemical reactions?",
            "reference": "The law of conservation of mass states that in a chemical reaction, the total mass of reactants equals the total mass of products. Matter is neither created nor destroyed during a chemical reaction — atoms are simply rearranged to form new substances. This means chemical equations must be balanced.",
            "correct_variants": [
                "Conservation of mass means total mass of reactants equals total mass of products in a chemical reaction. Atoms are rearranged, not created or destroyed, so chemical equations must be balanced to reflect this.",
                "In any chemical reaction, mass is conserved: what goes in equals what comes out. Atoms rearrange to form new compounds but no atoms are gained or lost. That's why we balance equations — to show equal atoms on both sides.",
                "The law states matter cannot be created or destroyed in chemical reactions. The total mass before and after a reaction stays the same because atoms are only rearranged. Balanced equations demonstrate this conservation.",
            ],
            "partial_variants": [
                "Mass is conserved in chemical reactions. Nothing is created or destroyed.",
                "The total mass doesn't change during a reaction.",
                "It means you have the same amount of stuff before and after a reaction.",
                "Mass in equals mass out. That's why equations need to be balanced.",
            ],
            "incorrect_variants": [
                "Mass can be created in exothermic reactions because energy is released.",
                "Some mass is always lost during chemical reactions as heat.",
                "The law only applies to reactions in closed containers.",
                "Conservation of mass means the number of molecules stays the same.",
            ],
        },
        {
            "context": "Physics — Electricity",
            "question": "What is Ohm's law and how are voltage, current, and resistance related?",
            "reference": "Ohm's law states that the voltage (V) across a conductor is directly proportional to the current (I) flowing through it, with resistance (R) as the constant of proportionality: V = IR. This means increasing voltage increases current (for constant resistance), and increasing resistance decreases current (for constant voltage).",
            "correct_variants": [
                "Ohm's law: V = IR. Voltage is proportional to current times resistance. For constant R, more voltage means more current. For constant V, more resistance means less current.",
                "V = IR states that voltage equals current multiplied by resistance. Current is directly proportional to voltage and inversely proportional to resistance. Doubling the voltage doubles the current if resistance stays the same.",
                "Ohm's law relates voltage (V), current (I), and resistance (R): V = IR. Higher voltage drives more current through a circuit, while higher resistance reduces current flow. The relationship is linear for ohmic conductors.",
            ],
            "partial_variants": [
                "V = IR. Voltage equals current times resistance.",
                "Ohm's law relates voltage, current, and resistance.",
                "More resistance means less current flows.",
                "The formula is V = IR and it describes how electricity flows.",
            ],
            "incorrect_variants": [
                "Ohm's law states that current increases when resistance increases.",
                "V = I/R — voltage equals current divided by resistance.",
                "Ohm's law only works for AC circuits, not DC.",
                "Resistance has no effect on current according to Ohm's law.",
            ],
        },
    ]
    
    samples = []
    label_weights = [0.30, 0.45, 0.25]
    labels_list = ["correct", "partially_correct_incomplete", "contradictory"]
    
    for i in range(n_samples):
        topic = topics[i % len(topics)]
        label = rng.choice(labels_list, p=label_weights)
        
        if label == "correct":
            answer = rng.choice(topic["correct_variants"])
            score = round(rng.uniform(4.0, 5.0), 1)
        elif label == "partially_correct_incomplete":
            answer = rng.choice(topic["partial_variants"])
            score = round(rng.uniform(2.0, 3.5), 1)
        else:
            answer = rng.choice(topic["incorrect_variants"])
            score = round(rng.uniform(0.0, 1.5), 1)
        
        # Add realistic noise
        if rng.random() < 0.12:
            words = answer.split()
            if len(words) > 4:
                idx = rng.integers(1, len(words) - 1)
                w = words[idx]
                if len(w) > 3:
                    words[idx] = w[:len(w) - 1]
                answer = " ".join(words)
        
        if rng.random() < 0.04:
            words = answer.split()
            cut = max(3, len(words) // 2)
            answer = " ".join(words[:cut]) + "..."
        
        samples.append({
            "id": f"stem_{i:04d}",
            "question": topic["question"],
            "reference": topic["reference"],
            "student": answer,
            "label": label,
            "score": score,
            "context": topic["context"],
        })
    
    rng.shuffle(samples)
    return samples


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "evaluation", "data")
    os.makedirs(output_dir, exist_ok=True)

    # 1. CS Domain dataset
    print("Creating CS Domain dataset (150 samples)...")
    cs_data = create_cs_domain_dataset(n_samples=150)
    cs_path = os.path.join(output_dir, "cs_domain_dataset.json")
    with open(cs_path, "w", encoding="utf-8") as f:
        json.dump(cs_data, f, indent=2, ensure_ascii=False)
    
    label_dist = {}
    for s in cs_data:
        label_dist[s["label"]] = label_dist.get(s["label"], 0) + 1
    print(f"  Saved to {cs_path}")
    print(f"  Label distribution: {label_dist}")

    # 2. STEM dataset
    print("\nCreating STEM Science dataset (200 samples)...")
    stem_data = create_vietnamese_stem_dataset(n_samples=200)
    stem_path = os.path.join(output_dir, "stem_science_dataset.json")
    with open(stem_path, "w", encoding="utf-8") as f:
        json.dump(stem_data, f, indent=2, ensure_ascii=False)
    
    label_dist = {}
    for s in stem_data:
        label_dist[s["label"]] = label_dist.get(s["label"], 0) + 1
    print(f"  Saved to {stem_path}")
    print(f"  Label distribution: {label_dist}")

    print(f"\n[OK] All datasets created in {output_dir}/")


if __name__ == "__main__":
    main()
