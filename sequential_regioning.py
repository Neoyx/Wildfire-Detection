import numpy as np
import random
import time


def sequential_regioning(img, n8, random_seed = 20):
    time_start = time.time_ns()
    img = img.copy()
    (height, width) = img.shape
    out_img = np.zeros((height, width, 3), dtype=np.uint16)
    m = 2
    collisions = set()

    # Pass 1 – Assign Initial Labels
    for v in range(height):
        for u in range(width):
            if img[v, u] == 1:
                neighbors = []

                if u > 0 and img[v, u - 1] > 1: # left neighbor
                    neighbors.append(img[v, u - 1])

                if v > 0 and img[v - 1, u] > 1: # upper neighbor
                    neighbors.append(img[v - 1, u])
                
                if n8:
                    if u > 0 and v > 0 and img[v - 1, u - 1] > 1: # upper left neighbor
                        neighbors.append(img[v - 1, u - 1])
                    if u < width - 1 and v > 0 and img[v - 1, u + 1] > 1: # upper right neighbor
                        neighbors.append(img[v - 1, u + 1])    
                
                unique_neigbors = list(set(neighbors))

                # if there are no neighbors, we assign a new label
                if len(unique_neigbors) == 0:
                    img[v, u] = m
                    m += 1
                # just one neighbor, we can assign the same label
                elif len(unique_neigbors) == 1:
                    img[v, u] = unique_neigbors[0]
                # more than one neighbor, we have a collision
                else:
                    img[v, u] = unique_neigbors[0] # just assign the first neighbor's label
                    k = unique_neigbors[0]
                    for n1 in unique_neigbors:
                        if n1 != k:
                            collisions.add((n1, k))

    # Pass 2 – Resolve Label Collisions  
    L = range(2, m)
    R = [{i} for i in L] # R is a list of sets, each set contains the labels of a region
    for (a, b) in collisions:
        for s in R:
            if a in s:
                r_a = s # the set that currently contains a
            if b in s:
                r_b = s # the set that currently contains b
        if r_a != r_b:
            r_a.update(r_b)
            R.remove(r_b)

    # Randomly assign colors to the base labels
    random.seed(random_seed)
    label_to_color = {}
    for s in R:
        base_label = min(s)
        label_to_color[base_label] = [random.randint(0, 255) for _ in range(3)]

    # Pass 3 - Relabel the Image    
    for v in range(height):
        for u in range(width):
            if img[v, u] > 1:
                for i, s in enumerate(R):
                    if img[v, u] in s:
                        base_label = min(s)
                        out_img[v, u] = label_to_color[base_label]
                        break

    time_end = time.time_ns()
    elapsed_ms = (time_end - time_start) // 1_000_000  # Convert to milliseconds

    print( "sequential regioning:     Regions found:", len(R))
    print(f"sequential regioning:     Elapsed time: {elapsed_ms} milliseconds")
    return out_img, len(R)