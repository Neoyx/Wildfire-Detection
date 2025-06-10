import numpy as np
import random
import time


def sequential_regioning(img, n8, random_seed = 20):
    time_start = time.time()
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

                if u > 0 and img[v, u - 1] > 1: # Nachbar links
                    neighbors.append(img[v, u - 1])

                if v > 0 and img[v - 1, u] > 1: # Nachbar oben
                    neighbors.append(img[v - 1, u])
                
                if n8:
                    if u > 0 and v > 0 and img[v - 1, u - 1] > 1: # Nachbar oben links
                        neighbors.append(img[v - 1, u - 1])
                    if u < width - 1 and v > 0 and img[v - 1, u + 1] > 1: # Nachbar oben rechts
                        neighbors.append(img[v - 1, u + 1])    
                
                unique_neigbors = list(set(neighbors))
                if len(unique_neigbors) == 0: # Alle Nachbarn sind Hintergrundpixel
                    img[v, u] = m
                    m += 1
                elif len(unique_neigbors) == 1: # Genau ein Nachbar hat Labelwert größer 1
                    img[v, u] = unique_neigbors[0]
                else: # Mehrere Nachbarnn haben Labelwert größer 1
                    # Nimm einfach den ersten als neuen Label
                    img[v, u] = unique_neigbors[0]
                    k = unique_neigbors[0]
                    for n1 in unique_neigbors:
                        if n1 != k:
                            collisions.add((n1, k))

    # Pass 2 – Resolve Label Collisions  
    L = range(2, m)
    R = [{i} for i in L]
    for (a, b) in collisions:
        for s in R:
            if a in s:
                r_a = s # der set, der gerade a enthält
            if b in s:
                r_b = s # der set, der gerade b enthält
        if r_a != r_b:
            r_a.update(r_b)
            R.remove(r_b) # WARN: vielleicht nur leer machen

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
                        #img[v, u] = min(s)
                        break
    time_end = time.time()
    print(f"Time taken for sequential regioning: {time_end - time_start:.2f} seconds\nRegions found: {len(R)}")
    return out_img