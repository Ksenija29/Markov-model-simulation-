import numpy as np
from collections import Counter, defaultdict
from itertools import product
from scipy.sparse import coo_matrix

def generate_transition_matrix(n, k, a, b, p):

    initial_macrostate = (n - 1,) + tuple([0] * (k - 2)) + (1,)

    indices = {initial_macrostate: 0}
    next_index = 1
    transitions = defaultdict(Counter)
    queue = [initial_macrostate]

    while queue:
        current_macrostate = queue.pop(0)

        state = []
        for level, count in enumerate(current_macrostate):
            state.extend([level] * count)


        for choice in product(range(n), repeat=p):
            counter = Counter(choice)

            new_state = state.copy()


            for idx, times in counter.items():
                new_state[idx] = min(new_state[idx] + a * times, k - 1)

            for idx in range(n):
                if idx not in counter:
                    new_state[idx] = max(new_state[idx] - b, 0)


            macro = [0] * k
            for value in new_state:
                macro[value] += 1

            new_macrostate = tuple(macro)

            if new_macrostate not in indices:
                indices[new_macrostate] = next_index
                next_index += 1
                queue.append(new_macrostate)

            transitions[indices[current_macrostate]][indices[new_macrostate]] += 1

    dim = len(indices)
    rows, cols, data = [], [], []

    for i, counter in transitions.items():
        total = sum(counter.values())
        for j, count in counter.items():
            rows.append(i)
            cols.append(j)
            data.append(count / total)

    P = coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    return P, indices
