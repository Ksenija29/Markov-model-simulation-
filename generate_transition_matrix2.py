from collections import Counter, defaultdict
from math import factorial
from scipy.sparse import coo_matrix

def  distributions(p, n):
    if n == 1:
        yield (p,)
        return
    for i in range(p + 1):
        for tail in distributions(p - i, n - 1):
            yield (i,) + tail


def weight(t, p):
    w = factorial(p)
    for x in t:
        w //= factorial(x)
    return w


def generate_transition_matrix2(n, k, a, b, p):
    initial_macrostate = (n - 1,) + (0,) * (k - 2) + (1,)
    indices = {initial_macrostate: 0}
    next_index = 1
    transitions = defaultdict(Counter)
    queue = [initial_macrostate]

    while queue:
        current_macrostate = queue.pop(0)
        i_cur = indices[current_macrostate]

        state = []
        for level, count in enumerate(current_macrostate):
            state.extend([level] * count)

        for t in distributions(p, n):
            w = weight(t, p)

            new_state = state.copy()

            for i in range(n):
                if t[i] > 0:
                    new_state[i] = min(new_state[i] + a * t[i], k - 1)
                else:
                    new_state[i] = max(new_state[i] - b, 0)

            macro = [0] * k
            for v in new_state:
                macro[v] += 1
            new_macrostate = tuple(macro)

            if new_macrostate not in indices:
                indices[new_macrostate] = next_index
                next_index += 1
                queue.append(new_macrostate)

            transitions[i_cur][indices[new_macrostate]] += w

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
