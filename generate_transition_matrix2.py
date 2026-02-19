from collections import Counter, defaultdict
from math import factorial
from scipy.sparse import coo_matrix

def  distributions(p, n): # nacini na koje mozemo da izaberemo p cvorova sa ponavljanjem od ukupno n cvorova 
    if n == 1: # kada ne ostane vise izbora, na poslednje mesto stavimo ono sta je ostalo od originalnog p i rekurzija se zavrsava
        yield (p,)
        return
    for i in range(p + 1): # prolazi kroz sve moguce vrednosti prve komponente
        for tail in distributions(p - i, n - 1): # ostatak p - i raspodeljujemo rekurzivno na n-1 mesta
            yield (i,) + tail 


def weight(t, p):
    w = factorial(p)
    for x in t:
        w //= factorial(x)
    return w  #tezinski parametar je broj nacina na koje mozemo dobiti kombinaciju


def generate_transition_matrix2(n, k, a, b, p):
    initial_macrostate = (n - 1,) + (0,) * (k - 2) + (1,) # pocetno stanje
    indices = {initial_macrostate: 0} # dodeljujemo indeks pocetnom stanju
    next_index = 1
    transitions = defaultdict(Counter)
    queue = [initial_macrostate]

    while queue: # while petlja sa slozenoscu (n+k-1) nad (k-1)
        #izvrsava se jednom po svakom makrostanju
        #dokle god je red neprazan, racunamo:
        current_macrostate = queue.pop(0) # uzimamo prvo sledece stanje, a brisemo ga iz reda 
        i_cur = indices[current_macrostate]

        state = []
        for level, count in enumerate(current_macrostate): # iz makrostanja u mikrostanje i iz tuple u list
            state.extend([level] * count) # (2,3,2) -> [0,0,1,1,1,2,2]

        for t in distributions(p, n): # npr. t = (1,0,1,1,0,0,2) svaki element predstavlja koliko je puta odabran dati cvor, u distributions imamo (p+n-1) nad  (n-1) kombinacija
            w = weight(t, p)

            new_state = state.copy()

            for i in range(n): # izvrsavamo pravila interakcije
                if t[i] > 0:
                    new_state[i] = min(new_state[i] + a * t[i], k - 1)
                else:
                    new_state[i] = max(new_state[i] - b, 0) 

            macro = [0] * k 
            for v in new_state: 
                macro[v] += 1 #[2,0,2,1,0,1, 1] -> [2,3,2] vracamo nazad u makrostanje
                
            new_macrostate = tuple(macro) # prebacujemo iz liste u tuple jer je hashable i jednom definisano stanje je nepromenljivo

            if new_macrostate not in indices:  # ako se stanje jos nije pojavljivalo, onda mu dodeljujemo indeks i stavljamo ga poslednjeg na red
                indices[new_macrostate] = next_index
                next_index += 1
                queue.append(new_macrostate)

            transitions[i_cur][indices[new_macrostate]] += w # transitions - 0: Counter({1:3, 2:1}) iz pocetnog stanja tri prelaza u stanje 1 i jedan prelaz u stanje 2
                                                             #               1: Counter({0:1, 3:2, 4:3}) etc.
    dim = len(indices) # dimenzija matrice je broj stanja
    rows, cols, data = [], [], [] #parametri za definisanje elementa u sparse matrici

    for i, counter in transitions.items(): 
        total = sum(counter.values()) # za i-to stanje racunamo ukupan broj prelaza iz stanja u sva druga stanja
        for j, count in counter.items(): # broj prelaza iz i-tog u j-to stanje
            rows.append(i)
            cols.append(j)
            data.append(count / total) # delimo sa ukupnim brojem prelaza da dobijemo verovatnocu prelaza iz stanja i u stanje j 

    P = coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr() # definisemo nenulte elemente u sparse matrici
    return P, indices # vracamo matricu tranzicije zajedno sa indeksiranim stanjima, a sustinski indeksi sa stanjima nam nisu potrebni, jedino zbog brzine izvrsavanja koda
