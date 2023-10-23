import numpy as np
from scipy.optimize import linprog

def print_results(read_arr):
    results = nash_equilibrium(read_arr)
    
    print('#' * 90)
    print(beautiful_print_vector("strategy 1", results[0]))
    print('#' * 90)
    print(beautiful_print_vector("strategy 2", results[1]))
    print('#' * 90)
    print(beautiful_print_vector("Game cost", results[2]))
    print('#' * 90)
    

def beautiful_print_vector(s, v, max_width=90):
    fill_symbol = '#'
    string = "{:{fill}{align}{width}}"
    s += ": ["

    for i in range(np.shape(v)[0]):
        s += "{:4f} ; ".format(v[i])

    return string.format(' ' + s[:-3] + "] ",
                         fill=fill_symbol, align='^', width=max_width)


def nash_equilibrium(A):
    n = A.shape[0]
    m = A.shape[1]

    b_ub1, b_ub2 = [-1 for i in range(m)], [1 for i in range(n)]
    bnd1 = [(0, float("inf")) for i in range(m)]
    bnd2 = [(0, float("inf")) for i in range(n)]

    add = abs(A.min())
    A_ub1 = np.transpose(A + add) * (-1)
    A_ub2 = A

    ans1 = linprog(
        c=b_ub2,
        A_ub=A_ub1,
        b_ub=b_ub1,
        bounds=bnd2,
        method="simplex")

    ans2 = linprog(
        c=b_ub1,
        A_ub=A_ub2,
        b_ub=b_ub2,
        bounds=bnd1,
        method="simplex")

    return (ans1.x / abs(ans1.fun),
            ans2.x / abs(ans2.fun), np.array([1 / ans1.fun - add]))


n = int(input())
matrix = [list(map(int, input().split())) for i in range(n)]

print_results(np.array(matrix))
