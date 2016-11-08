import argparse
from datetime import datetime
import numpy as np
from _test import mult, mult_broken


parser = argparse.ArgumentParser(description='Run test.')
parser.add_argument('-I', type=int, default=10)
parser.add_argument('-J', type=int, default=10)
parser.add_argument('-K', type=int, default=10)
parser.add_argument('-n', '--n_iter', type=int, default=1000)

args = parser.parse_args()
I = args.I
J = args.J
K = args.K
n_iter = args.n_iter
print('a is {} x {}'.format(I, J))
print('b is {} x {}'.format(J, K))

a = np.random.random((I, J))
b = np.random.random((J, K))
c = a.dot(b)
normc = np.linalg.norm(c)
print('norm of a x b: {}'.format(normc))

print()
print('compute a x b {} times using...'.format(n_iter))
for f in [mult, mult_broken]:
    print()
    print(f.__name__)
    n_correct = 0
    n_wrong = 0
    rel_err = 0.
    start_time = datetime.now()
    for i in range(n_iter):
        _c = f(a, b)
        d = np.linalg.norm(c - _c) / normc
        if np.allclose(c, _c):
            n_correct += 1
        else:
            rel_err += d
            n_wrong += 1
    avg_time = (datetime.now() - start_time) / n_iter
    print('number correct: {}'.format(n_correct))
    print('number wrong  : {}'.format(n_wrong))
    print('avg rel error : {}'.format(None if n_wrong == 0 else rel_err / n_wrong))
    print('avg time per  : {}'.format(avg_time))
