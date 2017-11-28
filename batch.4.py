# Update:
# 	H_B as in paper
# 	Deprecate Runge Kutta
# 	Projector

import numpy as np
import numpy.linalg as la
from scipy.misc import comb
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import progressbar

PROB_TYPE = sys.argv[1]
n = int(sys.argv[2])
SAMPLE = 100
PROB = 0.5
dt = 0.1
np.random.seed(123698745)

from helper import expr_g, expr_z, Hamming, Bit, classical_solve, HB

# Verify if given computation time guarantees sufficient probability
def run_single (n, k, G, T, arr):
	rank = np.zeros(2**n, dtype=int)
	knar = []
	now = 0
	for z in range(2**n):
		if Hamming(z) == k:
			rank[z] = now
			knar.append(z)
			now += 1
	cnk = int(comb(n, k))

	psi = np.array([cnk**(-0.5) for i in range(cnk)], dtype=complex)
	H_B = HB(n, k, cnk, knar)
	H_P = np.diag(np.array([(sum([(expr_g(G[i][j]))*expr_z(Bit(knar[ii], i))*expr_z(Bit(knar[ii], j)) for i in range(n) for j in range(i)])) for ii in range(cnk)], dtype=complex))

	for t in np.arange(0.0, T, dt):
		H = (1.0 - t/T) * H_B + t/T * H_P
		psi += (-1) * np.dot(H, psi) * dt
		psi /= la.norm(psi)
	
	prob = 0.0
	for z in arr:
		prob += (abs(psi[rank[z]]))**2
	return prob >= PROB

# Run algorithm on a single random graph of size n; return computation time to achieve probability threshold
def run_random (n):
	# Generate random graph: each edge exists by probability 1/2
	# G: adjacency matrix
	G = np.random.randint(2, size=(n, n))
	G = G ^ G.T  # enforce symmetry and zeros on diagonal
	k, arr = classical_solve(n, G)
	
	# binary search computation time
	T_min = 0
	T_max = 1
	while run_single(n, k, G, T_max*dt, arr) == False:
		T_max *= 2
	while T_max - T_min > 1:
		T = int((T_min + T_max) / 2)
		if run_single(n, k, G, T*dt, arr):
			T_max = T
		else:
			T_min = T
	return k, T_max*dt

ansall = []
ans = {}
for k in range(0, n+1):
	ans[k] = []
pbar = progressbar.ProgressBar(widgets=[
		progressbar.Percentage(), ' ', 
		progressbar.Bar(marker='>', fill='-'), ' ', 
		progressbar.ETA(), ' ', 
	])
for it in pbar(range(SAMPLE)):
	k, T = run_random(n)
	ansall.append(T)
	ans[k].append(T)
sys.stdout = open('output/' + PROB_TYPE + '_' + str(n) + '.txt', 'w')
print('mean: ' + str(np.mean(np.array(ansall))))
print('median: ' + str(np.median(np.array(ansall))))
print(ansall)
print(' ')
for k,v in ans.iteritems():
	print(str(k) + ': ' + str(v))

