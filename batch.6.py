# Update:
# 	Optimization
# 	walkers birth/death

import numpy as np
import numpy.linalg as la
from scipy.misc import comb
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import progressbar

np.random.seed(19260817)
PROB_TYPE = sys.argv[1]
n = int(sys.argv[2])
SAMPLE = 10
WALKER = 10000
PROB = 0.5
dt = 0.1

from helper import classical_solve, compress, HB, HP

# Verify if given computation time guarantees sufficient probability
def run_single (n, k, G, T, arr):
	cnk, rank, knar = compress(n, k)
	H_B = HB(n, k, cnk, knar)
	H_P = HP(n, G, cnk, knar)
	
# AQC evolution
	psi0 = np.array([cnk**(-0.5) for i in range(cnk)])
	for t in np.arange(0.0, T, dt):
		H = (1.0 - t/T) * H_B + t/T * H_P
		psi0 += (-1) * np.dot(H, psi0) * dt
		psi0 /= la.norm(psi0)
	
# Monte Carlo random walk
	# sample walkers from initial wave function amplitude
	walker_cnt = WALKER
	walkers = np.random.randint(cnk, size=walker_cnt)  # random walkers
	log_weights = np.zeros(walker_cnt)  # initial weights are 1
	# random walk
	for t in np.arange(0.0, T, dt):
		# walkers random diffusion
		H = (1.0 - t/T) * H_B + t/T * H_P
		G = np.identity(cnk) - dt * H
		for i in range(walker_cnt):
			walker = walkers[i]  # current value of walker
			weight = np.sum(G[:,walker])  # step weight
			dist = G[:,walker]/weight  # distribution of next values
			walkers[i] = np.random.choice(cnk, p=dist)  # random diffusion
			log_weights[i] += np.log(weight)  # walker weight multiplied by step weight
		log_weights -= np.average(log_weights)  # normalize product of weights to 1

		# split walkers with large weight
		idx_large = (log_weights > np.log(2))
		log_weights[idx_large] -= np.log(2)
		walkers = np.append(walkers, walkers[idx_large])
		log_weights = np.append(log_weights, log_weights[idx_large])

		# kill walkers with small weight
		idx_not_small = (log_weights >= np.log(0.5))
		walkers = walkers[idx_not_small]
		log_weights = log_weights[idx_not_small]
		
		walker_cnt = walkers.size
	# reconstruct wave function from random walkers
	psi = np.zeros(cnk)
	for i in range(walker_cnt):
		psi[walkers[i]] += np.exp(log_weights[i])
	psi /= la.norm(psi)  # normalize wave function

	print(str(la.norm(psi0[rank[arr]])**2) + ' ' + str(la.norm(psi[rank[arr]])**2))
	prob = np.sum(psi[rank[arr]]**2)
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

