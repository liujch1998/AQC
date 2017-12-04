# Update:
# 	custom Hamiltonian representation
# 	absolute or relative error 0.01 to reduce computation time
# 	faster classical solve assuming k <= 8 when n <= 32
# 	kill simulation if T > 163.84

import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import progressbar

np.random.seed(19260817)
PROB_TYPE = sys.argv[1]
n = int(sys.argv[2])
SAMPLE = 10
WALKER = 10000
PROB = 0.5
dt = 0.01
error = 0.01

from Hamiltonian import Hamiltonian
from helper import classical_solve, compress, HB, HP

# Verify if given computation time guarantees sufficient success probability
def run_single (T, ans, cnk, H_B, H_P):
	print(T)
	# Monte Carlo random walk
	# sample walkers from initial wave function amplitude
	walker_cnt = WALKER
	walkers = np.random.randint(cnk, size=walker_cnt)  # random walkers
	log_weights = np.zeros(walker_cnt)  # initial weights are 1
	# random walk
	for t in np.arange(0.0, T, dt):
		# walkers random diffusion
		H = H_B * (1.0 - t/T) + H_P * (t/T)
		G = H * (-dt) + Hamiltonian.identity(cnk)
		for i in range(walker_cnt):
			walker = walkers[i]  # current value of walker
			weight = G.col_sum(walker)  # step weight
			if np.random.random() >= G.diag_ratio(walker):
				walkers[i] = np.random.choice(G.off_index[walker])
			log_weights[i] += np.log(weight)  # walker weight multiplied by step weight
		log_weights -= np.average(log_weights)  # normalize product of weights to 1

		# split walkers with large weight
		idx_large = (log_weights > 2)
		log_weights[idx_large] -= np.log(2)
		walkers = np.append(walkers, walkers[idx_large])
		log_weights = np.append(log_weights, log_weights[idx_large])

		# kill walkers with small weight
		idx_not_small = (log_weights >= -2)
		walkers = walkers[idx_not_small]
		log_weights = log_weights[idx_not_small]
		
		walker_cnt = walkers.size
	# reconstruct wave function from random walkers
	psi = np.zeros(cnk)
	for i in range(walker_cnt):
		psi[walkers[i]] += np.exp(log_weights[i])
	psi /= np.linalg.norm(psi)  # normalize wave function

	'''
	# AQC evolution
	psi0 = np.array([cnk**(-0.5) for i in range(cnk)])
	for t in np.arange(0.0, T, dt):
		H = (1.0 - t/T) * H_B0 + t/T * H_P0
		psi0 += (-1) * np.dot(H, psi0) * dt
		psi0 /= np.linalg.norm(psi0)
	print(str(np.sum(psi0[ans]**2)) + ' ' + str(np.sum(psi[ans]**2)))
	'''

	prob = np.sum(psi[ans]**2)
	return prob >= PROB

# Run algorithm on a single random graph of size n; return computation time to achieve probability threshold
def run_random (n):
	# Generate random graph G: each edge exists by probability 1/2
	G = np.random.randint(2, size=(n, n))
	G = G ^ G.T  # enforce symmetry and zeros on diagonal
	k, ans = classical_solve(n, G)
	
	cnk, rank, knar = compress(n, k)
	for i in range(len(ans)):
		ans[i] = rank[ans[i]]
	H_B = HB(n, k, cnk, knar, rank)
	H_P = HP(n, G, cnk, knar)
	
	# binary search computation time
	T_min = 0
	T_max = 1
	while not run_single(T_max*dt, ans, cnk, H_B, H_P):
		T_max *= 2
		if T_max == 16384:
			return k, T_max*dt
	while T_max - T_min > 1 and (T_max - T_min) > T_max * error:
		T = (T_min + T_max) // 2
		if run_single(T*dt, ans, cnk, H_B, H_P):
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
sys.stdout = open('result/' + PROB_TYPE + '_' + str(n) + '.txt', 'w')
print('mean: ' + str(np.mean(np.array(ansall))))
print('median: ' + str(np.median(np.array(ansall))))
print(ansall)
print(' ')
for k,v in ans.iteritems():
	print(str(k) + ': ' + str(v))

