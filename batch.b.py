# Update: state vector only includes states with proper Hamming weight

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.misc import comb
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import progressbar

PROB_TYPE = sys.argv[1]
n = int(sys.argv[2])
SAMPLE = 1
PROB = 0.5
dt = 0.001
np.random.seed(123698745)

def expr_g (gij):
	return {
		'max_clique':		1-gij, 
		'max_vertex_indep':	gij, 
		'min_vertex_cover': gij, 
		'min_unknown':		1-gij
	}[PROB_TYPE]
def expr_z (z):
	return {
		'max_clique':		z, 
		'max_vertex_indep': z, 
		'min_vertex_cover': 1-z, 
		'min_unknown':		1-z
	}[PROB_TYPE]

# Get Hamming weight of binary number
def Hamming (num):
	return bin(num).count('1')

# Get idx-th bit from right
def Bit (num, idx):
	return (num >> idx) & 1

# Solve problem classically; return size of answer and array of answers
def classical_solve (n, G):
	return {
		'max_clique':		classical_solve_max(n, G), 
		'max_vertex_indep':	classical_solve_max(n, G), 
		'min_vertex_cover':	classical_solve_min(n, G), 
		'min_unknown':		classical_solve_min(n, G)
	}[PROB_TYPE]
def classical_solve_max (n, G):
	k = 0
	arr = []
	for z in range(2**n):
		flag = True
		for i in range(n):
			for j in range(n):
				if i != j and expr_g(G[i][j])*expr_z(Bit(z, i))*expr_z(Bit(z, j)):
					flag = False
		if flag:
			if k < Hamming(z):
				arr = []
				k = Hamming(z)
			if k == Hamming(z):
				arr.append(z)
	return k, arr
def classical_solve_min (n, G):
	k = n+1
	arr = []
	for z in range(2**n):
		flag = True
		for i in range(n):
			for j in range(n):
				if i != j and expr_g(G[i][j])*expr_z(Bit(z, i))*expr_z(Bit(z, j)):
					flag = False
		if flag:
			if k > Hamming(z):
				arr = []
				k = Hamming(z)
			if k == Hamming(z):
				arr.append(z)
	return k, arr

def diff (n, x, y):
	ans = 0
	for b in range(n):
		if Bit(x, b) != Bit(y, b):
			ans += 1
	return ans

def HB (n, k, cnk, knar):
	H_B = np.zeros((cnk, cnk))
	for i in range(cnk):
		for j in range(cnk):
			if i == j:
				H_B[i][j] = (k * (k-1) + (n-k) * (n-k-1)) / 2
			elif diff(n, knar[i], knar[j]) == 2:
				H_B[i][j] = 1
	return -H_B

# Verify if given computation time guarantees sufficient probability
def run_single (n, k, G, T, arr, debug=False):
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
#	H_B = np.identity(cnk) - np.outer(psi, psi)
	H_B = HB(n, k, cnk, knar)
	H_P = np.diag(np.array([(sum([(expr_g(G[i][j]))*expr_z(Bit(knar[ii], i))*expr_z(Bit(knar[ii], j)) for i in range(n) for j in range(i)])) for ii in range(cnk)], dtype=complex))

	for t in np.arange(0.0, T, dt):
		H = (1.0 - t/T) * H_B + t/T * H_P
		if sys.argv[3] == '0':
			H_ = (1.0 - (t+dt/2)/T) * H_B + (t+dt/2)/T * H_P
			H__ = (1.0 - (t+dt)/T) * H_B + (t+dt)/T * H_P
			k1 = (-1j) * np.dot(H, psi)
			k2 = (-1j) * np.dot(H_, psi+(dt/2)*k1)
			k3 = (-1j) * np.dot(H_, psi+(dt/2)*k2)
			k4 = (-1j) * np.dot(H__, psi+dt*k3)
			psi += (k1 + 2*k2 + 2*k3 + k4) / 6 * dt
		elif sys.argv[3] == '1':
			psi += (-1j) * np.dot(H, psi) * dt
		#	psi /= la.norm(psi)
		elif sys.argv[3] == '2':
			tao = dt
			psi += (-1) * np.dot(H, psi) * tao
			psi /= la.norm(psi)
		else:
			tao = dt * 0.301
			psi = np.dot(sla.expm(-tao * H), psi)
			psi /= la.norm(psi)
		if debug == True:
			print(H)
			print(abs(psi))
# N=4 0.301
# N=5 0.469
# N=6 0.210
	
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
	
	print(G)
	run_single(n, k, G, 5.0, arr, True)
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
sys.stdout = open('output_projector/' + PROB_TYPE + '_' + str(n) + '.txt', 'w')
print('mean: ' + str(np.mean(np.array(ansall))))
print('median: ' + str(np.median(np.array(ansall))))
print(ansall)
print(' ')
for k,v in ans.iteritems():
	print(str(k) + ': ' + str(v))

