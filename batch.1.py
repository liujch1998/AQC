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
T_EPS = 0.0001
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

# Verify if given computation time guarantees sufficient probability
def run_single (n, k, G, T, arr):
	psi = np.array([comb(n, k)**(-0.5) if Hamming(z)==k else 0 for z in range(2**n)], dtype=complex)  # uniform superposition of eligible states
	H_B = np.diag(np.array([(Hamming(z) - k)**2 for z in range(2**n)], dtype=complex))
	H_P = np.diag(np.array([(sum([(1-expr_g(G[i][j]))*expr_z(Bit(z, i))*expr_z(Bit(z, j)) for i in range(n) for j in range(i)])) for z in range(2**n)], dtype=complex))
	
	for t in np.arange(0.0, T, dt):
		H = (1.0 - t/T) * H_B + t/T * H_P
		psi += (-1j) * np.dot(H, psi) * dt
		psi /= la.norm(psi)
	
	prob = 0.0
	for z in arr:
		prob += (abs(psi[z]))**2
	return prob >= PROB

# Run algorithm on a single random graph of size n; return computation time to achieve probability threshold
def run_random (n):
	# Generate random graph: each edge exists by probability 1/2
	# G: adjacency matrix
	G = np.random.randint(2, size=(n, n))
	G = G ^ G.T  # enforce symmetry and zeros on diagonal
	k, arr = classical_solve(n, G)
	
	# binary search computation time
	T_min = 0.0
	T_max = 1.0
	while run_single(n, k, G, T_max, arr) == False:
		T_max *= 2
	while T_max - T_min > T_EPS:
		T = (T_min + T_max) / 2
		if run_single(n, k, G, T, arr):
			T_max = T
		else:
			T_min = T
	return k, T_max

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
	ans[k].append(T)
sys.stdout = open(str(n) + '.txt', 'w')
for k,v in ans.iteritems():
	print(str(k) + ': ' + str(v))
#print(ans)
#plt.hist(ans, bins=20, range=(0,60), normed=True)

