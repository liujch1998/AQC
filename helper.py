import numpy as np
from scipy.misc import comb
import sys

from Hamiltonian import Hamiltonian

PROB_TYPE = sys.argv[1]

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
	for k in range(8, 0, -1):
		ans = []
		cnk, rank, knar = compress(n, k)
		for z in knar:
			flag = True
			index = []
			for i in range(n):
				if expr_z(Bit(z, i)):
					index.append(i)
			for i in index:
				for j in index:
					if i != j and expr_g(G[i][j]):
						flag = False
						break
				if not flag:
					break
			if flag:
				ans.append(z)
		if ans != []:
			break
	return k, ans
	'''
	k = 0
	ans = []
	for z in range(2**n):
		flag = True
		for i in range(n):
			for j in range(i):
				if expr_g(G[i][j]) and expr_z(Bit(z, i)) and expr_z(Bit(z, j)):
					flag = False
					break
			if not flag:
				break
		if flag:
			if k < Hamming(z):
				ans = []
				k = Hamming(z)
			if k == Hamming(z):
				ans.append(z)
	return k, ans
	'''
def classical_solve_min (n, G):
	k = n+1
	ans = []
	for z in range(2**n):
		flag = True
		for i in range(n):
			for j in range(i):
				if expr_g(G[i][j]) and expr_z(Bit(z, i)) and expr_z(Bit(z, j)):
					flag = False
					break
			if not flag:
				break
		if flag:
			if k > Hamming(z):
				ans = []
				k = Hamming(z)
			if k == Hamming(z):
				ans.append(z)
	return k, ans

# Compute subset information
# Input
# 	n: size of set
# 	k: size of subset
# Output
# 	cnk: n choose k
# 	rank: map from binary number to rank in subset list
# 	knar: map from rank in subset list to binary number
def compress (n, k):
	cnk = int(comb(n, k))
	rank = {}
	knar = np.zeros(cnk, dtype=int)
	pos = 0
	z = (1 << k) - 1
	while z < (1 << n):
		rank[z] = pos
		knar[pos] = z
		pos += 1
		x = z & -z
		y = z + x
		z = (((z & ~y) / x) >> 1) | y
	return cnk, rank, knar

# Compute Hamiltonian beginning
# Complexity: cnk * n^2
def HB (n, k, cnk, knar, rank, off_index):
	on_diag = np.repeat(-(k*(k-1)+(n-k)*(n-k-1)), cnk)
	off_diag = -1
#	off_index = np.zeros((cnk, k*(n-k)))
	for ii in range(cnk):
		pos = 0
		for i in range(n):
			for j in range(n):
				if Bit(knar[ii], i) == 1 and Bit(knar[ii], j) == 0:
					off_index[ii][pos] = rank[knar[ii] - (1<<i) + (1<<j)]
					pos += 1
	H_B = Hamiltonian(on_diag, off_diag, k*(n-k))
	return H_B

def HP (n, k, cnk, knar, G):
	on_diag = np.array([(sum([(expr_g(G[i][j]))*expr_z(Bit(knar[ii], i))*expr_z(Bit(knar[ii], j)) for i in range(n) for j in range(i)])) for ii in range(cnk)])
	off_diag = 0
	H_P = Hamiltonian(on_diag, off_diag, k*(n-k))
	return H_P

'''
def classical_solve_clique (n, G):
	return nx.algorithms.clique.graph_clique_number(nx.from_numpy_matrix(G)), []

def is_clique (z, G):
	n = G.shape[0]
	flag = True
	for i in range(n):
		for j in range(n):
			if i != j and expr_g(G[i][j])*expr_z(Bit(z, i))*expr_z(Bit(z, j)):
				flag = False
	return flag	

def HB_csc (n, k, cnk, knar, rank):
	data = []
	row_ind = []
	col_ind = []
	for ii in range(cnk):
		data.append(-(k * (k-1) + (n-k) * (n-k-1)) / 2)
		row_ind.append(ii)
		col_ind.append(ii)
		for i in range(n):
			for j in range(n):
				if Bit(knar[ii], i) == 1 and Bit(knar[ii], j) == 0:
					data.append(-1)
					row_ind.append(ii)
					col_ind.append(rank[knar[ii] - (1<<i) + (1<<j)])
	H_B = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(cnk, cnk))
	return H_B

def HP_csc (n, G, cnk, knar):
	data = []
	row_ind = []
	col_ind = []
	for ii in range(cnk):
		data.append(sum([(expr_g(G[i][j]))*expr_z(Bit(knar[ii], i))*expr_z(Bit(knar[ii], j)) for i in range(n) for j in range(i)]))
		row_ind.append(ii)
		col_ind.append(ii)
	H_P = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(cnk, cnk))
	return H_P
'''
def HB_diff (n, x, y):
	ans = 0
	for b in range(n):
		if Bit(x, b) != Bit(y, b):
			ans += 1
	return ans

def HB_dense (n, k, cnk, knar):
	H_B = np.zeros((cnk, cnk))
	for i in range(cnk):
		for j in range(cnk):
			if i == j:
				H_B[i][j] = -(k * (k-1) + (n-k) * (n-k-1)) / 2
			elif HB_diff(n, knar[i], knar[j]) == 2:
				H_B[i][j] = -1
	return H_B

def HP_dense (n, G, cnk, knar):
	H_P = np.diag([(sum([(expr_g(G[i][j]))*expr_z(Bit(knar[ii], i))*expr_z(Bit(knar[ii], j)) for i in range(n) for j in range(i)])) for ii in range(cnk)])
	return H_P


