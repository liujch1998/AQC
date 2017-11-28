import numpy as np
import sys

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

def compress (n, k):
	rank = np.zeros(2**n, dtype=int)
	knar = np.array([], dtype=int)
	pos = 0
	for z in range(2**n):
		if Hamming(z) == k:
			rank[z] = pos
			knar = np.append(knar, z)
			pos += 1
	cnk = pos
	return cnk, rank, knar

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
				H_B[i][j] = -(k * (k-1) + (n-k) * (n-k-1)) / 2
			elif diff(n, knar[i], knar[j]) == 2:
				H_B[i][j] = -1
	return H_B

def HP (n, G, cnk, knar):
	H_P = np.diag([(sum([(expr_g(G[i][j]))*expr_z(Bit(knar[ii], i))*expr_z(Bit(knar[ii], j)) for i in range(n) for j in range(i)])) for ii in range(cnk)])
	return H_P

