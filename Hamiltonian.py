import numpy as np

class Hamiltonian:
	def __init__ (self, on_diag, off_diag, off_size = 0):
		self.on_diag = on_diag
		self.off_diag = off_diag
		self.off_size = off_size

	def __add__ (self, H):
		return Hamiltonian(
			self.on_diag + H.on_diag, 
			self.off_diag + H.off_diag, 
			self.off_size)
	
	def __mul__ (self, c):
		return Hamiltonian(
			self.on_diag * c, 
			self.off_diag * c, 
			self.off_size)
	
	def col_sum (self, col):
		return self.on_diag[col] + self.off_diag * self.off_size
	
	def diag_ratio (self, col):
		return self.on_diag[col] / self.col_sum(col)
	
	@staticmethod
	def identity (size):
		return Hamiltonian(np.repeat(1, size), 0)

