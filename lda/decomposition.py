import numpy as np

from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig

from config import Config
from .tensor import Tensor


class Decomposition ( object ):

	@staticmethod
	def powermethod( M3, inner_iters, outer_iters ):
		"""
			It estimates the eigenvectors and eigenvalues of symmetric tensor M3
			using the tensor decomposition power method ( Anandkumar et. al. 2014 )

			Based on:

				https://github.com/UIUC-data-mining/STOD

			Parameters:
			----------
			M3: a k x k x k tensor representing the trilinear map representation of
				   the modifief third moment of a given corpus
			
			inner_iters - an integer specifying the number of power iterations to perform

			outer_iters - an integer specifying the number of random draws to compute
			
			
			Returns:
			--------
			
			val - a vector of k estimated eigenvalues
			vec - a k x k matrix of estimated eigenvectors
			deflated - the deflated M3 matrix

		"""
		k, _, _ = M3.shape
		T_tilde = Tensor.trilinear( M3 )

		val = np.NINF
		vec = np.zeros( k )

		for tau in range( outer_iters ):
			# draw uniformly at random from the unit sphere
			theta = np.random.normal( size=( k, ) )
			theta /= np.linalg.norm( theta )

			for t in range( inner_iters ):
				# compute power iteration update
				theta_t = T_tilde( np.identity( k ), theta, theta ) # size=(k, )
				vval = theta.dot( theta_t )
				theta += theta_t
				theta /= np.linalg.norm( theta )

			if not val or ( vval > val ):
				val = vval
				vec = theta


		return val, vec, M3 - val*Tensor.tridot( vec )


	@staticmethod
	def e2( E2, k ):
		"""
		It decomposes the empirical second moment 
		as per step 2.2 of Algorithm 2

		and https://github.com/UIUC-data-mining/STOD

		Parameters:
		-----------
		E2 - a sparse, V x V matrix representing
		     the empirical second moment from the corpus
		     where V is the length of the vocabulary. If 
		     E2 is not sparse, it is converted to a sparse
		     matrix.

		k  - a scalar specifying the desired number of
		     eigenpairs to return

		Returns:
		--------

		vals - a k-length array of eigenvalues

		vecs - a V x k matrix of k eigenvectors. The column
		       vecs[:, i] is the eigenvector corresponding
		       to the value vals[i]

		"""
		# we require a sparse second moment matrix 
		if not issparse( E2 ):
			E2 = csr_matrix( E2 )
		# we also require E2 to be symmetric so we can find its
		# eigen decomposition using eigsh
		if ( E2 != E2.T ).toarray().any():
			raise ValueError( 'Second moment is not symmetric.' )
		
		# return the largest k eigenpairs
		return eigsh( E2, k, which='LA' )

	@staticmethod
	def m2_( M2_ ):
		"""
		It computes the spectral decomposition of the modified 
		second moment as per step 2.4 of Algorithm 2

		and https://github.com/UIUC-data-mining/STOD

		Parameters:
		-----------
		
		M2_prime - a k x k matrix representing the modified second moment
				   where k is the number of features/topics

		Returns:
		--------
		
		vals - a k-length array of eigenvalues

		vecs - a k x k matrix of eigenvectors. The column vecs[:, i] 
		       is the eigenvector corresponding to the value vals[i]

		"""
		return eig( M2_ )


			
