import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

from .tensor import Tensor

MIN_DOCUMENT_WORD_LENGTH = 3

class Moments( object ):
	"""
		Implementation of moments as described in:
        
            “Scalable Moment-Based Inference for Latent Dirichlet Allocation” 
            C. Wang, X. Liu, Y. Song, and J. Han, 2014

        and
            https://github.com/UIUC-data-mining/STOD
	
	"""
	@staticmethod
	def first( C ):
		# V = number of words in the vocabulary
		# D = number of documents in the corpus
		V, D = C.shape 
		# inverse of the document lengths
		doc_lengths = 1./np.sum( C, axis=0 )
		# compute M1 eq. (7)
		M1 = np.sum( doc_lengths * C, axis=1 )/D
		return M1

	@staticmethod
	def second( C ):
		V, D = C.shape

		doc_lengths = np.sum( C, axis=0 )

		assert ( doc_lengths >= MIN_DOCUMENT_WORD_LENGTH ).all(), 'At least three words in a document are required to compute up to third moment.'

		lengths_factor = 1./doc_lengths / ( doc_lengths - 1 )
		pr_w_given_d = csr_matrix( np.sqrt( lengths_factor ) * C )
		# compute E2 eq. (7) note that E2 will be sparse
		E2 = pr_w_given_d.dot( pr_w_given_d.T ) - spdiags( C.dot( lengths_factor ), 0, V, V )
		E2 /= D

		return E2	

	@staticmethod
	def third( C, W, M1, alpha0 ):
		V, D = C.shape
		_, k = W.shape

		doc_lengths = np.sum( C, axis=0 )
		rho = 1. / ( doc_lengths * ( doc_lengths - 1. ) * ( doc_lengths - 2. ) )
		RC = rho * C
		WTC = ( W.T.dot( C ) ) # size(WTC) = (k, D)
		A1_ = Tensor.trilinear3( rho, WTC )
		A3_ = Tensor.trilinear3( 2. * np.sum( RC, axis=1 ), W.T )
		E3 = A1_ + A3_
		k2 = k*k
		W2 = np.zeros( ( V, k2 ) )
		for i in range( V ):
			W2[i, :] = np.kron( W[i, :], W[i, :] )
		A2 = WTC.dot( RC.T.dot( W2 ) ) # size(A2) = (k, k2)
		A2 = A2.reshape( ( k, k, k ), order='F' )
		E3 = E3 - A2 - A2.transpose( 1, 0, 2 ) - A2.transpose( 1, 2, 0 )
		WTM1 = W.T.dot( M1 )
		U1 = np.tensordot( np.identity( k ) + alpha0*Tensor.bidot( WTM1 ), WTM1, axes=0 )
		U1U2U3 = U1 + U1.transpose( 0, 2, 1 ) + U1.transpose( 2, 0, 1 )
		M3 = E3/D * ( alpha0 + 1 )*( alpha0 + 2 )/2 - alpha0/2.*U1U2U3  + alpha0*alpha0*Tensor.tridot( WTM1 )

		return M3