import numpy as np

class Tensor ( object ):
	@staticmethod
	def bilinear( A ):
		# bilinear means 2D!
		assert A.ndim == 2, 'A should be a 2D matrix'

		def _map( V1, V2 ):
			return V1.transpose().dot( A ).dot( V2 )

		return _map

	@staticmethod
	def trilinear( A ):
		assert A.ndim == 3, 'A should be a 3D matrix'

		def _map( V1, V2, V3 ):
			p1 = np.tensordot( V1, A, axes=( 0, 0 ) if V1.ndim > 1 else 1 )
			p2 = np.tensordot( p1, V2, axes=( 1, 0 ) if V2.ndim > 1 else 1 )

			return np.tensordot( p2, V3, axes=( 1, 0 ) if V3.ndim > 1 else 1 )
			
		return _map
		
	@staticmethod
	def tridot( A ):
		return np.tensordot( A, np.tensordot( A, A, axes=0 ), axes=0 )

	@staticmethod
	def bidot( A ):
		return np.tensordot( A, A, axes=0 )

	@staticmethod
	def trilinear3( v, M ):
		"""
			Based on: https://github.com/UIUC-data-mining/STOD
		"""
		m, n = M.shape

		x = np.zeros( shape=( m*m*m, ) )

		for i in range( n ):
			mi = M[:, i]
			p = mi * mi[:, np.newaxis]
			p = mi * p.flatten( 'F' )[:, np.newaxis]
			x = x + v[i]*p.flatten( 'F' )

		return x.reshape( ( m, m, m ), order='F' )