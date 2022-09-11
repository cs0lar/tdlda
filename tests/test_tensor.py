import logging
import unittest

import numpy as np
from numpy.random import default_rng

from lda.tensor import Tensor

def multilinear_map_naive( A ):
	def _bilinear( V1, V2 ):
		W = np.zeros( ( V1.shape[1], V2.shape[1] ), dtype=np.int32 )

		for i1 in range( W.shape[0] ):
			for i2 in range( W.shape[1] ):
				for j1 in range( V1.shape[0] ):
					for j2 in range( V2.shape[0] ):
						W[i1, i2] += A[j1, j2] * V1[j1, i1] * V2[j2, i2]

		return W 

	def _trilinear( V1, V2, V3 ):
		W = np.zeros( ( V1.shape[1], V2.shape[1], V3.shape[1] ) , dtype=np.int32 )

		for i1 in range( W.shape[0] ):
			for i2 in range( W.shape[1] ):
				for i3 in range( W.shape[2] ):
					for j1 in range( V1.shape[0] ):
						for j2 in range( V2.shape[0] ):
							for j3 in range( V3.shape[0] ):
								W[i1, i2, i3] += A[j1, j2, j3] * V1[j1, i1] * V2[j2, i2] * V3[j3, i3]
		return W

	def _map( *args ):
		numargs = len( args )
		
		if len( args ) > 3 or len( args ) < 2:
			raise ValueError( 'only bilinear or trilinear maps are supported' ) 

		if numargs == 2:
			return _bilinear( *args )

		return _trilinear( *args )

	return _map

class TestTensor( unittest.TestCase ):
	def setUp( self ):
		self.rng = default_rng( 12345 )

	def testMultilinearMappings( self ):
		_A = self.rng.integers( 0, 20, size=( 3, 3 ) )
		V1 = self.rng.integers( 0, 20, size=( 3, 2 ) )
		V2 = self.rng.integers( 0, 20, size=( 3, 4 ) )
		V3 = self.rng.integers( 0, 20, size=( 3, 2 ) )

		A = multilinear_map_naive( _A )
		AA = Tensor.bilinear( _A )

		expected = A( V1, V2 )
		actual = AA( V1, V2 )

		np.testing.assert_array_equal( expected, actual )

		_A = self.rng.integers( 0, 20, size=( 3, 3, 3 ) )

		A = multilinear_map_naive( _A )
		AA = Tensor.trilinear( _A )
		
		expected = A( V1, V2, V3 )
		actual = AA( V1, V2, V3 )

		np.testing.assert_array_equal( expected, actual )


if __name__ == '__main__':
	unittest.main()