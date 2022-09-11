import unittest
import os
import warnings

import numpy as np
from numpy.random import default_rng

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy import stats

from lda.lda import TensorDecompositionLDA

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud

from config import Config 
logging = Config.logging

def topicRecoveryError( Beta, PhiGrnd ):
	"""
		Parameters:
		-----------
		Beta     - a d x k matrix of estimated word distributions for each of the k topics
		PhiGrnd - a d x k matrix of the true word distributions for each topic

		Returns:
		--------
		error - a scalar representing the recovery error of the estimated Beta with respect to the ground truth
	"""
	# compute the full pairwise L1 distance between
	# the ground beta and the estimated Beta
	cost = distance_matrix( Beta.T, PhiGrnd.T, p=1 )
	# construct a bipartite graph with negative L1 distances as edge weights
	# and solve the linear assignment problem to match estimated topics
	# with ground truth topics
	row_ind, col_ind = linear_sum_assignment( cost )
	# average the k L1 distances between matched pairs to compute the error
	return np.average( cost[ row_ind, col_ind ] )


class TestTensorDecompositionLDA( unittest.TestCase ):
	def setUp( self ):
		self.rng = default_rng( 12018231 )
		self.assetpath = os.path.join( os.getcwd(), 'tests/assets' )

	def testTensorDecompositionLDA( self ):
		# pick the number of topics to infer
		k = 50
		# pick the size of the corpus
		num_documents = self.rng.integers( 100, 200 )
		# pick the size of the vocabulary
		num_words = 500
		# pick model parameter alpha0
		alpha0 = 1.
		# create the corpus as a word frequency matrix
		C = self.rng.integers( 0, 3, size=( num_words, num_documents ) )
		# create tensor LDA with default arguments
		TLDA = TensorDecompositionLDA( k, alpha0 )
		# train
		alphas, Phi = TLDA.fit( C )
		# the alphas should sum to alpha0
		self.assertEqual( ( num_words, k ), Phi.shape )
		# each column of Phi should be a distribution over words representing a topic
		np.testing.assert_array_almost_equal( np.ones( k ), np.sum( Phi, axis=0 ) )
		self.assertAlmostEqual( alpha0, np.sum( alphas ) )

	def testTDLDAInference( self ):
		"""
			Evaluation using synthetic data from Wang et.al. 2014
		"""
		k       = 50	# the number of topics
		V       = 3000 # the size of the vocabulary
		eta     = 100 # the Poisson distribution parameter for drawing document lengths
		alpha   = 1./k * np.ones( k ) # the Dirichlet parameters for drawing topic distributions
		Beta    = 200./V * np.ones( shape=( V, k ) ) # the Dirichlet parameters for drawing conditional word distributions
		D       = 5000 # number of documents to generate
		PhiBase = np.tile( 1./V * np.ones( V ), ( k, 1 ) ).T # uniform distribution as baseline

		iters = 5
		baseline_errors = []
		tdlda_errors = []
		slda_errors = []

		for t in range( iters ):
			# generate the synthetic corpus and get the true Beta
			C, PhiGrnd = TensorDecompositionLDA.sample( eta, alpha, Beta, k, D)
			# run the inference algorithm
			TLDA = TensorDecompositionLDA( k, np.sum( alpha ) )
			alphas, Phi = TLDA.fit( C )
			# compute the error
			tdlda_errors.append( topicRecoveryError( Phi, PhiGrnd ) )
			baseline_errors.append( topicRecoveryError( PhiBase, PhiGrnd ) )

		# compute error stats
		S = stats.describe( tdlda_errors )
		S_baseline = stats.describe( baseline_errors )
		# we should be doing better than random!!
		self.assertLess( S.mean, S_baseline.mean )
		
		# plot the stats
		fig, ax = plt.subplots()
		x_pos = np.arange( 2 )
		ax.bar( x_pos, [ S.mean, S_baseline.mean ], yerr=[ np.sqrt( S.variance ), np.sqrt( S_baseline.variance ) ], align='center', alpha=0.5, ecolor='black', capsize=10)
		ax.set_ylabel( '$L_1$ error' )
		ax.set_xticks( x_pos )
		ax.set_xticklabels( [ 'TDLDA', 'Uniform' ] )
		ax.yaxis.grid( True )

		# Save the figure and show
		plt.tight_layout()
		plt.title( 'Topic Recover Error' )
		plt.savefig( 'topic_recovery_error.png' )


	def testTLDANYTDataset( self ):
		# create the vocabulary map
		vocab_file  = 'nyt_vocab.dat.txt'
		corpus_file = 'nyt_data.txt'
		V = []
		freqs = []
		# read in the vocabulary 
		with open( os.path.join( self.assetpath, vocab_file ) ) as f:
			V = f.readlines()
		V = [ w.strip().strip( '\n' ).strip( '\'' ) for w in V if w != '' ]
		# read in the articles: each article is a line of comma-separated
		# pairs of word index and frequency
		with open( os.path.join( self.assetpath, corpus_file ) ) as f:
			freqs = f.readlines()
		freqs = [ d.strip().strip( '\n' ).strip( '\'' ) for d in freqs if d != '' ]
		C = np.zeros( ( len( V ), len( freqs ) ) )
		num_words, num_docs = C.shape
		# utility to split each index:count pair from the frequencies file
		pair = lambda x: tuple( [ int( p ) for p in x.split( ':' ) ] )
		# build the corpus as a num_words x num_docs matrix of word counts
		for j in range( num_docs ):
			pairs = [ pair( x ) for x in freqs[j].split( ',' ) ]
			for ( w, count ) in pairs:
				C[w-1, j] = count 
		num_test_docs = 20
		# split into training set and test set
		train_set = C[:, :-num_test_docs]
		test_set  = C[:, -num_test_docs:]

		# fit the model
		k = 10
		alpha0 = 1.
		TLDA = TensorDecompositionLDA( k, alpha0 )
		alphas, Phi = TLDA.fit( train_set )
		# show the top words for each topic
		topics = TLDA.topic_words( 5 )
		# map indices back to words for readability
		for k, v in topics.items():
			logging.info ( f'{k}, {[ ( V[idx], p ) for ( idx, p ) in v ]}' )

		# get the topic distributions for the test docs
		doc_topics = TLDA.doc_topics( test_set )
		idx = 0
		for k, v in doc_topics.items():
			logging.info ( f'The most probable topic for this document is {np.argmax( v )}' )
			doc = test_set[:, idx]
			# show original test document words gathered from the frequency matrix
			logging.info ( f'Tokens from original article: {[ V[j] for j in range( len( doc ) ) if doc[j] > 0 ]}\n\n' )
			idx += 1

		doc_topics_as_matrix = np.array( [ dist for dist in doc_topics.values() ] )
		np.testing.assert_array_almost_equal( np.ones( num_test_docs ), np.sum( doc_topics_as_matrix, axis=1 ) )

	def testTLDA20ng( self ):
		news = fetch_20newsgroups( subset='all' )
		vectorizer = CountVectorizer( max_df=0.5, min_df=20, stop_words='english' )
		C = vectorizer.fit_transform( news[ 'data' ] ).toarray().T
		V = vectorizer.get_feature_names()
		k = 20
		alpha0 = 1.
		TLDA = TensorDecompositionLDA( k, alpha0 )
		alphas, Phi = TLDA.fit( C )
		fig, axs = plt.subplots( 7, 3, figsize=( 14, 24 ) )

		for n in range( k ):
			i, j = divmod( n, 3 )
			ax = axs[i, j]
			t = TLDA.topic_words( 100, n )
			freqs = { V[idx]:p*1000  for ( idx, p ) in t[ f'topic {n}' ] }

			with warnings.catch_warnings():
				# hide deprecation warnings
				warnings.simplefilter( 'ignore' )
				wc = WordCloud( background_color="white", width=800, height=500 )
				wc = wc.generate_from_frequencies( freqs )
				ax.set_title( 'Topic %d' % (n + 1) )
				ax.imshow( wc, interpolation='bilinear' )
				ax.axis( 'off' )

		axs[-1, -1].axis( 'off' )
		plt.show()
		
if __name__ == '__main__':
	unittest.main()
