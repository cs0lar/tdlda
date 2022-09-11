import os

from tqdm import tqdm

import numpy as np

from .moments import Moments
from .decomposition import Decomposition
from config import Config

logging = Config.logging


def phinorm( Phi ):
    sp = np.sum( np.maximum( Phi, 0 ), 0 )
    sn = np.sum( np.maximum( -Phi, 0 ), 0 )

    B = np.maximum( Phi.dot( np.diag( 2.*( sp>sn ).astype( float )-1. )), 0 )
    B /= np.sum( B, 0 )
    
    return B


class TensorDecompositionLDA( object ):
    """
        It implements the Scalable Tensor Orthogonal Decomposition (STOD)
        from:
        
            “Scalable Moment-Based Inference for Latent Dirichlet Allocation” 
            C. Wang, X. Liu, Y. Song, and J. Han, 2014

        and
            https://github.com/UIUC-data-mining/STOD

        Parameters:
        -----------

        k - the number of topics to infer

        alpha0 - the sum of "pseudo -counts", i.e. alpha0 = alpha_1 + ... + alpha_k
                 with each alpha_j being a parameter of the Dirichlet distribution
                 for topics.

         - the number of inner iterations to perform during the power method process

        inner_iters - the number of inner iterations to perform during the power method process 
        outer_iters - the number of outer iterations to perform during the power method process
    """
    def __init__( self, k, alpha0, inner_iters=30, outer_iters=30 ):
        self._k      = k
        self._alpha0 = alpha0
        self._inner  = inner_iters
        self._outer  = outer_iters

    @staticmethod
    def sample( eta, alpha, Beta, k, D ):
        """
            It generates synthetic data using the 
            LDA generative process from Blei et. al. 2003

            Parameters:
            -----------
            eta   - parameter for the Poisson distribution modelling document lengths
            alpha - a k-length vector of Dirichlet parameters for topic distribution
            Beta  - a V x k matrix of Dirichlet parameters for the conditional word probabilities
            k     - the number of topics
            D     - the number of documents to generate
            
            Returns:
            --------
            C - a V x D matrix of words frequencies
        """

        V, _ = Beta.shape
        Phi = np.zeros( shape=Beta.shape )
        
        for i in range( k ):
            # Phi[:, i] holds the ground-truth conditional  
            # probabilities p(w |zi,β) for each word w \in V
            Phi[:, i] = np.random.dirichlet( Beta[:, i] )

        C = np.zeros( shape=( V, D ) )
        logging.info( f'Generating {D} documents using LDA process...' )
        
        for c in tqdm( range( D ) ):
            # Choose length N of next document from Poisson(ξ).
            N = np.random.poisson( eta )
            # Choose θ ∼ Dir(α).
            Theta = np.random.dirichlet( alpha )
            # For each of the N words wn:
            # (a) Choose a topic zn ∼ Multinomial(θ).
            Z = np.random.multinomial( N, Theta ) # a k-length array containing the frequency at which each topic was drawn after N experiments
            W = np.zeros( V )

            for i in range( k ):
                # (b) Choose a word wn from p(wn |zn,β), a multinomial probability conditioned on the topic zn.
                # Z[i] words were drawn belonging to topic i, so we extract the ith dirichlet parameter vector
                # and draw the distribution to use for drawing those words
                phi = Phi[:, i] 
                # now extract Z[i] words from phi
                W += np.random.multinomial( Z[i], phi ) # size=(d, )
            
            # add this document to the corpus
            assert np.sum( W ) == N
            C[:, c] = W

        return C, Phi

    def fit( self, C ):
        """
            It estimates matrix Phi \in R^{V x k} where V is the size
            of the vocabulary, k is the number of topics and each 
            column of Phi represents a topic as a distribution of words
            in the vocabulary.
            It implements Algorithm 2 in Wang et.al. 2014

            Parameters:
            -----------

            C - the corpus as V x D sparse matrix where D is the number of documents
                in the corpus, V is the number of words in the vocabulary
                and a cell i,j in C is the frequency of word i in document j

            Returns:
            --------
            alphas - a vector of Dirichlet parameter such that sum(alphas) = alpha0
            Phi   - a V x k matrix of word distributions for each topic
        
        """
        # V is the number of words in the vocabulary and D is the number 
        # of documents in the corpus C
        V, D = C.shape 
        alpha0 = self._alpha0
        k = self._k
        # 2.1 First scan of data: compute M1 and E2
        M1 = Moments.first( C )
        E2 = Moments.second( C )
        # 2.2 Find k largest orthonormal eigenpairs of E2
        s, U = Decomposition.e2( E2, k ) # size(s) = (k, ), size(U) = (V, k)
        # 2.3 Compute M1' = UM1
        M1_ = U.T.dot( M1 ) # size(M1_) = (k, )
        # 2.4 Compute spectral decomposition for M2'
        M2_ = ( alpha0 + 1. )*np.diag( s ) - alpha0 * ( M1_ * M1_[:, np.newaxis] ) # size(M2_)=(k, k)
        s, U_ = Decomposition.m2_( M2_ )
        # if the eigenvalues are negative, we have a problem
        assert ( s > 0 ).all(), 'Found negative eigenvalues!'
        if np.iscomplexobj( s ) and ( s.imag == 0 ).all():
            s = s.real
        # 2.5 Compute the whitening matrix W and its pseudo inverse W_
        M = U.dot( U_ ) # size(M) = (V, k)
        W = M.dot( np.diag( 1./np.sqrt( s ) ) )
        W_ = M.dot( np.diag( np.sqrt( s ) ) )
        # 2.6 Second scan of the data: Compute T_ according to eq. (16)
        M3 = Moments.third( C, W, M1, alpha0 )
        # 2.7 Perform power method line  1.6 to 1.16 in Algorithm 1
        lambdas = np.zeros( k )
        thetas  = np.zeros( ( k, k ) )

        logging.info( f'Running tensor decomposition power method for k={k}' )
        
        for i in tqdm( range( k ) ):
            val, vec, deflated = Decomposition.powermethod( M3, self._inner, self._outer )
            M3 = deflated
            lambdas[i] = ( 1./val )**2
            thetas[:, i] = vec

        # retrieve the LDA parameters
        alphas = alpha0*lambdas
        Phi = W_.dot( thetas )
        # normalise alphas and Phis
        self.Phi = phinorm( Phi )
        self.alphas = alpha0 * ( alphas / np.sum( alphas ) )

        return self.alphas, self.Phi

    def topic_words( self, n, z=None ):
        """
            It returns the n most probable words from the Vocabulary
            for each of the k topics inferred by the LDA

            Parameters:
            -----------

                n - the number of most probable words to return

                z - optional, default to None. An integer that specifies
                    the single topic to return word probabilites for

            Returns:
            --------

                T - A dictionary where each key represents a topic
                    and each value is a list of n pairs. The first
                    element of a pair is the index of a word in the
                    vocabulary and the second element is the value
                    p(w|z) of the probability of that word given the 
                    key topic. 

        """

        assert self.Phi is not None, 'The topic distributions are not defined.'
        V, k = self.Phi.shape
        # sort the indices descending and take the top n results
        idxs = np.flip( np.argsort( self.Phi, axis=0 ), axis=0 )[:n, :]
        vals = np.take_along_axis( self.Phi, idxs, axis=0 )

        if z:
            assert z < k, 'Invalid topic number.'

            return { f'topic {z}': list( zip( idxs[:, z], np.around( vals[:, z], 5 ) ) ) }

        return { f'topic {t}':list( zip( idxs[:, t], np.around( vals[:, t], 5 ) ) ) for t in range( k ) }


    def doc_topics( self, C ):
        """
            It returns the topic distribution for each document in C

            Parameters:
            -----------
                C - a V x D matrix where cell C(i, j) is the number
                    of occurrences of the ith word in the vocabulary 
                    in the jth document.

            Returns:
            --------
                D - A dictionary where each key represents a document
                    and each value is list where the ith element is the
                    mass of the ith topic on the document.
        """
        assert self.Phi is not None, 'The topic distributions are not defined.'
        V, D = C.shape
        # turn counts into probabilities
        p_word_given_doc = C / np.sum( C, 0 )
        # compute the join probabilities p(topic, doc)
        p_topic_doc = self.Phi.T.dot( p_word_given_doc )
        # normalise by the marginal document probabilities p(doc)
        p_topic_give_doc = p_topic_doc / np.sum( p_topic_doc, 0 )

        return { f'document {d}': p_topic_give_doc[:,d] for d in range( D ) }