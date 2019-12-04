import timeit
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from alfa_algorithms import AlfaKnn


class Cluwords:
    """
    Description
    -----------
    Create the cluwords DataFrame from the pre-treined embedding model (e.g., GloVe, Wiki News - FastText).

    Parameters
    ----------
    algorithm: str
        The algorithm to use as cluwords distance limitation (alfa).
        'knn' : use NearestNeighbors.
        'k-means' : use K-Means.
        'dbscan' : use DBSCAN.
    embedding_file_path: str
        The path to embedding pre-treined model.
    n_words: int
        Number of words in the dataset.
    k_neighbors: boolean
        Number of neighbors desire for each cluword.
    cosine_lim: float, (default = .85)
        The cosine limit to consider the value of cosine siliarity between two words in the model.

        Note: if two words have the cosine similiarity under cosine limit, the value of cosine similiarty
            is equal zero.
    n_jobs: int, (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only :meth:`kneighbors` and :meth:`kneighbors_graph` methods.
    verbose: int, (default = 0)
        Enable verbose output.

    Attributes
    ----------

    """

    def __init__(self, algorithm, embedding_file_path, n_words, k_neighbors, threshold=.85, n_jobs=1, verbose=0):
        if verbose:
            print('K: {}'.format(k_neighbors))
            print('Cossine: {}'.format(threshold))

        if algorithm == 'knn_cosine':
            print('kNN...')
            knn = AlfaKnn(threshold=threshold,
                          n_threads=n_jobs)
            knn.create_cosine_cluwords(input_vector_file=embedding_file_path,
                                       n_words=n_words,
                                       k_neighbors=k_neighbors)
        elif algorithm == 'knn_mahalanobis':
            print('kNN Mahalanobis...')
            knn = AlfaKnn(threshold=threshold,
                          n_threads=n_jobs)
            knn.create_mahalanobis_cluwords(input_vector_file=embedding_file_path,
                                            n_words=n_words,
                                            k_neighbors=k_neighbors)
        # elif algorithm == 'k-means':
        #     pass
        # elif algorithm == 'dbscan':
        #     pass
        # elif algorithm == 'w2vsim':
        #     w2vsim = W2VSim(file_path_cluwords=path_to_save_cluwords,
        #                     save=False)
        #     self.df_cluwords = w2vsim._create_cluwords(input_vector_file=embedding_file_path,
        #                                                n_words=n_words,
        #                                                n_words_sim=k_neighbors)
        else:
            print('Invalid method')
            exit(0)


class CluwordsTFIDF:
    """
    Description
    -----------
    Calculates Terme Frequency-Inverse Document Frequency (TFIDF) for cluwords.

    Parameters
    ----------
    dataset_file_path : str
        The complete dataset file path.
    n_words : int
        Number of words in the dataset.
    path_to_save_cluwords : list, default None
        Path to save the cluwords file.
    class_file_path: str, (default = None)
        The path to the file with the class of the dataset.

    Attributes
    ----------
    dataset_file_path: str
        The dataset file path passed as parameter.
    path_to_save_cluwords_tfidf: str
        The path to save cluwords passed as parameter, with the addition of the file name.
    n_words: int
        Number of words passed as paramter.
    cluwords_tf_idf: ndarray
        Product between term frequency and inverse term frequency.
    cluwords_idf:

    """

    def __init__(self, dataset_file_path, n_words, path_to_save_cluwords, class_file_path=None,
                 has_class=False, cossine_filter=1.0):
        self.dataset_file_path = dataset_file_path
        self.path_to_save_cluwords_tfidf = path_to_save_cluwords + '/cluwords_features.libsvm'
        self.n_words = n_words
        self.cluwords_tf_idf = None
        self.cluwords_idf = None
        self.cossine_filter = cossine_filter
        loaded = np.load('cluwords.npz')
        self.vocab = loaded['index']
        self.vocab_cluwords = loaded['cluwords']
        self.cluwords_data = loaded['data']
        self.has_class = has_class

        if self.has_class:
            self.Y = []
            with open(class_file_path, 'r') as input_file:
                for _class in input_file:
                    self.Y.append(np.int(_class))
                input_file.close()
                self.Y = np.asarray(self.Y)

        print('Matrix{}'.format(self.cluwords_data.shape))
        del loaded
        print('\nCosine Filter: {}'.format(cossine_filter))

        self._read_input()

    def _read_input(self):
        arq = open(self.dataset_file_path, 'r')
        doc = arq.readlines()
        arq.close()

        self.documents = list(map(str.rstrip, doc))
        self.n_documents = len(self.documents)

    def fit_transform(self):
        """Compute cluwords tfidf."""

        # Set number of cluwords
        self.n_cluwords = self.n_words

        """
        # Redundant Cluwords to remove #######################################
        print('Search for redundant cluwords...')
        m_cluwords = []
        for w_1 in range(len(self.vocab)):
            hw_w_1 = list(self.cluwords_data[w_1])
            m_cluwords.append(hw_w_1)

        m_cluwords = np.asarray(a=m_cluwords,
                                  dtype=np.float32)

        print('Fitting Nearest Neighbors...')
        start = timeit.default_timer()
        nbrs = NearestNeighbors(n_neighbors=len(self.vocab),
                                algorithm='auto',
                                metric='cosine',
                                n_jobs=1).fit(m_cluwords)
        end = timeit.default_timer()
        print('Time {}\n'.format(end - start))

        print('Nearest Neighbors...')
        start = timeit.default_timer()
        distance, hw_sim = nbrs.kneighbors(m_cluwords)
        to_remove = []
        for _hw in range(len(distance)):
            if _hw not in to_remove:
                similarity = (1. - distance[_hw]) >= self.cossine_filter
                to_remove += [self.vocab[hw_sim[_hw][i]] for i in range(len(similarity)) if
                              similarity[i] and hw_sim[_hw][i] != _hw]

        # Get arg of list of words -> to_remove
        to_remove_arg = []
        for w in to_remove:
            to_remove_arg.append(int(np.where(self.cluwords == w)[0]))
        to_remove_arg = np.sort(np.array(to_remove_arg, dtype=np.uint32))
        # print(to_remove_arg)

        print('Number of redundant cluwords: {}'.format(len(to_remove)))
        if to_remove:
            print('Removing redundant cluwords...')
            # Remove row of matrix
            self.cluwords_data = np.delete(self.cluwords_data, to_remove_arg, axis=0)
            # Remove redundant cluwords
            self.cluwords = np.delete(self.cluwords, to_remove_arg)
        end = timeit.default_timer()
        print('Time {}\n'.format(end - start))
        """
        ########################################################################

        # Set vocabulary of cluwords
        self.n_cluwords = len(self.vocab_cluwords)
        print('Number of cluwords {}'.format(len(self.vocab_cluwords)))
        print('Matrix{}'.format(self.cluwords_data.shape))

        print('\nComputing TF...')
        self._cluwords_tf()
        # print('\nComputing IDF...')
        # self._cluwords_idf()

        print(self.cluwords_tf_idf.shape)
        # print (self.cluwords_idf.shape)
        # self.cluwords_tf_idf = np.multiply(self.cluwords_tf_idf, np.transpose(self.cluwords_idf))
        # self._save_tf_idf_features_libsvm()
        return self.cluwords_tf_idf

    def _raw_tf(self, binary=False, dtype=np.float32):
        tf_vectorizer = CountVectorizer(max_features=self.n_words, binary=binary, vocabulary=self.vocab)
        tf = tf_vectorizer.fit_transform(self.documents)
        return tf

    def _cluwords_tf(self):
        start = timeit.default_timer()
        tf = self._raw_tf()

        print('tf shape {}'.format(tf.shape))

        # self.cluwords_tf_idf = np.zeros((self.n_documents, self.n_cluwords), dtype=np.float16)
        # print('{}'.format())

        self.hyp_aux = []
        for w in range(0, len(self.vocab_cluwords)):
            self.hyp_aux.append(np.asarray(self.cluwords_data[w], dtype=np.float16))

        self.hyp_aux = np.asarray(self.hyp_aux, dtype=np.float32)
        self.hyp_aux = csr_matrix(self.hyp_aux, shape=self.hyp_aux.shape, dtype=np.float32)  # test sparse matrix!

        self.cluwords_tf_idf = np.dot(tf, np.transpose(self.hyp_aux))
        self.cluwords_tf_idf = tf.dot(self.hyp_aux.transpose())

        end = timeit.default_timer()
        print("Cluwords TF done in %0.3fs." % (end - start))

    def _cluwords_idf(self):
        start = timeit.default_timer()
        print('Read data')
        tf = self._raw_tf(binary=True, dtype=np.float32)
        import pdb
        pdb.set_trace()
        self.hyp_aux = self.hyp_aux.todense()
        # tf = csr_matrix(tf, shape=(tf.shape[0], self.n_words), dtype=np.float32)  # test sparse matrix!

        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Bin Doc')
        # print(tf)

        start = timeit.default_timer()
        print('Dot tf and hyp_aux')
        _dot = np.dot(tf, np.transpose(self.hyp_aux))  # np.array n_documents x n_cluwords # Correct!
        # pdb.set_trace()
        # _dot = tf.dot(self.hyp_aux.transpose())  # Test sparse matrix!
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Dot matrix:')
        # print(_dot)

        start = timeit.default_timer()
        print('Divide hyp_aux by itself')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pdb.set_trace()
            # self.hyp_aux = self.hyp_aux.todense()
            # pdb.set_trace()
            bin_hyp_aux = np.nan_to_num(np.divide(self.hyp_aux, self.hyp_aux))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Bin cluwords')
        # print(bin_hyp_aux)

        start = timeit.default_timer()
        print('Dot tf and bin hyp_aux')
        # out = np.empty((tf.shape[0], np.transpose(bin_hyp_aux).shape[1]), dtype=np.float32)
        _dot_bin = np.dot(tf, np.transpose(bin_hyp_aux))
        # pdb.set_trace()
        # bin_hyp_aux = csr_matrix(bin_hyp_aux, shape=bin_hyp_aux.shape)
        # pdb.set_trace()
        # _dot_bin = tf.dot(bin_hyp_aux)

        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Count Dot')
        # print(_dot_bin)

        # pdb.set_trace()
        # _dot = _dot.todense()
        # pdb.set_trace()
        # _dot_bin = _dot_bin.todense()
        # pdb.set_trace()

        start = timeit.default_timer()
        print('Divide _dot and _dot_bin')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_hyp = np.nan_to_num(np.divide(_dot, _dot_bin))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Div dot by bin cluwords')
        # print(mu_hyp)

        ##TODO
        # \mu _{c,d} = \frac{1}{\left | \mathcal{V}_{d,c} \right |} \cdot  \sum_{t \in \mathcal{V}_{d,c}} w_t
        #
        ##

        start = timeit.default_timer()
        print('Sum')
        self.cluwords_idf = np.sum(mu_hyp, axis=0)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))

        # print('Mu')
        # print(self.cluwords_idf)

        start = timeit.default_timer()
        print('log')
        self.cluwords_idf = np.log10(np.divide(self.n_documents, self.cluwords_idf))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('IDF:')
        # print(self.cluwords_idf)

    def _save_tf_idf_features_libsvm(self):
        tf = self._raw_tf(binary=True, dtype=np.float32)
        with open('{}'.format(self.path_to_save_cluwords_tfidf), 'w') as file:
            for x in range(self.cluwords_tf_idf.shape[0]):
                if self.has_class:
                    file.write('{} '.format(self.Y[x]))
                for y in range(1, self.cluwords_tf_idf.shape[1]):
                    if tf[x][y]:
                        file.write('{}:{} '.format(y + 1, self.cluwords_tf_idf[x][y]))
                file.write('\n')
            file.close()
