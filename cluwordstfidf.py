import timeit
import warnings
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from alfa_algorithms import AlfaKnn
from incremental_coo_matrix import IncrementalCOOMatrix


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

    def __init__(self, dataset, algorithm, embedding_file_path, n_words, k_neighbors, threshold=.85, n_jobs=1,
                 verbose=0):
        if verbose:
            print('K: {}'.format(k_neighbors))
            print('Cossine: {}'.format(threshold))

        if algorithm == 'knn_cosine':
            print('kNN...')
            knn = AlfaKnn(threshold=threshold,
                          n_threads=n_jobs)
            knn.create_cosine_cluwords(input_vector_file=embedding_file_path,
                                       n_words=n_words,
                                       k_neighbors=k_neighbors,
                                       dataset=dataset)
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
    n_words: int
        Number of words passed as paramter.
    cluwords_tf_idf: ndarray
        Product between term frequency and inverse term frequency.
    cluwords_idf:

    """

    def __init__(self, n_words, npz_path, npz_sim_path, npz_sim_bin_path, n_jobs=1, smooth_neighbors=False,
                 sublinear_tf=False):
        self.n_words = n_words
        self.n_jobs = n_jobs
        self.smooth_neighbors = smooth_neighbors
        self.sublinear_tf = sublinear_tf
        self.cluwords_tf_idf = None
        self.cluwords_idf = None
        self.is_fit = 0
        self.probability_term = None
        self.frequency_term = None
        self.probability_class = None
        self.documents = None
        self.n_documents = None
        self.hyp_mutual_info = None
        try:
            loaded = np.load(npz_path)
            self.vocab = loaded['index']
            self.vocab_cluwords = loaded['cluwords']
            # self.similarity_matrix = loaded['data']
            # del loaded['data']
            self.similarity_matrix = sparse.csr_matrix(sparse.load_npz(npz_sim_path), dtype=np.float32)
            self.similarity_matrix_bin = sparse.csr_matrix(sparse.load_npz(npz_sim_bin_path), dtype=np.float32)
            del loaded
            print('Matrix{}'.format(self.similarity_matrix.shape))
        except IOError:
            print("Error opening file .npz")
            exit(0)

    def read_input(self, file):
        arq = open(file, 'r')
        doc = arq.readlines()
        arq.close()
        documents = list(map(str.rstrip, doc))
        n_documents = len(documents)
        return documents, n_documents

    def raw_tf(self, data, binary=False, dt=np.float32):
        tf_vectorizer = CountVectorizer(max_features=self.n_words, binary=binary, vocabulary=self.vocab)
        documents, n_documents = self.read_input(data)
        tf_vectorizer.fit(self.documents)
        tf = tf_vectorizer.transform(documents)
        return np.asarray(tf.toarray(), dtype=dt)

    @staticmethod
    def normalize(data, normalization_function='l2', has_test=False, train_size=0):
        n_rows = data.shape[0]
        normalize(data, norm=normalization_function,
                  axis=1, copy=False, return_norm=False)
        if has_test:
            train = np.take(data, np.arange(train_size), axis=0)
            test_size = n_rows - train_size
            test = np.take(data, (np.arange(test_size) + train_size), axis=0)
            return train, test
        else:
            return data

    def _cluwords_idf_sparse(self, data):
        # Smoothes terms that as neighbors of all CluWords - Fit
        if self.smooth_neighbors:
            self.tfIdfTransformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
            self.tfIdfTransformer.fit_transform(self.similarity_matrix)

        start = timeit.default_timer()
        print('Read data')
        tf = self.raw_tf(binary=True, dt=np.float32, data=data)
        tf = sparse.csr_matrix(tf.copy())
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Dot tf and hyp_aux')
        ### WITH ERROR ####
        # out = np.empty((tf.shape[0], self.hyp_aux.shape[1]), dtype=np.float32)
        ######## CORRECTION #######

        _dot = tf.dot(sparse.csr_matrix.transpose(self.similarity_matrix))  # np.array n_documents x n_cluwords
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # start = timeit.default_timer()
        # print('Divide hyp_aux by itself')
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     bin_hyp_aux = np.divide(self.similarity_matrix, self.similarity_matrix)
        #     # bin_hyp_aux[np.isneginf(bin_hyp_aux)] = 0
        #     bin_hyp_aux = np.nan_to_num(bin_hyp_aux)
        #
        # end = timeit.default_timer()
        # print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Dot tf and bin hyp_aux')
        _dot_bin = tf.dot(sparse.csr_matrix.transpose(self.similarity_matrix_bin))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Divide _dot and _dot_bin')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_hyp = np.nan_to_num(np.divide(_dot, _dot_bin))

        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Sum')
        self.cluwords_idf = np.sum(mu_hyp, axis=0)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('log')
        self.cluwords_idf = np.log10(np.divide(self.n_documents, self.cluwords_idf))
        print('IDF shape {}'.format(self.cluwords_idf.shape))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))

    def fit(self, data, path_to_save='./', dataset_name='', fold=0, log_file=True):
        print('Reading data...')
        self.documents, self.n_documents = self.read_input(data)
        print('Computing IDF...')
        self._cluwords_idf_sparse(data=data)

        if log_file:
            print('Writing log files...')
            np.savez_compressed('{}/{}_cluwords_information_{}.npz'.format(path_to_save,
                                                                           dataset_name,
                                                                           fold),
                                cluwords_vocab=self.vocab_cluwords,
                                words_vocab=self.vocab)
            sparse.save_npz('{}/{}_cluwords_cosine_{}.npz'.format(path_to_save,
                                                                  dataset_name,
                                                                  fold), sparse.csr_matrix(self.similarity_matrix))

        self.is_fit = True
        return

    def transform(self, data, compacted=False, idf=True):
        if self.is_fit:
            print('Computing TF-IDF...')
            tf_idf = self._tf(data=data)
            if idf:
                tf_idf = np.multiply(tf_idf, self.cluwords_idf)

            if compacted:
                tf = self.raw_tf(binary=True, dt=np.float32, data=data)
                tf_idf = np.multiply(tf_idf, tf)

            return tf_idf
        else:
            print("Error! No test set found!")
            exit(1)

    def _tf(self, data):
        start = timeit.default_timer()
        tf = self.raw_tf(data=data)
        warnings.simplefilter("error", RuntimeWarning)
        if self.sublinear_tf:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tf = 1. + np.log10(tf)
                tf[np.isneginf(tf)] = 0

        # cluwords_tf = np.dot(tf, np.transpose(self.hyp_sim))
        cluwords_tf = sparse.csr_matrix(tf).dot(sparse.csr_matrix.transpose(self.similarity_matrix)).toarray()
        end = timeit.default_timer()
        print("Cluwords TF done in %0.3fs." % (end - start))
        print(f'TF {cluwords_tf.shape}')
        return cluwords_tf
