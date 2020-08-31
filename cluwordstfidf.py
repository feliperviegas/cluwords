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

    def _cluwords_idf(self, data):
        # self.hyp_aux = []
        # for w in range(0, len(self.vocab_cluwords)):
        #     self.hyp_aux.append(np.asarray(self.similarity_matrix[w], dtype=np.float32))
        #
        # self.hyp_aux = np.asarray(self.hyp_aux, dtype=np.float32)
        # Smoothes terms that as neighbors of all CluWords - Fit
        if self.smooth_neighbors:
            self.tfIdfTransformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
            self.tfIdfTransformer.fit_transform(self.similarity_matrix)

        start = timeit.default_timer()
        print('Read data')
        tf = self.raw_tf(binary=True, dt=np.float32, data=data)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Dot tf and hyp_aux')
        ### WITH ERROR ####
        # out = np.empty((tf.shape[0], self.hyp_aux.shape[1]), dtype=np.float32)
        ######## CORRECTION #######
        out = np.empty((tf.shape[0], np.transpose(self.similarity_matrix).shape[1]), dtype=np.float32)
        _dot = np.dot(tf, np.transpose(self.similarity_matrix), out=out)  # np.array n_documents x n_cluwords
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Divide hyp_aux by itself')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bin_hyp_aux = np.divide(self.similarity_matrix, self.similarity_matrix)
            # bin_hyp_aux[np.isneginf(bin_hyp_aux)] = 0
            bin_hyp_aux = np.nan_to_num(bin_hyp_aux)

        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        start = timeit.default_timer()
        print('Dot tf and bin hyp_aux')
        out = np.empty((tf.shape[0], np.transpose(bin_hyp_aux).shape[1]), dtype=np.float32)
        _dot_bin = np.dot(tf, np.transpose(bin_hyp_aux), out=out)
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

    def compute_norm_cond_mi(self, data, data_class, path_to_save='./', dataset_name='',
                             class_id=0, fold=0, log_file=True):
        if self.is_fit:
            try:
                self._compute_norm_cond_mutual_info(data=data, class_id=class_id, data_class=data_class)
            except Exception as e:
                print('Error: {}'.format(e))

            if log_file:
                print('Writing log file...')
                sparse.save_npz('{}/{}_cluwords_ncmi_{}_{}.npz'.format(path_to_save,
                                                                       dataset_name,
                                                                       class_id,
                                                                       fold), sparse.csr_matrix(self.hyp_mutual_info))
        else:
            raise NameError('Must fit model.')

        return

    @staticmethod
    def conditional_entropy(joint_proba_absence, joint_proba_presence, proba_presence):
        conditional_proba_presence = joint_proba_presence / proba_presence if proba_presence > .0 else .0
        conditional_proba_absence = joint_proba_absence / proba_presence if proba_presence > .0 else .0
        part_presence = - conditional_proba_presence * np.log2(conditional_proba_presence) \
            if conditional_proba_presence > .0 else .0
        part_absence = - conditional_proba_absence * np.log2(conditional_proba_absence) \
            if conditional_proba_absence > .0 else .0

        # if round(conditional_proba_presence, 4) > 1.:
        #     print('<conditional_proba_presence> P(x|y):{} P(x,y):{} P(y):{}'.format(conditional_proba_presence,
        #                                                                             joint_proba_presence,
        #                                                                             proba_presence))
        #
        # if round(conditional_proba_absence, 4) > 1.:
        #     print('<conditional_proba_abscence> P(x|y):{} P(x,y):{} P(y):{}'.format(conditional_proba_absence,
        #                                                                             joint_proba_absence,
        #                                                                             proba_presence))

        # print('<conditional_entropy> {} {} {}'.format(part_presence, part_absence, part_presence + part_absence))
        return part_presence + part_absence

    @staticmethod
    def probability(numerator, denominator):
        return (numerator / denominator) if denominator > .0 else .0

    @staticmethod
    def normalized_mutual_infomation(mutual_information, entropy_term, entropy_class):
        num = 2 * mutual_information
        den = entropy_term + entropy_class
        norm_cond_mutual_info = (num / den) \
            if den != .0 else .0
        return norm_cond_mutual_info

    def compute_probabilities(self, X, y, n_terms, n_classes, n_docs):
        # self.probability_class = np.empty((n_classes, 2), dtype=np.float32)
        # for class_id in range(0, n_classes):
        #     current_class = (y == class_id)
        #     bool_class = (current_class == 1)  # set class = 1
        #     # class = 1
        #     self.probability_class[class_id, 0] = self.probability(np.count_nonzero(bool_class == True), n_docs)
        #     bool_class = (current_class == 0)  # set class = 0
        #     # class = 0
        #     self.probability_class[class_id, 1] = self.probability(np.count_nonzero(bool_class == True), n_docs)

        self.probability_term = np.empty((n_terms, 2), dtype=np.float32)
        self.frequency_term = np.empty((n_terms, 2), dtype=np.float32)
        for term_id in range(0, n_terms):
            current_term = X[:, term_id]
            bool_term = (current_term == 1)  # set term = 1
            # term = 1
            self.probability_term[term_id, 0] = self.probability(np.count_nonzero(bool_term == True), n_docs)
            self.frequency_term[term_id, 0] = np.count_nonzero(bool_term == True)
            bool_term = (current_term == 0)  # set term = 0
            # term = 0
            self.probability_term[term_id, 1] = self.probability(np.count_nonzero(bool_term == True), n_docs)
            self.frequency_term[term_id, 1] = np.count_nonzero(bool_term == True)

    @staticmethod
    def log2(numerator, denominator):
        if numerator != 0 and denominator != 0:
            return np.log2(numerator / denominator)
        else:
            return 0

    def _conditional_mutual_information(self, n_11, n_10, n_01, n_00, n_1_, n__1, n_0_, n__0, n):
        return (n_11 / n) * self.log2((n * n_11), (n_1_ * n__1)) \
               + (n_01 / n) * self.log2((n * n_01), (n_0_ * n__1)) \
               + (n_10 / n) * self.log2((n * n_10), (n_1_ * n__0)) \
               + (n_00 / n) * self.log2((n * n_00), (n_0_ * n__0))

    def _gen_cond_mutual_info(self, X, y, class_id, conditional_term_id, term_id):
        current_class = (y == class_id)
        conditional_term = X[:, conditional_term_id]
        bool_conditional_term = (conditional_term == 1) * 1  # set conditional_term = 1
        confusion_matrix = np.zeros((2, 2))
        current_term = X[:, term_id]

        bool_term_1 = (current_term != 0) * 1  # set term = 1
        bool_term_0 = (current_term == 0) * 1  # set term = 0
        bool_class_1 = (current_class == 1) * 1  # set class = 1
        bool_class_0 = (current_class == 0) * 1  # set class = 0

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_1]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_1]), axis=0)
        confusion_matrix[0, 0] = np.count_nonzero(sum_all == 3)  # term = 1 | class = 1
        del sum_cond_term_and_class
        del sum_all

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_1]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_0]), axis=0)
        confusion_matrix[1, 0] = np.count_nonzero(sum_all == 3)  # term = 0 | class = 1
        del sum_cond_term_and_class
        del sum_all

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_0]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_0]), axis=0)
        confusion_matrix[1, 1] = np.count_nonzero(sum_all == 3)  # term = 0 | class = 0
        del sum_cond_term_and_class
        del sum_all

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_0]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_1]), axis=0)
        confusion_matrix[0, 1] = np.count_nonzero(sum_all == 3)  # term = 1 | class = 0
        del sum_cond_term_and_class
        del sum_all

        if np.sum(confusion_matrix):
            return self._conditional_mutual_information(n_11=confusion_matrix[0, 0],
                                                        n_10=confusion_matrix[0, 1],
                                                        n_01=confusion_matrix[1, 0],
                                                        n_00=confusion_matrix[1, 1],
                                                        n_1_=np.sum(confusion_matrix, axis=1)[0],
                                                        n__1=np.sum(confusion_matrix, axis=0)[0],
                                                        n_0_=np.sum(confusion_matrix, axis=1)[1],
                                                        n__0=np.sum(confusion_matrix, axis=0)[1],
                                                        n=np.sum(confusion_matrix))
        else:
            return 0.0

    def _compute_cond_mutual_info(self, data, data_class, class_id):
        tf = self.raw_tf(data=data)
        y, n_y = self.read_input(file=data_class)
        y = np.array(y, dtype=np.int)
        self.hyp_mutual_info = np.zeros((len(self.vocab_cluwords), len(self.vocab)), dtype=np.float32)
        for w in range(0, len(self.vocab_cluwords)):
            nonzero_indices = np.nonzero(self.similarity_matrix[w])
            for cond_term_id in nonzero_indices[0]:
                self.hyp_mutual_info[w][cond_term_id] = self._gen_cond_mutual_info(X=tf,
                                                                                   y=y,
                                                                                   class_id=class_id,
                                                                                   conditional_term_id=cond_term_id,
                                                                                   term_id=w)

        return

    def _norm_conditional_mutual_information(self, n_11, n_10, n_01, n_00, n_1_, n__1, n_0_, n__0, n,
                                             n_docs, probability_term):
        mutual_information = (n_11 / n) * self.log2((n * n_11), (n_1_ * n__1)) \
                             + (n_01 / n) * self.log2((n * n_01), (n_0_ * n__1)) \
                             + (n_10 / n) * self.log2((n * n_10), (n_1_ * n__0)) \
                             + (n_00 / n) * self.log2((n * n_00), (n_0_ * n__0))

        joint_prob_class_pres = self.probability(n__1, n_docs)
        joint_prob_class_abs = self.probability(n__0, n_docs)
        joint_prob_term_pres = self.probability(n_1_, n_docs)
        joint_prob_term_abs = self.probability(n_0_, n_docs)

        cond_entropy_class = self.conditional_entropy(joint_proba_absence=joint_prob_class_abs,
                                                      joint_proba_presence=joint_prob_class_pres,
                                                      proba_presence=probability_term[0])
        cond_entropy_term = self.conditional_entropy(joint_proba_absence=joint_prob_term_abs,
                                                     joint_proba_presence=joint_prob_term_pres,
                                                     proba_presence=probability_term[0])
        return self.normalized_mutual_infomation(mutual_information, cond_entropy_term, cond_entropy_class)

    def _gen_norm_cond_mutual_info(self, X, y, class_id, conditional_term_id, term_id):
        current_class = (y == class_id)
        conditional_term = X[:, conditional_term_id]
        bool_conditional_term = (conditional_term == 1) * 1  # set conditional_term = 1
        confusion_matrix = np.zeros((2, 2))
        current_term = X[:, term_id]

        bool_term_1 = (current_term != 0) * 1  # set term = 1
        bool_term_0 = (current_term == 0) * 1  # set term = 0
        bool_class_1 = (current_class == 1) * 1  # set class = 1
        bool_class_0 = (current_class == 0) * 1  # set class = 0

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_1]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_1]), axis=0)
        confusion_matrix[0, 0] = np.count_nonzero(sum_all == 3)  # term = 1 | class = 1
        del sum_cond_term_and_class
        del sum_all

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_1]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_0]), axis=0)
        confusion_matrix[1, 0] = np.count_nonzero(sum_all == 3)  # term = 0 | class = 1
        del sum_cond_term_and_class
        del sum_all

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_0]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_0]), axis=0)
        confusion_matrix[1, 1] = np.count_nonzero(sum_all == 3)  # term = 0 | class = 0
        del sum_cond_term_and_class
        del sum_all

        sum_cond_term_and_class = np.sum(np.asarray([bool_conditional_term, bool_class_0]), axis=0)
        sum_all = np.sum(np.asarray([sum_cond_term_and_class, bool_term_1]), axis=0)
        confusion_matrix[0, 1] = np.count_nonzero(sum_all == 3)  # term = 1 | class = 0
        del sum_cond_term_and_class
        del sum_all

        if np.sum(confusion_matrix):
            return self._norm_conditional_mutual_information(n_11=confusion_matrix[0, 0],
                                                             n_10=confusion_matrix[0, 1],
                                                             n_01=confusion_matrix[1, 0],
                                                             n_00=confusion_matrix[1, 1],
                                                             n_1_=np.sum(confusion_matrix, axis=1)[0],
                                                             n__1=np.sum(confusion_matrix, axis=0)[0],
                                                             n_0_=np.sum(confusion_matrix, axis=1)[1],
                                                             n__0=np.sum(confusion_matrix, axis=0)[1],
                                                             n=np.sum(confusion_matrix),
                                                             n_docs=X.shape[0],
                                                             probability_term=self.probability_term[conditional_term_id]
                                                             )
        else:
            return 0.0

    def _compute_norm_cond_mutual_info(self, data, data_class, class_id):
        tf = self.raw_tf(data=data)
        y, n_y = self.read_input(file=data_class)
        y = np.array(y, dtype=np.int)
        self.compute_probabilities(X=tf,
                                   y=y,
                                   n_terms=tf.shape[1],
                                   n_classes=len(np.unique(y)),
                                   n_docs=tf.shape[0])

        # self.hyp_mutual_info = np.zeros((len(self.vocab_cluwords), len(self.vocab)), dtype=np.float32)
        # for w in range(0, len(self.vocab_cluwords)):
        #     nonzero_indices = np.nonzero(self.similarity_matrix[w])
        #     for cond_term_id in nonzero_indices[0]:
        #         self.hyp_mutual_info[w][cond_term_id] = self._gen_norm_cond_mutual_info(X=tf,
        #                                                                                 y=y,
        #                                                                                 class_id=class_id,
        #                                                                                 conditional_term_id=cond_term_id,
        #                                                                                 term_id=w)

        hyp_mutual_info = IncrementalCOOMatrix(shape=(len(self.vocab_cluwords), len(self.vocab)), dtype=np.float32)
        for w in range(0, len(self.vocab_cluwords)):
            nonzero_indices = np.nonzero(self.similarity_matrix[w])
            for cond_term_id in nonzero_indices[0]:
                hyp_mutual_info.append(w,
                                       cond_term_id,
                                       self._gen_norm_cond_mutual_info(X=tf,
                                                                       y=y,
                                                                       class_id=class_id,
                                                                       conditional_term_id=cond_term_id,
                                                                       term_id=w))

        self.hyp_mutual_info = hyp_mutual_info.tocoo()
        return
