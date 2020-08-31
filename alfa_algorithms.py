import codecs
import timeit
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from incremental_coo_matrix import IncrementalCOOMatrix


class AlfaKnn:
    def __init__(self, threshold, n_threads):
        self.threshold = threshold
        self.n_threads = n_threads
        print('N Threads: {}'.format(self.n_threads))

    def _mem_usage(self, pandas_obj):
        if isinstance(pandas_obj, pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else:  # we assume if not a df it's a series
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
        return "{:03.2f} MB".format(usage_mb)

    def create_cosine_cluwords(self, input_vector_file, n_words, k_neighbors, dataset):
        df, labels_array = self.build_word_vector_matrix(input_vector_file, n_words)
        print('NearestNeighbors K={}'.format(k_neighbors))
        start = timeit.default_timer()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='cosine', n_jobs=self.n_threads).fit(
            df)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        print('NN Distaces')
        start = timeit.default_timer()
        distances, indices = nbrs.kneighbors(df)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        print('Saving cluwords')

        self._save_cluwords(labels_array, n_words, k_neighbors, distances, indices, dataset)

        return

    def _save_cluwords(self, labels_array, n_words, k_neighbors, distances, indices, dataset):
        """
        Description
        -----------
        Save the cluwords of each word to csv using pandas. Dataframe.
        
        """
        # list_cluwords = np.zeros((n_words, n_words), dtype=np.float16)
        # list_cluwords_bin = sparse.csr_matrix((n_words, n_words), dtype=np.float16)

        list_cluwords = IncrementalCOOMatrix(shape=(n_words, n_words), dtype=np.float32)
        list_cluwords_bin = IncrementalCOOMatrix(shape=(n_words, n_words), dtype=np.float32)

        # Check if cosine limit was set
        if self.threshold:
            for p in range(0, n_words):
                for i, k in enumerate(indices[p]):
                    # .875, .75, .625, .50

                    if 1 - distances[p][i] >= self.threshold:
                        # list_cluwords[p][k] = round(1 - distances[p][i], 2)
                        list_cluwords.append(p, k, 1. - round(distances[p][i], 2))
                        list_cluwords_bin.append(p, k, round(1))
                    # else:
                    #     list_cluwords[p][k] = 0.0
        else:
            for p in range(0, n_words):
                for i, k in enumerate(indices[p]):
                    # list_cluwords[p][k] = round(1 - distances[p][i], 2)
                    list_cluwords.append(p, k, round(1 - distances[p][i], 2))
                    list_cluwords_bin.append(p, k, round(1))

        np.savez_compressed('cluwords_{}.npz'.format(dataset),
                            index=np.asarray(labels_array),
                            cluwords=np.asarray(labels_array),
                            k_neighbors=k_neighbors,
                            threshold=self.threshold)

        sparse.save_npz('cluwords_sim_{}.npz'.format(dataset), list_cluwords.tocoo())
        sparse.save_npz('cluwords_sim_bin_{}.npz'.format(dataset), list_cluwords_bin.tocoo())

    @staticmethod
    def build_word_vector_matrix(vector_file, n_words):
        """Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays"""
        numpy_arrays = []
        labels_array = []

        with codecs.open(vector_file, 'r', 'utf-8') as f:
            _ = next(f)  # Skip the first line

            for c, r in enumerate(f):
                sr = r.split()
                labels_array.append(sr[0])
                numpy_arrays.append(np.array([float(i) for i in sr[1:]]))

                if c == n_words:
                    return np.array(numpy_arrays), labels_array

        return np.array(numpy_arrays), labels_array
