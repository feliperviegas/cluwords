from __future__ import division

import codecs
from numbers import Number

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
import timeit
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

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

    def create_cosine_cluwords(self, input_vector_file, n_words, k_neighbors):
        input_vector_file = input_vector_file
        df, labels_array = build_word_vector_matrix(input_vector_file, n_words)
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

        self._save_cluwords(labels_array, n_words, k_neighbors, distances, indices)

        return

    def plot_dist(self, df, invcovmx):
        dist = []
        for x in range(0, df.shape[0]):
            for y in range(0, df.shape[0]):
                dist.append(mahalanobis(df[x], df[y], invcovmx))

        # Plot Distribution
        sns.set_style('white')

        plt.figure(num=None, figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k')

        sns.distplot(dist, bins=None)

        plt.title('Mahalanobis Distance Histogram for FastText Crawl')
        plt.xlabel('', fontsize=16)
        plt.ylabel('', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.show()
        plt.tight_layout()
        plt.savefig('dist_mahalanobis_histogram_crawl.png')

    def create_mahalanobis_cluwords(self, input_vector_file, n_words, k_neighbors):
        input_vector_file = input_vector_file
        df, labels_array = build_word_vector_matrix(input_vector_file, n_words)
        print('NearestNeighbors K={}'.format(k_neighbors))
        start = timeit.default_timer()

        covmx = np.cov(np.transpose(df))
        invcovmx = sp.linalg.inv(covmx)
        self.plot_dist(df, invcovmx)

        nbrs = NearestNeighbors(n_neighbors=k_neighbors,
                                algorithm='brute',
                                metric='mahalanobis',
                                metric_params={'V': invcovmx},
                                n_jobs=self.n_threads).fit(df)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        print('NN Distaces')
        start = timeit.default_timer()
        distances, indices = nbrs.kneighbors(df)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        print('Saving cluwords')
        self._save_cluwords(labels_array, n_words, k_neighbors, distances, indices)

        return

    def _save_cluwords(self, labels_array, n_words, k_neighbors, distances, indices):
        """
        Description
        -----------
        Save the cluwords of each word to csv using pandas. Dataframe.
        
        """
        list_cluwords = np.zeros((n_words, n_words), dtype=np.float16)

        # Check if cosine limit was set
        if self.threshold:
            for p in range(0, n_words):
                for i, k in enumerate(indices[p]):
                    # .875, .75, .625, .50
                    if 1 - distances[p][i] >= self.threshold:
                        list_cluwords[p][k] = round(1 - distances[p][i], 2)
                    else:
                        list_cluwords[p][k] = 0.0
        else:
            for p in range(0, n_words):
                for i, k in enumerate(indices[p]):
                    list_cluwords[p][k] = round(1 - distances[p][i], 2)

        np.savez_compressed('cluwords.npz',
                            data=list_cluwords,
                            index=np.asarray(labels_array),
                            cluwords=np.asarray(labels_array))


class W2VSim:
    def __init__(self, file_path_cluwords, save=True):
        self.__save = save
        self._file_path_cluwords = file_path_cluwords

    def _create_cluwords(self, input_vector_file, n_words, n_words_sim):
        word_vectors = KeyedVectors.load_word2vec_format(fname=input_vector_file, binary=False)
        model_vocab = list(word_vectors.vocab.keys())

        self.cluwords_df = pd.DataFrame(data=0,
                                        index=model_vocab,
                                        columns=model_vocab,
                                        dtype=np.float16)
        self.cluwords_df.values[[np.arange(n_words)] * 2] = 1

        for word in model_vocab:
            sim_words = word_vectors.similar_by_word(word, topn=n_words_sim)
            for sim_word in sim_words:
                self.cluwords_df[word][sim_word[0]] = float(sim_word[1])

        if self.__save:
            self.cluwords_df.to_csv(path_or_buf=self._file_path_cluwords,
                                    sep='\t', encoding='utf-8')

        return self.cluwords_df


class autovivify_list(dict):
    """Pickleable class to replicate the functionality of collections.defaultdict"""

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        """Override addition for numeric types when self is empty"""
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        """Also provide subtraction method"""
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


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


def find_word_clusters(labels_array, cluster_labels):
    """Read the labels array and clusters label and return the set of words in each cluster"""
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words
