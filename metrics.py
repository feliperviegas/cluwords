import numpy as np
import scipy.spatial.distance as sci_dist
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class Evaluation:
    @staticmethod
    def count(docPre):
        # print docPre
        count_vec = CountVectorizer(binary=True)  # , min_df=0.0,max_df=1.0, token_pattern=r"(?u)\b\w+\b")
        count = count_vec.fit_transform(docPre)
        words = list(map(str, count_vec.get_feature_names()))

        # Sem print words nao funciona kkk
        # print words
        # input("ok")
        # exit()

        n_terms = len(words)

        count_t = count.transpose()

        word_frequency = {}
        term_docs = {}

        # print type(count)
        # trocar essa parte (change)
        for i in range(n_terms):
            word_frequency[words[i]] = float(count_t[i].getnnz(1))
            term_docs[words[i]] = set(count_t[i].nonzero()[1])

        # print term_docs

        # word_frequency['mm'] = word_frequency['tum']
        # term_docs['mm'] = term_docs['tum']

        # print word_frequency["video"]
        # print word_frequency["call"]

        return n_terms, words, word_frequency, term_docs


    @staticmethod
    def count_tf_idf_repr(topics, cw_words, tf_idf_t):
        cw_frequency = {}
        cw_docs = {}
        for iter_topic in topics:
            topic = iter_topic.split(' ')
            for word in topic:
                word_index = np.where(cw_words == word)[0]
                cw_frequency[word] = float(tf_idf_t[word_index].getnnz(1))
                cw_docs[word] = set(tf_idf_t[word_index].nonzero()[1])

        n_docs = 0
        for _cw in range(tf_idf_t.shape[0]):
            n_docs += float(tf_idf_t[_cw].getnnz(1))

        return cw_frequency, cw_docs, n_docs

    @staticmethod
    def coherence(topic, word_frequency, term_docs):
        coherence = []

        for t in range(len(topic)):
            topico = topic[t]
            top_w = topico.split(" ")

            coherence_t = 0.0
            for i in range(1, len(top_w)):
                for j in range(0, i):
                    cont_wi = word_frequency[top_w[j]]
                    cont_wi_wj = float(
                        len(term_docs[top_w[j]].intersection(term_docs[top_w[i]])))
                    coherence_t += np.log((cont_wi_wj + 1.0) / cont_wi)

            coherence.append(coherence_t)

        return coherence

    @staticmethod
    def tfidf_coherence(topic, doc_pre):
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
        tfidf = tfidf_vectorizer.fit_transform(doc_pre)
        words = list(map(str, tfidf_vectorizer.get_feature_names()))
        tfidf_t = tfidf.transpose()

        tfidf_coherence = []
        term_docs = {}
        n_terms = len(words)

        for i in range(n_terms):
            term_docs[words[i]] = set(tfidf_t[i].nonzero()[1])

        for t in range(len(topic)):
            topico = topic[t]
            top_w = topico.split(" ")

            tfidf_coh_t = 0.0
            for i in range(1, len(top_w)):
                for j in range(0, i):
                    ti = top_w[j]
                    tj = top_w[i]

                    wi_index = words.index(ti)
                    wj_index = words.index(tj)

                    wi = tfidf_t[wi_index]
                    wj = tfidf_t[wj_index]

                    sum_tfidf_wi = sum(wi.data)

                    docs_with_wi = set(wi.nonzero()[1])
                    docs_with_wj = set(wj.nonzero()[1])

                    docs_with_wi_and_wj = docs_with_wi.intersection(docs_with_wj)

                    aux = 0.0
                    for k in docs_with_wi_and_wj:
                        aux += tfidf_t.getrow(wi_index).getcol(
                            k).data[0] * tfidf_t.getrow(wj_index).getcol(k).data[0]

                    tfidf_coh_t += np.log(((aux + 0.01) / sum_tfidf_wi))

            tfidf_coherence.append(tfidf_coh_t)

        return tfidf_coherence

    @staticmethod
    def pmi(topics, word_frequency, term_docs, n_docs, n_top_words):
        pmi = []
        npmi = []

        n_top_words = float(n_top_words)

        for t in range(len(topics)):
            top_w = topics[t]
            # top_w = topico.split(' ')

            pmi_t = 0.0
            npmi_t = 0.0

            for j in range(1, len(top_w)):
                for i in range(0, j):
                    ti = top_w[i]
                    tj = top_w[j]

                    c_i = word_frequency[ti]
                    c_j = word_frequency[tj]
                    c_i_and_j = len(term_docs[ti].intersection(term_docs[tj]))

                    pmi_t += np.log(((c_i_and_j + 1.0) / float(n_docs)) /
                                    ((c_i * c_j) / float(n_docs) ** 2))

                    npmi_t += -1.0 * np.log((c_i_and_j + 0.01) / float(n_docs))

            peso = 1.0 / (n_top_words * (n_top_words - 1.0))

            pmi.append(peso * pmi_t)
            npmi.append(pmi_t / npmi_t)

        return pmi, npmi

    @staticmethod
    def lcp(topic, word_frequency, term_docs):
        lcp = []

        for t in range(len(topic)):
            topico = topic[t]
            top_w = topico.split(' ')

            lcp_t = 0
            for j in range(1, len(top_w)):
                for i in range(0, j):
                    ti = top_w[i]
                    tj = top_w[j]

                    c_i = word_frequency[ti]
                    c_i_and_j = len(term_docs[ti].intersection(term_docs[tj]))

                    lcp_t += np.log(c_i_and_j / c_i)

            lcp.append(lcp_t)

        return lcp

    @staticmethod
    def w2v_metric(topics, t, path_to_save_model, distance_type, dataset, embedding_type=False):
        word_vectors = KeyedVectors.load_word2vec_format(
            fname='{}/{}.txt'.format(path_to_save_model, dataset), binary=embedding_type)
        model = word_vectors.wv
        values = []

        for topic in topics:
            words = topic.split(' ')
            value = Evaluation._calc_dist_2(words, model, distance_type, t)
            values.append(value)

        return values

    @staticmethod
    def _calc_dist_2(words, w2v_model, distance_type, t):
        l1_dist = 0
        l2_dist = 0
        cos_dist = 0
        coord_dist = 0
        t = float(t)

        for word_id1 in range(len(words)):
            for word_id2 in range(word_id1 + 1, len(words)):
                # Calcular L1 w2v metric
                l1_dist += (sci_dist.euclidean(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

                # Calcular L2 w2v metric
                l2_dist += (sci_dist.sqeuclidean(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

                # Calcular cos w2v metric
                cos_dist += (sci_dist.cosine(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

                # Calcular coordinate w2v metric
                coord_dist += (sci_dist.sqeuclidean(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

        if distance_type == 'l1_dist':
            return l1_dist / (t * (t - 1.0))
        elif distance_type == 'l2_dist':
            return l2_dist / (t * (t - 1.0))
        elif distance_type == 'cos_dist':
            return cos_dist / (t * (t - 1.0))
        elif distance_type == 'coord_dist':
            return coord_dist / (t * (t - 1.0))

        return .0
