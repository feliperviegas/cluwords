import os
import pandas as pd
import numpy as np
from pyjarowinkler import distance
import timeit
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from cluwords import Cluwords, CluwordsTFIDF
from metrics import Evaluation
from embedding import CreateEmbeddingModels
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer


def _nearest_neighbors(X_topic, X_raw, vocab, n_topics, dataset):
    X = _raw_tf(X_raw, vocab, binary=True)
    neigh = NearestNeighbors(n_neighbors=n_topics, algorithm='auto', metric='cosine')
    neigh.fit(X_topic)
    dist, ind = neigh.kneighbors(X)
    output = open('document_distribution_{}'.format(dataset), 'w')
    for doc in range(0, dist.shape[0]):
        topic_dist = np.zeros(dist.shape[1])
        for index in range(0, dist.shape[1]):
            topic_index = ind[index]
            topic_dist[topic_index] = 1. - dist[doc, topic_index]

        total_dist = np.sum(topic_dist)
        output.write('{} '.format(doc))
        for index in range(0, dist.shape[1]):
            output.write(' {}:{}'.format(index, round(topic_dist[index]/total_dist if total_dist > 0 else .0, 4)))

        output.write('\n')

    return

def _raw_tf(documents, vocab, binary=False):
    tf_vectorizer = CountVectorizer(max_features=len(vocab), binary=binary, vocabulary=vocab)
    tf = tf_vectorizer.fit_transform(documents)
    return tf


def get_one_hot_topics(topics, top, vocab, dataset):
    one_hot_topics = []
    for topic in topics:
        topic_top = topic[:top]
        one_hot_topic = np.zeros(len(vocab))
        for word in topic_top:
            index_vocab = np.argwhere(vocab == word)[0]
            one_hot_topic[index_vocab] = 1

        one_hot_topics.append(one_hot_topic)

    one_hot_topics = np.array(one_hot_topics)
    np.savez_compressed('one_hot_topics_{}.npz'.format(dataset),
                        one_hot=one_hot_topics)
    return one_hot_topics


def parse_topics(topics):
    topics_t = []
    for topic in topics:
        topic_t = topic.split(' ')
        topics_t.append(topic_t)

    return topics_t


def remove_redundant_words(topics):
    topics_t = []
    for topic in topics:
        filtered_topic = []
        insert_word = np.ones(len(topic))
        for w_i in range(0, len(topic)-1):
            if insert_word[w_i]:
                filtered_topic.append(topic[w_i])
                for w_j in range((w_i + 1), len(topic)):
                    if distance.get_jaro_distance(topic[w_i], topic[w_j], winkler=True, scaling=0.1) > 0.75:
                        insert_word[w_j] = 0

        topics_t.append(filtered_topic)

    return topics_t


def top_words(model, feature_names, n_top_words):
    topico = []
    for topic_idx, topic in enumerate(model.components_):
        top = ''
        top2 = ''
        top += ' '.join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
        top2 += ''.join(str(sorted(topic)[:-n_top_words - 1:-1]))

        topico.append(str(top))

    return topico


def print_results(cluwords_freq, cluwords_docs, path_to_save_results, topics, n_docs):
    print(path_to_save_results)
    for t in [5, 10, 20]:
        with open('{}/result_topic_{}.txt'.format(path_to_save_results, t), 'w') as f_res:
            f_res.write('Topics {}\n'.format(t))
            f_res.write('Topics:\n')
            topics_t = []
            for topic in topics:
                topics_t.append(topic[:t])
                for word in topic[:t]:
                    f_res.write('{} '.format(word))

                f_res.write('\n')

            # coherence = Evaluation.coherence(topics, cluwords_freq, cluwords_docs)
            # f_res.write('Coherence: {} ({})\n'.format(np.round(np.mean(coherence), 4),
                                                      # np.round(np.std(coherence), 4)))
            # f_res.write('{}\n'.format(coherence))

            pmi, npmi = Evaluation.pmi(topics=topics_t,
                                       word_frequency=cluwords_freq,
                                       term_docs=cluwords_docs,
                                       n_docs=n_docs,
                                       n_top_words=t)
            # f_res.write('PMI: {} ({})\n'.format(np.round(np.mean(pmi), 4), np.round(np.std(pmi), 4)))
            # f_res.write('{}\n'.format(pmi))
            f_res.write('NPMI:\n')
            for score in npmi:
                f_res.write('{}\n'.format(score))

            f_res.write('avg NPMI: {} ({})\n'.format(np.round(np.mean(npmi), 4), np.round(np.std(npmi), 4)))

            # w2v_l1 = Evaluation.w2v_metric(topics, t, path_to_save_model, 'l1_dist', dataset)
            # f_res.write('W2V-L1: {} ({})\n'.format(np.round(np.mean(w2v_l1), 4), np.round(np.std(w2v_l1), 4)))
            # f_res.write('{}\n'.format(w2v_l1))

            f_res.close()


def save_results(model, tfidf_feature_names, path_to_save_model, dataset, cluwords_freq,
                 cluwords_docs, path_to_save_results):
    res_mean = []
    coherence_mean = ['coherence']
    lcp_mean = ['lcp']
    npmi_mean = ['npmi']
    w2v_l1_mean = ['w2v-l1']

    for t in [5, 10, 20]:
        topics = top_words(model, tfidf_feature_names, t)

        # Write topics in a file
        file = open('{}/topics_{}.txt'.format(path_to_save_results, t), 'w+')
        file.write('TOPICS WITH {} WORDS\n\n'.format(t))
        for i, topic in enumerate(topics):
            file.write('Topic %d\n' % i)
            file.write('%s\n' % topic)
        file.close()

        coherence = Evaluation.coherence(topics, cluwords_freq, cluwords_docs)
        coherence_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(coherence),
                                                           np.std(coherence))])

        lcp = Evaluation.lcp(topics, cluwords_freq, cluwords_docs)
        lcp_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(lcp),
                                                     np.std(lcp))])

        _, npmi = Evaluation.pmi(topics, cluwords_freq, cluwords_docs,
                                 sum([freq for word, freq in cluwords_freq.items()]), t)
        npmi_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(npmi),
                                                      np.std(npmi))])

        w2v_l1 = Evaluation.w2v_metric(topics, t, path_to_save_model, 'l1_dist', dataset)
        w2v_l1_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(w2v_l1),
                                                        np.std(w2v_l1))])
    res_mean.extend([coherence_mean, lcp_mean, npmi_mean, w2v_l1_mean])

    df_mean = pd.DataFrame(res_mean, columns=['metric', '5 words', '10 words', '20 words'])

    df_mean.to_csv(path_or_buf='{}/results.csv'.format(path_to_save_results))


def create_embedding_models(dataset, embedding_file_path, embedding_type, datasets_path, path_to_save_model):
    # Create the word2vec models for each dataset
    word2vec_models = CreateEmbeddingModels(embedding_file_path=embedding_file_path,
                                            embedding_type=embedding_type,
                                            document_path=datasets_path,
                                            path_to_save_model=path_to_save_model)
    n_words = word2vec_models.create_embedding_models(dataset)

    return n_words


def generate_topics(dataset, word_count, path_to_save_model, datasets_path,
                    path_to_save_results, n_threads, k, threshold, cossine_filter,
                    has_class, class_path, n_components, algorithm_type):
    # Path to files and directories
    embedding_file_path = """{}/{}.txt""".format(path_to_save_model, dataset)
    dataset_file_path = """{}/{}Pre.txt""".format(datasets_path, dataset)
    path_to_save_results = '{}/{}'.format(path_to_save_results, dataset)

    try:
        os.mkdir('{}'.format(path_to_save_results))
    except FileExistsError:
        pass

    Cluwords(algorithm=algorithm_type,
             embedding_file_path=embedding_file_path,
             n_words=word_count,
             k_neighbors=k,
             threshold=threshold,
             n_jobs=n_threads
             )

    cluwords = CluwordsTFIDF(dataset_file_path=dataset_file_path,
                             n_words=word_count,
                             cossine_filter=cossine_filter,
                             path_to_save_cluwords=path_to_save_results,
                             class_file_path=class_path,
                             has_class=has_class)
    print('Computing TFIDF...')
    cluwords_tfidf = cluwords.fit_transform()
    # cluwords_tfidf = csr_matrix(cluwords_tfidf)  # Convert the cluwords_tfidf array matrix to a sparse cluwords

    start = timeit.default_timer()
    # Fit the NMF model
    print("\nFitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (cluwords.n_documents, cluwords.n_cluwords))
    nmf = NMF(n_components=n_components,
              random_state=1,
              alpha=.1,
              l1_ratio=.5).fit(cluwords_tfidf)

    end = timeit.default_timer()
    print("NMF done in {}.".format(end - start))

    with open('{}/matrix_w.txt'.format(path_to_save_results), 'w') as f:
        w = nmf.fit_transform(cluwords_tfidf)  # matrix W = m x k
        h = nmf.components_.transpose()  # matrix H = n x k
        print('W: {} H:{}'.format(w.shape, h.shape))
        for x in range(w.shape[0]):
            for y in range(w.shape[1]):
                f.write('{} '.format(w[x][y]))
            f.write('\n')
        f.close()
        del w
        del h

    vocab_cluwords = cluwords.vocab_cluwords
    documents = cluwords.documents
    del cluwords
    #Load topics
    topics = top_words(nmf, list(vocab_cluwords), 101)

    # Load Cluwords representation for metrics
    cluwords_freq, cluwords_docs, n_docs = Evaluation.count_tf_idf_repr(topics,
                                                                        vocab_cluwords,
                                                                        cluwords_tfidf.transpose())
    topics = parse_topics(topics)
    one_hot_topics = get_one_hot_topics(topics, 101, np.array(vocab_cluwords), dataset)
    _nearest_neighbors(one_hot_topics, documents, vocab_cluwords, n_components, dataset)
    topics = remove_redundant_words(topics)

    # Remove variable
    del cluwords_tfidf

    # print('n_terms: {}'.format(n_cluwords))
    # print('words1: {}'.format(cluwords_vocab))
    # print('word_frequency: {}'.format(cluwords_freq))
    # print('term_docs: {}'.format(cluwords_docs))

    print_results(cluwords_freq=cluwords_freq,
                  cluwords_docs=cluwords_docs,
                  path_to_save_results=path_to_save_results,
                  topics=topics,
                  n_docs=n_docs
                  )
