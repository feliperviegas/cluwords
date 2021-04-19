from script_functions import create_embedding_models, generate_topics
import os

DATASETS = {
    '20News': 29442
    # 'acm': 16333,
    # 'ang': 1903,
    # 'drop': 2427,
    # 'ever': 6296,
    # 'face': 5152,
    # 'info': 6061,
    # 'pinter': 2058,
    # 'trip': 3147,
    # 'tweets': 8002,
    # 'uber': 5504,
    # 'wpp': 1775
}

# Paths and files paths
MAIN_PATH = '/home/felipeviegas/Codes_phd/'
# EMBEDDING_RESULTS = 'fasttext_wiki_mahalanobis'
EMBEDDING_RESULTS = 'fasttext_wiki'
PATH_TO_SAVE_RESULTS = '{}/cluwords/{}/results'.format(MAIN_PATH, EMBEDDING_RESULTS)
PATH_TO_SAVE_MODEL = '{}/cluwords/{}/datasets/gn_w2v_models'.format(MAIN_PATH, EMBEDDING_RESULTS)
DATASETS_PATH = '{}cluwords/bases'.format(MAIN_PATH)
# DATASETS_PATH = '/mnt/d/Work/textual_datasets'
CLASS_PATH = '{}cluwords/bases/textual_topic_datasets/acm_so_score_Pre'.format(MAIN_PATH)
HAS_CLASS = False
# EMBEDDINGS_FILE_PATH = '{}/GoogleNews-vectors-negative300.bin'.format(MAIN_PATH)
# EMBEDDINGS_BIN_TYPE = True
# EMBEDDINGS_FILE_PATH = '{}cluwords/wiki-news-300d-1M.vec'.format(MAIN_PATH)
# EMBEDDINGS_FILE_PATH = '{}cluwords/embedding_viegas_concat/embedding_acmPre_bert_concat_avg'.format(MAIN_PATH)
EMBEDDINGS_BIN_TYPE = False
DATASET = 'acm'
N_THREADS = 6
N_COMPONENTS = 20
# ALGORITHM_TYPE = 'knn_mahalanobis'
ALGORITHM_TYPE = 'knn_cosine'

for DATASET, values in DATASETS.items():

    # EMBEDDINGS_FILE_PATH = '{}cluwords/viegas_dataset_avg_max/embedding_{}Pre_max'.format(MAIN_PATH,
    #                                                                                       DATASET)
    EMBEDDINGS_FILE_PATH = '{}cluwords/embedding_viegas_concat/embedding_{}Pre_bert_concat_avg'.format(MAIN_PATH,
                                                                                                       DATASET)
    # Creates directories if they don't exist
    try:
        os.mkdir('{}/cluwords/{}'.format(MAIN_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/cluwords/{}/results'.format(MAIN_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/cluwords/{}/datasets'.format(MAIN_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/cluwords/{}/datasets/gn_w2v_models'.format(MAIN_PATH, EMBEDDING_RESULTS))
    except FileExistsError:
        pass

    # Create the word2vec models for each dataset
    print('Filter embedding space to {} dataset...'.format(DATASET))
    n_words = create_embedding_models(dataset=DATASET,
                                      embedding_file_path=EMBEDDINGS_FILE_PATH,
                                      embedding_type=EMBEDDINGS_BIN_TYPE,
                                      datasets_path=DATASETS_PATH,
                                      path_to_save_model=PATH_TO_SAVE_MODEL)

    print('Build topics...')
    results = generate_topics(dataset=DATASET,
                              word_count=n_words,
                              path_to_save_model=PATH_TO_SAVE_MODEL,
                              datasets_path=DATASETS_PATH,
                              path_to_save_results=PATH_TO_SAVE_RESULTS,
                              n_threads=N_THREADS,
                              algorithm_type=ALGORITHM_TYPE,
                              # k=n_words,
                              k=500,
                              threshold=0.4,
                              cossine_filter=0.9,
                              class_path=CLASS_PATH,
                              has_class=HAS_CLASS,
                              n_components=N_COMPONENTS)
