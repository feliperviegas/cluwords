import os
import argparse
from parse_splits import ParseRaw
from utils import copy_file
from script_functions import create_embedding_models
from script_functions import generate_representation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',
                        action='store',
                        type=str,
                        dest='dataset',
                        required=True,
                        help='-d [dataset folder name]')
    parser.add_argument('-e', '--embedding',
                        action='store',
                        type=str,
                        dest='embedding',
                        default=None,
                        help='-e [embedding file name]')
    parser.add_argument('-dt', '--dtrain',
                        action='store',
                        type=str,
                        default='',
                        dest='dtrain',
                        help='-s [training features]')
    parser.add_argument('-ct', '--ctrain',
                        action='store',
                        type=str,
                        default='',
                        dest='ctrain',
                        help='-ct [training labels]')
    args = parser.parse_args()
    # Paths and files paths
    MAIN_PATH = '/cluwords_preprocess'
    EMBEDDING_RESULTS = 'fasttext_wiki'
    PATH_TO_SAVE_RESULTS = '{}/{}/results'.format(MAIN_PATH, EMBEDDING_RESULTS)
    EMBEDDING_DIMENSION = 300
    EMBEDDINGS_BIN_TYPE = False
    SUBLINEAR_TF = True
    DATASET = args.dataset
    N_THREADS = 6
    ALGORITHM_TYPE = 'knn_cosine'
    PATH_TO_SAVE_REPRESENTATION = '{}/{}/datasets/{dataset}'.format(MAIN_PATH, EMBEDDING_RESULTS,
                                                                    dataset=DATASET)

    print('{}/{}/datasets/{dataset}'.format(MAIN_PATH, EMBEDDING_RESULTS, dataset=args.dataset))
    try:
        os.mkdir('{}/{}'.format(MAIN_PATH, EMBEDDING_RESULTS))
    except FileExistsError:
        pass

    try:
        os.mkdir('{}/{}/datasets'.format(MAIN_PATH, EMBEDDING_RESULTS))
    except FileExistsError:
        pass

    try:
        os.mkdir('{path}'.format(path=PATH_TO_SAVE_REPRESENTATION))
    except FileExistsError:
        pass

    if args.split != '' and args.texts != '' and args.labels != '':
        ParseRaw(split_file=args.split,
                 document_file=args.texts,
                 label_file=args.labels,
                 fold=args.fold,
                 save_path=PATH_TO_SAVE_REPRESENTATION).run()
    elif args.dtrain != '' and args.ctrain != '':
        copy_file(source=args.dtrain, destination='{path}/d_train_data_{fold}.txt'
                  .format(path=PATH_TO_SAVE_REPRESENTATION,
                          fold=args.fold))
        copy_file(source=args.ctrain, destination='{path}/c_train_data_{fold}.txt'
                  .format(path=PATH_TO_SAVE_REPRESENTATION,
                          fold=args.fold))
    else:
        print('Error...')
        exit(1)

    # #Create the word2vec models for each dataset
    print('Filter embedding space to {} dataset...'.format(DATASET))

    n_words = create_embedding_models(dataset=DATASET,
                                      embedding_dimension=EMBEDDING_DIMENSION,
                                      embedding_file_path=args.embedding,
                                      embedding_type=EMBEDDINGS_BIN_TYPE,
                                      datasets_path='{path}/d_train_data_{fold}.txt'.format(path=PATH_TO_SAVE_REPRESENTATION,
                                                                                            fold=args.fold),
                                      path_to_save_model=PATH_TO_SAVE_REPRESENTATION,
                                      fold=args.fold)

    nearest_neighbors = 500 if n_words > 500 else n_words

    print('Build representations...')
    generate_representation(training_path='{path}/d_train_data_{fold}.txt'.format(path=PATH_TO_SAVE_REPRESENTATION,
                                                                                  fold=args.fold),
                            word_count=n_words,
                            path_to_save_model=PATH_TO_SAVE_REPRESENTATION,
                            path_to_save_results=PATH_TO_SAVE_RESULTS,
                            n_threads=N_THREADS,
                            k=nearest_neighbors,
                            threshold=0.4,
                            class_path='{path}/c_train_data_{fold}.txt'.format(path=PATH_TO_SAVE_REPRESENTATION,
                                                                               fold=args.fold),
                            algorithm_type=ALGORITHM_TYPE,
                            sublinear_tf=SUBLINEAR_TF,
                            fold=args.fold,
                            n_classes=args.n_classes,
                            dataset=DATASET)


if __name__ == '__main__':
    main()
