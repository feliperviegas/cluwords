from time import time
from gensim.models import KeyedVectors
from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer


class CreateEmbeddingModels:
    """
    Description
    -----------
    Creates a Word2Vec model for each dataset passed by parameters (based on Google News Word2Vec model).

    Parameters
    ----------
    embedding_file_path: str
        The complete embedding pre-treined model file path.
    document_path: str
        The raw document file path.
    path_to_save_model: str
        Path to save the result Word2Vec model.
    embedding_type: boolean
        To specify if the pre-treined embedding model is binary.
    """

    def __init__(self, embedding_file_path, embedding_type, document_path, path_to_save_model):
        self.document_path = document_path
        self.path_to_save_model = path_to_save_model
        if embedding_file_path:
            print('Reading embedding...')
            self.model = self._read_embedding(embedding_file_path, embedding_type)
        else:
            print('Creating embedding...')
            documents = self._read_raw_dataset(document_path)
            model_fasttext = FastText(documents, size=300, window=5, min_count=5, workers=4, sg=1)
            self.write_embedding(model_fasttext, documents)
            self.model = model_fasttext

    def filter_embedding_models(self, dataset, dimension, fold):
        """
        Description
        -----------
        Create the cluwords for one dataset,

        Parameters
        ----------
        dataset: str
            The name of the dataset in documents_path to create the filtered embedding model.

        Return
        ------
        n_words: int
            Number of words of dataset present in the pre-treined model.
        """
        documents = self._read_raw_dataset(self.document_path)

        # Count the words in dataset
        dataset_cv = CountVectorizer().fit(documents)
        dataset_words = dataset_cv.get_feature_names()

        # Select just the words in dataset from Google News Word2Vec Model
        words_values = []
        for word in dataset_words:
            aux = None
            if word in self.model:
                aux = str(word)
                for latents in self.model[word]:
                    aux += ' ' + str(latents)
            # TODO BERT Embedding - Just leave for further analysis
            # elif ('##' + str(word)) in self.model:
            #     aux = word
            #     for latents in self.model[('##' + str(word))]:
            #         aux += ' ' + str(latents)

            if aux:
                words_values.append(aux)

        # filtered_vocabulary = []
        # for wv in words_values:
        #     filtered_vocabulary.append(wv.split(' ')[0])

        n_words = len(words_values)  # Number of words selected

        print('{}:{}'.format(dataset, n_words))

        # save .txt model
        file = open("""{path}/{dataset}_embedding_{fold}.txt""".format(path=self.path_to_save_model,
                                                                       dataset=dataset,
                                                                       fold=fold), 'w+', encoding="utf-8")
        file.write('{0} {1}\n'.format(n_words, dimension))
        for word_vec in words_values:
            file.write("%s\n" % word_vec)

        return n_words

    @staticmethod
    def _read_embedding(embedding_file_path, binary):
        t0 = time()
        model = KeyedVectors.load_word2vec_format(embedding_file_path, binary=binary)
        print('Embedding model read in %0.3fs.' % (time() - t0))
        return model

    @staticmethod
    def write_embedding(model, documents):
        dataset_cv = CountVectorizer().fit(documents)
        dataset_words = dataset_cv.get_feature_names()
        with open('embedding.txt', 'w') as file:
            file.write(f'{len(dataset_words)} 300\n')
            for word in dataset_words:
                file.write(f'{word}')
                for dimension in model.wv[word]:
                    file.write(f' {dimension}')

                file.write('\n')

            file.close()

    @staticmethod
    def _read_raw_dataset(document_path):
        arq = open(document_path, 'r', encoding="utf-8")
        doc = arq.readlines()
        arq.close()
        documents = list(map(str.rstrip, doc))

        return documents
