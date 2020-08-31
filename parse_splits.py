import unicodedata
import string
import re
import spacy
import argparse
from spacymoji import Emoji
from emoticons import parse_emoticons
from sklearn.feature_extraction.text import CountVectorizer

# To install language --> python -m spacy download en_core_web_sm
NLP = spacy.load("en_core_web_sm")
emoji = Emoji(NLP)
NLP.add_pipe(emoji, first=True)


class ParseRaw:
    def __init__(self,
                 split_file: str,
                 document_file: str,
                 label_file: str, 
                 fold: int,
                 save_path: str):
        self.split_file = split_file
        self.document_file = document_file
        self.label_file = label_file
        self.fold = fold
        self.save_path = save_path

    @staticmethod
    # Function to remove accents
    def remove_accented_chars(text):
        new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return new_text

    @staticmethod
    # Function to remove special characters
    def remove_special_characters(text):
        # define the pattern to keep
        pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
        return re.sub(pat, '', text)

    @staticmethod
    # Function to remove punctuation
    def remove_punctuation(text):
        text = ''.join([c for c in text if c not in string.punctuation])
        return text

    @staticmethod
    def transform_emoticons_emoji(text):
        pre_text = parse_emoticons(text)
        doc = NLP(pre_text)
        return ' '.join([word._.emoji_desc if word._.is_emoji is True else word.lower_ for word in doc])

    def text_preprocessing(self, text):
        pre_text_0 = self.transform_emoticons_emoji(text)
        pre_text_1 = self.remove_special_characters(pre_text_0)
        pre_text_2 = self.remove_accented_chars(pre_text_1)
        pre_text_3 = self.remove_punctuation(pre_text_2)
        return pre_text_3

    def tokenization(self, text):
        pre_text = self.text_preprocessing(text)
        doc = NLP(pre_text)

        # Token's list
        tokens = []

        # Remove punctuation
        doc = NLP(' '.join([t.lower_ for t in doc if t.is_punct is False]))

        # Receive tokens
        for token in doc:
            if token.is_stop is False and token.like_url is False and token.prefix_ != '@':
                tokens.append(token.lower_)

        doc = re.sub(' +', ' ', ' '.join(tokens))
        tokens = doc.split()

        return tokens

    def load_splits_ids(self, fold=0, with_val=False):
        fold_ref = 0
        with open(self.split_file, encoding='utf8', errors='ignore') as fileout:
            for line in fileout.readlines():
                parts = line.split(';')
                if len(parts) == 2:
                    train_index, test_index = parts
                    train_index = list(map(int, train_index.split()))
                    test_index = list(map(int, test_index.split()))
                    if fold_ref == fold:
                        return train_index, test_index
                elif len(parts) == 3:
                    train_index, val_index, test_index = parts
                    test_index = list(map(int, test_index.split()))
                    val_index = list(map(int, val_index.split()))
                    train_index = list(map(int, train_index.split()))
                    if not with_val:
                        train_index.extend(val_index)
                        val_index = []
                    if fold_ref == fold:
                        return train_index, val_index, test_index
                else:
                    raise Exception("")

                fold_ref += 1

        return None, None

    def read_file_preprocess_save_array(self):
        X_raw = []
        with open(self.document_file, 'r') as file:
            for document in file:
                doc_arrar = []
                tokens = self.tokenization(document.strip())
                for iter_token in range(0, len(tokens)):
                    doc_arrar.append(tokens[iter_token])

                X_raw.append(doc_arrar)

            file.close()

        return X_raw

    def read_file_save_array(self):
        X_raw = []
        with open(self.document_file, 'r') as file:
            for document in file:
                doc_arrar = []
                tokens = document.strip().split(' ')
                for iter_token in range(0, len(tokens)):
                    doc_arrar.append(tokens[iter_token])

                X_raw.append(doc_arrar)

            file.close()

        return X_raw

    def read_labels(self):
        y = []
        with open(self.label_file, 'r') as file:
            for document in file:
                y.append(document.strip())

            file.close()

        return y

    @staticmethod
    def get_array(data, idxs):
        return [data[idx] for idx in idxs]

    def save_train_file(self, X_train, y_train, train_idx):
        try:
            with open("{path}/d_train_data_{fold}.txt".format(path=self.save_path,
                                                              fold=self.fold), 'w') as out:
                doc_train_id = 0
                for doc in X_train:
                    if len(doc) > 0:
                        out.write('{}'.format(doc[0]))
                        for word_id in range(1, len(doc)):
                            out.write(' {}'.format(doc[word_id]))

                        out.write('\n')
                    else:
                        out.write('\n')
                    doc_train_id += 1

                out.close()
        except Exception as e:
            print(f'Training Document Error - {doc_train_id}')
            print('Document:')
            print(doc)
            print(train_idx[doc_train_id])
            print(f'{e}')

        try:
            with open("{path}/c_train_data_{fold}.txt".format(path=self.save_path,
                                                              fold=self.fold), 'w') as out:
                doc_test_id = 0
                for doc in y_train:
                    out.write('{}\n'.format(doc[0]))
                    doc_test_id += 1

                out.close()
        except Exception as e:
            print(f'Test Document Error - {doc_test_id}')
            print(f'{e}')

    def run(self, apply_preprocess=False):
        if apply_preprocess:
            X_raw = self.read_file_preprocess_save_array()
        else:
            X_raw = self.read_file_save_array()

        y = self.read_labels()
        # train_idx, val_idx, test_idx = self.load_splits_ids(fold=self.fold)
        train_idx, test_idx = self.load_splits_ids(fold=self.fold)
        X_train = self.get_array(X_raw, train_idx)
        y_train = self.get_array(y, train_idx)
        X_test = self.get_array(X_raw, test_idx)
        self.save_train_file(X_train, y_train, train_idx)

    def get_vocabulary(self):
        X_raw = self.read_file_save_array()
        train_idx, test_idx = self.load_splits_ids(fold=self.fold)
        X_train = self.get_array(X_raw, train_idx)

        X_raw = []
        for doc in X_train:
            X_raw.append(' '.join(word for word in doc))

        del X_train
        dataset_cv = CountVectorizer().fit(X_raw)
        dataset_words = dataset_cv.get_feature_names()

        with open(f'{self.save_path}/vocabulary.txt', 'w') as output_file:
            for word in dataset_words:
                output_file.write(f'{word}\n')

            output_file.close()

    def preprocess_texts(self, output_file):
        id = 0
        X_raw = []
        with open(output_file, 'w') as output:
            with open(self.document_file, 'r') as file:
                for document in file:
                    doc_arrar = []
                    tokens = self.tokenization(document.strip())
                    for iter_token in range(0, len(tokens)):
                        doc_arrar.append(tokens[iter_token])

                    X_raw.append(doc_arrar)
                    id += 1
                    if id == 100:
                        for text in X_raw:
                            output.write("{text}\n".format(text=' '.join(w for w in text)))

                        X_raw.clear()
                        id = 0

                file.close()

            if id != 0:
                for text in X_raw:
                    output.write("{text}\n".format(text=' '.join(w for w in text)))

            output.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--texts',
                        action='store',
                        type=str,
                        dest='texts',
                        help='-t [texts folder name]')
    parser.add_argument('-l', '--labels',
                        action='store',
                        type=str,
                        dest='labels',
                        help='-l [labels folder name]')
    parser.add_argument('-s', '--split',
                        action='store',
                        type=str,
                        default='',
                        dest='split',
                        help='-s [splits folder name]')
    parser.add_argument('-f', '--fold',
                        action='store',
                        type=int,
                        dest='fold',
                        required=True,
                        help='--fold [TRAIN/TEST FOLD]')
    parser.add_argument('-p', '--path',
                        action='store',
                        type=str,
                        default='.',
                        dest='path',
                        help='-p [output path]')
    parser.add_argument('-o', '--output',
                        action='store',
                        type=str,
                        default='',
                        dest='output',
                        help='-o [output file]')
    args = parser.parse_args()
    parse = ParseRaw(split_file=args.split,
                     document_file=args.texts,
                     label_file=args.labels,
                     fold=args.fold,
                     save_path=args.path)

    parse.preprocess_texts(output_file=args.output)


if __name__ == '__main__':
    main()
