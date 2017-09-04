import gensim
import pickle
import string
import numpy as np
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class X2Vec:
    """
    General template for an X2Vec model generator.
    """

    CACHED_STOP_WORDS = set(stopwords.words('english'))
    PUNCTUATION = string.punctuation
    BAD_WORDS = set(['\'\'', '``', 'oed'] + list(string.ascii_lowercase) + list(string.ascii_uppercase))

    def __init__(self, window_size=5, size=100, min_count=5):
        """
        Init function for our model.
        :param window_size: Size of the window for Word2Vec
        :param size: Size of model for Word2Vec
        :param min_count: Minimum amount of times a word must appear to be trained on
        :return:
        """
        # the Word2Vec model
        self.model = None

        # Word2Vec parameters
        self.window_size = window_size
        self.model_size = size
        self.min_count = min_count

        # remember the state of the model and keep track of words and tokens
        self.tokenized_corpus = None
        self.known_words = set()
        self.token_dict = dict()
        self.trained = False
        self.tokenized = False

        # filepath parameters
        self.corpus_path = None

    @staticmethod
    def read_corpus(filename, num_lines=-1):
        """
        Static method that reads a given corpus file and returns it as a string.
        :param filename: a path to the corpus file.
        :param num_lines: The amount of lines to read from the corpus
        :return: the corpus as a string.
        """
        corpus = []
        with open(filename, encoding='latin-1') as f_corpus:
            cur_sent = []
            content = f_corpus.readlines()
            for line in content[:num_lines]:
                line = line.strip()
                if line == '</s>':
                    corpus.append(cur_sent)
                    cur_sent = []
                elif line == '<s>' or line.startswith('<text'):
                    continue
                else:
                    cur_sent.append(line)
        return corpus

    def clean_corpus(self, corpus):
        """
        Method that cleans a corpus from stop words and bad words, also makes each word lowercase
        :param corpus: the corpus as a string.
        :return: the corpus as an all lower-case string without any stopwords, punctuation or irrelevant characters.
        """
        cleaned_corpus = [[word.lower() for word in sentence if (word not in self.CACHED_STOP_WORDS)
                           and (word not in self.PUNCTUATION) and (word not in self.BAD_WORDS)] for sentence in corpus]
        # keep all known words in the corpus in a set.
        self.known_words = set([word for sentence in cleaned_corpus for word in sentence])
        return cleaned_corpus

    def tokenize_corpus(self, corpus, tokenize=True):
        """
        A method that must be implemented by each inheriting model. Deals with the tokenization of the corpus prior
        to training.
        :param corpus: the corpus as a string.
        :param tokenize: True if should tokenize the corpus beforehand.
        :return: the tokenized corpus.
        """
        raise NotImplementedError

    def train_model(self, corpus, train=True, tokenize=True):
        """
        Method that handles the cleaning, tokenizing of the corpus and training of the model on that corpus.
        :param corpus: The corpus, in a format ready for tokenization
        :param train: Whether to train the model
        :param tokenize: Whether to tokenize the corpus
        :param clean: Whether to clean the corpus
        :return:
        """

        corpus = self.clean_corpus(corpus)
        if tokenize:
            self.tokenized_corpus = self.tokenize_corpus(corpus, tokenize=tokenize)
            self.tokenized = True
        else:
            self.tokenized_corpus = corpus
        if train:
            print('Starting to train model')
            try:
                self.model = gensim.models.Word2Vec(self.tokenized_corpus, min_count=self.min_count,
                                                    window=self.window_size, size=self.model_size, iter=20)
            except RuntimeError:
                print('No word appeared %d times, reran with min_count=1'%self.min_count)
                self.model = gensim.models.Word2Vec(self.tokenized_corpus, min_count=1,
                                                    window=self.window_size, size=self.model_size, iter=20)
            self.trained = True

    def save_model(self, model_path):
        """
        :param model_path: the path to save the model to.
        """
        pickle.dump(self, open(model_path, "wb"))

    def visualize_embedding(self, num_words=1000):
        print('Visualizing embedding')
        # get vocab by unique numbers
        vocab_num = self.model[self.model.wv.vocab]
        # get the actual vocab names
        vocab = self.model.wv.vocab
        # get the count of each word in vocab
        vocab_count = []
        vocab_words = []
        for word, vocab_obj in self.model.wv.vocab.items():
            vocab_count.append(int(vocab_obj.count))
            vocab_words.append(word)
        vocab_count = np.array(vocab_count)
        vocab_words = np.array(vocab_words)

        # arg sort it in descending order
        sorted_vocab_ind = vocab_count.argsort()[::-1]

        # create list of only the num_words highest appearing vocab_num and vocab
        low_ind = min(600, len(vocab_num))
        high_ind = min(num_words + 600, len(vocab_num))
        highest_vocab = vocab_words[sorted_vocab_ind][low_ind:high_ind]
        highest_vocab_num = vocab_num[sorted_vocab_ind][low_ind:high_ind]

        # Run the TSNE algorithm and plot the results
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(highest_vocab_num)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        for label, x, y in zip(highest_vocab, X_tsne[:, 0], X_tsne[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.show()

    @staticmethod
    def load_model(model_path):
        """
        :param model_path: the path to read the model file from.
        :return: a python object of the model.
        """
        return pickle.load(open(model_path, "rb"))

