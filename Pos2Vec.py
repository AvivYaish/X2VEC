import string
import nltk
from nltk.corpus import stopwords
from X2Vec import X2Vec
import itertools

CACHED_STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation
BAD_WORDS = ['\'\'', '``']


class Pos2Vec(X2Vec):
    """
    A word embedding model that generates different embeddings for identical words based on their POS.
    """

    def clean_corpus(self, corpus):
        """
        Method that cleans a corpus from stop words and bad words, also makes each word lowercase
        :param corpus: the corpus as a string.
        :return: the corpus as an all lower-case string without any stopwords, punctuation or irrelevant characters.
        """
        cleaned_corpus = [word.lower() for word in corpus[:] if (word not in CACHED_STOP_WORDS) and
                              (word not in PUNCTUATION) and (word not in BAD_WORDS)]
        # keep all known words in the corpus in a set.
        self.known_words = set([word for sentence in cleaned_corpus for word in sentence])
        return cleaned_corpus

    def tokenize_corpus(self, corpus, tokenize=True):
        """
        Method that tokenizes the corpus prior to training. For each word in the corpus we compute the POS of that
        word and change it with word_POS. For example: cat can become cat_NN
        :param corpus: the corpus as a string.
        :param tokenize: True if should tokenize the corpus beforehand.
        :return: the tokenized corpus.
        """
        preprocessed_corpus = corpus
        self.known_words = set([word for word in preprocessed_corpus])
        preprocessed_corpus_joined = ' '.join(preprocessed_corpus)
        if tokenize:
            print('Starting to tag corpus')
            tokenized_corpus = nltk.tokenize.word_tokenize(preprocessed_corpus_joined)
            corpus_tags = nltk.pos_tag(tokenized_corpus)
            tagged_corpus = [word + '_' + pos for word, pos in corpus_tags]
            for word, tag in zip(preprocessed_corpus, tagged_corpus):
                if tag is None:
                    continue
                cur_set = self.token_dict.get(word, set())
                cur_set.add(tag)
                self.token_dict[word] = cur_set
            final_corpus = [tagged_corpus[i:i + 10] for i in range(0, len(tagged_corpus), 10)]
        return final_corpus

    @staticmethod
    def read_corpus(filename, num_lines=-1):
        """
        Static method that reads a given corpus file and returns it as a string.
        :param filename: a path to the corpus file.
        :param num_lines: The amount of lines to read from the corpus
        :return: the corpus as a list of strings.
        """
        with open(filename, encoding='latin-1') as f_corpus:
            sentences = []
            cur_sent = []
            content = f_corpus.readlines()
            for line in content[:num_lines]:
                line = line.strip()
                if line == '</s>':
                    sentences.append(cur_sent)
                    cur_sent = []
                elif line == '<s>' or line.startswith('<text'):
                    continue
                else:
                    cur_sent.append(line)
        corpus = list(itertools.chain.from_iterable(sentences))
        return corpus

# m = Pos2Vec()
# print('Reading corpus')
# corpus = Pos2Vec.read_corpus('./corpus_ex2')
# print('Done reading corpus')
# # brown_corpus = brown.words()
# m.train_model(corpus)
# m.save_model('./Pos2Vec_obj')
# m.model.save('./Pos2Vec_model')
# print('here')
