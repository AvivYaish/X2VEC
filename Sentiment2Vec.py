from X2Vec import X2Vec
from textblob import TextBlob


class Sentiment2Vec(X2Vec):
    """
    A word embedding model that generates different embeddings for identical words based on their meaning.
    """

    # Tokens for the various sentiments
    POSITIVE_SENTIMENT = "POSITIVE"
    NEUTRAL_SENTIMENT = "NEUTRAL"
    NEGATIVE_SENTIMENT = "NEGATIVE"

    # Margin around 0 polarity to be considered as a neutral sentiment
    NEUTRAL_MARGIN = 0.15

    @staticmethod
    def _get_sentence_tokens(sentence):
        """
        :param sentence: a string.
        :return: a list with a token for each word in the string.
        """
        sentiment = Sentiment2Vec.NEUTRAL_SENTIMENT
        polarity = TextBlob(sentence).sentiment.polarity
        if polarity >= Sentiment2Vec.NEUTRAL_MARGIN:
            sentiment = Sentiment2Vec.POSITIVE_SENTIMENT
        elif polarity <= -Sentiment2Vec.NEUTRAL_MARGIN:
            sentiment = Sentiment2Vec.NEGATIVE_SENTIMENT
        return [(word, sentiment) for word in sentence.split()]

    def tokenize_corpus(self, corpus, tokenize=True):
        """
        Method that tokenizes the corpus prior to training. For each word in the corpus we compute the Sentiment of that
        word and change it with word_Sentiment. For example: cat can become cat_POSITIVE/cat_NEGATIVE/cat_NEUTRAL
        :param corpus: the corpus as a string.
        :param tokenize: True if should tokenize the corpus beforehand.
        :return: the tokenized corpus.
        """
        # join the sentence based corpus
        corpus = [' '.join(sentence) for sentence in corpus]
        if not tokenize:
            return corpus

        print('Starting to tag corpus')
        corpus_tags = [Sentiment2Vec._get_sentence_tokens(sentence) for sentence in corpus]
        for sentence in corpus_tags:
            for tag in sentence:
                if tag[1] is None:
                    continue
                cur_set = self.token_dict.get(tag[0], set())
                cur_set.add(tag[1])
                self.token_dict[tag[0]] = cur_set

        tagged_corpus = [[word[0] + "_" + word[1] for word in sentence if word[1] is not None]
                         for sentence in corpus_tags]
        return tagged_corpus
