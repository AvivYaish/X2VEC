from X2Vec import X2Vec
from pywsd import disambiguate


class Sense2Vec(X2Vec):
    """
    A word embedding model that generates different embeddings for identical words based on their meaning.
    """

    def tokenize_corpus(self, corpus, tokenize=True):
        """
        Method that tokenizes the corpus prior to training. For each word in the corpus we compute the sense of that
        word and change it with word_sense. For example: cat can become cat_n.01
        :param corpus: the corpus as a string.
        :param tokenize: True if should tokenize the corpus beforehand.
        :return: the tokenized corpus.
        """
        # convert the corpus to be sentence
        corpus = [' '.join(sentence) for sentence in corpus]
        if not tokenize:
            return corpus

        print('Starting to tag corpus')
        corpus_tags = []
        counter = 0.0
        for sentence in corpus:
            if (counter % 100000) == 0:
                print(counter/len(corpus)*100, " percent complete         \r",)
            try:
                # get the sense of each word in the sentence
                tagged_sentence = disambiguate(sentence)
                corpus_tags.append(tagged_sentence)
            except IndexError:
                print("pywsd can't handle the sentence: " + sentence)
            counter += 1
        # create a dictionary of each word and all the senses it was mapped to
        for sentence in corpus_tags:
            for tag in sentence:
                if tag[1] is None:
                    continue
                cur_set = self.token_dict.get(tag[0], set())
                cur_set.add(tag[1].name())
                self.token_dict[tag[0]] = cur_set
        # create the tagged corpus in a format ready for training
        tagged_corpus = [[word[1].name() for word in sentence if word[1] is not None] for sentence in corpus_tags]
        return tagged_corpus
