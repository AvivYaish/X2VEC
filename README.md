# X2VEC
### Authors: Aviv Yaish, Dan Kufra

We have implemented, expanded and reviewed the paper “Sense2Vec - A Fast and Accurate Method For Word Sense Disambiguation In Neural Word Embeddings" by Andrew Trask, Phil Michalak and John Liu.

The paper can be found at:  
https://arxiv.org/abs/1511.06388

#### Installation

    1. Install libraries:
        string,
        time,
        textblob,
        pywsd,
        nltk (preferable to download all associated addons/corpora/etc')
        matplotlib
        sklearn.manifold
        numpy
    2. Download ex2's corpus, move it to the project's directory and rename it to corpus.
    
#### Usage
    The code is split up into multiple classes.
        X2Vec:
            Main class that other models inherit from. Implements many of the general functions our models need.
        Pos2Vec:
            A model that implements POS tagging as its tokenization method.
        Sense2Vec:
            A model that implements Word Sense Disambiguation as its tokenization method.
        Sentiment2Vec:
            A model that implements Sentiment Analysis as its tokenization method.
    In addition, we have provided the run_models.py file. This file can be used for easily training, saving
    and evaluating new models.

    To run it simply call "python run_models.py" in your shell.
    To change the model types you wish to train simply alter 'model_types_to_train'.
    To change the models and methods you wish to evaluate simply alter the evaluate if.

    For evaluation, we have provided 8 files of manually chosen words as an example for evaluation
    (in the evaluation_files directory). This is simply to give a feel for how we see the evaluation can be done when
    given a tagged corpus or label.
    If you wish to evaluate new words then simply create files in the same format and give them to the evaluation
    function instead.


#### Notes
    1. Sense2Vec training is SLOW! Very Slow.
    2. The evaluation is simply a proof of concept of how it can be done when given a proper labeled dataset.
    3. The TSNE visualization should be given a reasonable number (say, 1000-2000).
