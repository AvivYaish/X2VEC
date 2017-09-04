from X2Vec import X2Vec
from Sense2Vec import Sense2Vec
from Pos2Vec import Pos2Vec
from Sentiment2Vec import Sentiment2Vec
import numpy as np
import os


def remove_unknown_words(known_words, eval_arrays, question_answer):
    """
    Function that removes any words our model was not trained on from the evaluation files
    :param known_words: The words known by the model
    :param eval_arrays: The words we wish to evaluate on
    :param question_answer: Whether the evaluation is a question answer evaluation
    :return:
    A filtered version of the original eval_arrays
    """
    mask = np.ones(eval_arrays.shape[0])
    if question_answer:
        tag_num = 4
    else:
        tag_num = 2
    for i in np.arange(tag_num):
        mask = np.logical_and(mask, np.in1d(eval_arrays[:, i], np.array(list(known_words))))
    filtered_evals = eval_arrays[mask]
    return filtered_evals


def create_filtered_files(model, corpus_path, evaluation_path, question_answer=True):
    """
    Creates a filtered version of the evaluation file that only includes words in our corpus
    :param model: Trained model
    :param corpus_path: Path to the corpus
    :param evaluation_path: Path to evaluation file
    :param question_answer: Whether the evaluation is question_answer
    :return:
    Path to filtered version of the evaluation file
    """
    with open(evaluation_path, encoding='latin-1') as f_eval_file:
        # read the file and split into np.array of arrays
        eval_arrays = np.char.lower(np.array([sentence.split() for sentence in f_eval_file.readlines()]))

        # if the model has known_words then use them, otherwise read the corpus and tokenize to create known words
        if model.tokenized:
            known_words = model.known_words
        else:
            corpus = X2Vec.read_corpus(corpus_path)
            model.tokenize_corpus(corpus, tokenize=False)
            known_words = model.known_words
        # get only the lines for evaluation that our model trained on
        filtered_evals = remove_unknown_words(known_words, eval_arrays, question_answer)

        # write new evaluation file to working directory
        # TODO change to better file name
        f_out_name = evaluation_path[:] + '_filtered'
        with open(f_out_name, 'w+') as f_out_name:
            for l in filtered_evals:
                if question_answer:
                    f_out_name.write('%s %s %s %s\n' % (l[0], l[1], l[2], l[3]))
                else:
                    f_out_name.write('%s %s %s\n' % (l[0], l[1], l[2]))
        return f_out_name


def evaluate_similarity(model, sim_path, threshold):
    """
    Function that evaluates the similarity between pairs of words and computes the accuracy
    :param model: Model we wish to evaluate
    :param sim_path: Path to file of similarities
    :param threshold: Threshold of similarity above which we are "correct"
    :return: Model's accuracy
    """
    with open(sim_path, 'r') as f_sim:
        with open(sim_path[:-3] + 'res.txt', 'w') as f_res:
            f_res.write('Result\tword1\tword2\tsim_score\n')
            # read the lines of word pairs
            lines = f_sim.readlines()
            correct = 0
            for l in lines:
                l = l.split()
                # try except for case where the word was not in the model's training corpus
                try:
                    model.wv.vocab[l[0]]
                    model.wv.vocab[l[1]]
                except KeyError:
                    continue
                # compute the similarity score between the two words
                # if the score is above the threshold then add to our correct counter
                sim_scoring = model.wv.similarity(l[0], l[1])
                if sim_scoring > threshold:
                    f_res.write('Correct\t%s\t%s\t%f\n' % (l[0], l[1], sim_scoring))
                    correct += 1
                else:
                    f_res.write('Incorrect\t%s\t%s\t%f\n' % (l[0], l[1], sim_scoring))
            # return the accuracy of our model
            return float(correct)/len(lines)


def evaluate_question_answer(model, qa_path):
    """
    Function that evaluates the question answer skill of our model and computes the accuracy
    :param model: Model we wish to evaluate
    :param qa_path: Path to file of question answer
    :return: Model's accuracy
    """
    with open(qa_path, 'r') as f_sim:
        with open(qa_path[:-3] + 'res.txt', 'w') as f_res:
            # read the lines of question answers
            lines = f_sim.readlines()
            correct = 0
            for l in lines:
                l = l.split()
                # try except for case where the word was not in the model's training corpus
                try:
                    model.wv.vocab[l[0]]
                    model.wv.vocab[l[1]]
                    model.wv.vocab[l[2]]
                    model.wv.vocab[l[3]]
                except KeyError:
                    continue
                # compute the most similar words to our question
                sim_scoring = np.array(model.wv.most_similar(positive=[l[0], l[1]], negative=[l[2]], topn=20))
                # check whether our answer is in the most similar
                if np.in1d(l[3], sim_scoring[:, 0]):
                    to_write = 'Correct\t%s\t%s\t%s\t%s\n' % (l[0], l[1], l[2], l[3])
                    correct += 1
                else:
                    to_write = 'Incorrect\t%s\t%s\t%s\t%s\n' % (l[0], l[1], l[2], l[3])
                f_res.write(to_write)
            # return the accuracy of our model
            return float(correct)/len(lines)


def train_model(model_type, path_suffix, overwrite=False, num_lines=-1):
    """
    Function that trains our models and saves them.
    :param model_type: Can be either "Pos2Vec", "Sense2Vec", "Sentiment2Vec", "Word2Vec"
    :param path_suffix: Suffix of our file name
    :param overwrite: Whether to overwrite in case the new file name exists
    :return:
    """
    print("Training a new model for model type: %s" % model_type)
    # create new file name for the trained model
    obj_path = model_type + '_obj_' + path_suffix
    # check whether the objects of that file name exist
    obj_exists = os.path.exists(obj_path)
    exit_flag = False
    if not overwrite:
        if obj_exists:
            exit_flag = True
            print('File %s exists: Please give a new suffix for the model or set overwrite to true' % obj_path)
    if exit_flag:
        return
    # create X2Vec objects based on model type and read the corpus accordingly
    if model_type == 'Sense2Vec':
        m = Sense2Vec()
        print('Reading corpus')
        corpus = Sense2Vec.read_corpus('./corpus', num_lines)
        m.corpus_path = './corpus'
        print('Done reading corpus')
        m.train_model(corpus, tokenize=True, train=True)
    elif model_type == 'Pos2Vec':
        m = Pos2Vec()
        print('Reading corpus')
        corpus = Pos2Vec.read_corpus('./corpus', num_lines)
        m.corpus_path = './corpus'
        print('Done reading corpus')
        m.train_model(corpus, tokenize=True, train=True)
    elif model_type == 'Sentiment2Vec':
        m = Sentiment2Vec()
        print('Reading corpus')
        corpus = Sentiment2Vec.read_corpus('./corpus', num_lines)
        m.corpus_path = './corpus'
        print('Done reading corpus')
        m.train_model(corpus, tokenize=True, train=True)
    elif model_type == 'Word2Vec':
        m = X2Vec()
        print('Reading corpus')
        corpus = X2Vec.read_corpus('./corpus', num_lines)
        m.corpus_path = './corpus'
        print('Done reading corpus')
        m.train_model(corpus, tokenize=False, train=True)
    else:
        print('Model type is not a valid type, please enter one of the following:'
              ' Word2Vec, Pos2Vec, Sense2Vec, Sentiment2Vec\n')
        return
    # save the model
    print('Saving model')
    m.save_model(obj_path)


def evaluate_model(model_type, model_path, visualize_embedding=0, question_answer_path=None,
                   similarity_path=None, filter=False):
    """
    Function that evaluates a given model.
    :param model_type: Can be either "Pos2Vec", "Sense2Vec", "Sentiment2Vec", "Word2Vec"
    :param model_path: Path to our model
    :param visualize_embedding: Whether we wish to run TSNE embedding, and how many examples to use
    :param question_answer_path: Path to the question_answer evaluation file
    :param similarity_path: Path to the similarity evaluation file
    :param filter: Whether to filter those files
    """
    print('Evaluating model type %s from path %s' % (model_type, model_path))
    # Load the model according to the model type
    if model_type == 'Sense2Vec':
        m = Pos2Vec.load_model(model_path)
        threshold = 0.4  # TODO change this accordingly
    elif model_type == 'Pos2Vec':
        m = Pos2Vec.load_model(model_path)
        threshold = 0.4
    elif model_type == 'Sentiment2Vec':
        m = Sentiment2Vec.load_model(model_path)
        threshold = 0.4  # TODO change this accordingly
    elif model_type == 'Word2Vec':
        m = X2Vec.load_model(model_path)
        threshold = 0.4
    else:
        print('Model type is not a valid type, please enter one of the following: '
              'Word2Vec, Pos2Vec, Sense2Vec, Sentiment2Vec\n')
        return

    # if visualize_embedding was chosen, run the TSNE visualization on the amount of examples chosen
    if visualize_embedding:
        m.visualize_embedding(visualize_embedding)
    # If a question_answer file was given, evaluate our model's performance on it
    if question_answer_path is not None:
        if filter:
            question_answer_path = create_filtered_files(model=m, corpus_path=m.corpus_path,
                                                         evaluation_path=question_answer_path, question_answer=True)
        qa_accuracy = evaluate_question_answer(m.model, question_answer_path)
        print('Accuracy of the model in question answer tests is: %f' % qa_accuracy)
    # If a similarity file was given, evaluate our model's performance on it
    if similarity_path is not None:
        if filter:
            similarity_path = create_filtered_files(model=m.model, corpus_path=m.corpus_path,
                                                    evaluation_path=similarity_path, question_answer=False)
        sim_accuracy = evaluate_similarity(m.model, similarity_path, threshold=threshold)
        print('Accuracy of the model in similarity tests is: %f' % sim_accuracy)


model_types_to_train = ['Pos2Vec', 'Sense2Vec', 'Sentiment2Vec', 'Word2Vec']
train_flag = False
evaluate_flag = True

if train_flag:
    # train new models
    print('Training models')
    for model in model_types_to_train:
        train_model(model_type=model, path_suffix="New_train", overwrite=True)

if evaluate_flag:
    print('Evaluating models')
    # evaluate the models previously trained
    evaluate_model(model_type='Pos2Vec', model_path='./Pos2Vec_obj',
                   similarity_path='./evaluation_files/Pos2Vec_Sim.txt',
                   question_answer_path='./evaluation_files/Pos2Vec_QA.txt')
    evaluate_model(model_type='Word2Vec', model_path='./Word2Vec_obj',
                   similarity_path='./evaluation_files/Word2Vec_Sim.txt',
                   question_answer_path='./evaluation_files/Word2Vec_QA.txt')
    evaluate_model(model_type='Sense2Vec', model_path='./Sense2Vec_obj',
                   similarity_path='./evaluation_files/Sense2Vec_Sim.txt',
                   question_answer_path='./evaluation_files/Sense2Vec_QA.txt')
    evaluate_model(model_type='Sentiment2Vec', model_path='./Sentiment2Vec_obj',
                   similarity_path='./evaluation_files/Sentiment2Vec_Sim.txt',
                   question_answer_path='./evaluation_files/Sentiment2Vec_QA.txt')
