from ir_query_engine.common import sample_by_index_generator, DataStore
from sklearn.cross_validation import KFold
import random

__author__ = 'Deyang'


class TestDataSet(object):

    def __init__(self, raw_data):
        # testing questions
        self.questions = []
        self.question_topic_words = []
        # the correct answers
        self.top_answers = []
        self.answer_topic_words = []
        self.relevant_answers = []
        # matched question from the model
        self.retrieved_questions = []
        # final answer of the matched qa pair
        self.retrieved_answers = []
        # the relevance between the retrieved answer and the top answer, measured by the similarity defined by the model
        self.relevance_scores = []
        # 1 or 0, indicating retrieved_answer == top_answer
        self.judgement_labels = []
        # percentage of correctly retrieved answer
        self.accuracy = 0.0
        # average of relevance
        self.avg_relevance_score = 0.0

        for segment in raw_data:
            self.questions.append(segment['_question'])
            self.question_topic_words.append(segment['_question_topic_words'])
            self.top_answers.append(segment['answer'])
            self.answer_topic_words.append(segment['answer_topic_words'])
            self.relevant_answers.append(segment['qa_pairs_with_matching_score'])


def split_raw_data_k_fold(raw_data, num_folds):
    """Return tuple of training data store, training data set for evaluation, and the unseen test data set."""
    assert isinstance(raw_data, list)
    # since our generated raw_data has an inherent order,
    # we must shuffle it before k-folds
    random.shuffle(raw_data)
    train_test_data_tuples = []
    kf = KFold(len(raw_data), n_folds=num_folds)
    for train_index, test_index in kf:
        train_data = []
        test_data = []

        for data_piece in sample_by_index_generator(train_index, raw_data):
            train_data.append(data_piece)
        for data_piece in sample_by_index_generator(test_index, raw_data):
            test_data.append(data_piece)

        train_test_data_tuples.append(
            (DataStore(train_data, load_rank_training_data=False), TestDataSet(train_data), TestDataSet(test_data))
        )

    return train_test_data_tuples
