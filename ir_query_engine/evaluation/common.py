from ir_query_engine.common import sample_by_index_generator, DataStore
from sklearn.cross_validation import KFold

__author__ = 'Deyang'


class TestDataSet(object):

    def __init__(self, raw_data):
        self.questions = []
        self.top_answers = []
        self.relevant_answers = []
        self.relevance_scores = []
        self.retrieved_results = []
        self.judgement_labels = []
        self.relevance_scores = []
        self.accuracy = 0.0
        self.avg_relevance_score = 0.0

        for segment in raw_data:
            self.questions.append(segment['_question'])
            self.top_answers.append(segment['answer'])
            self.relevant_answers.append(segment['qa_pairs_with_matching_score'])


def split_raw_data_k_fold(raw_data, num_folds):
    """Return tuple of training data store, training data set for evaluation, and the unseen test data set."""
    assert isinstance(raw_data, list)
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
            (DataStore(train_data), TestDataSet(train_data), TestDataSet(test_data))
        )

    return train_test_data_tuples
