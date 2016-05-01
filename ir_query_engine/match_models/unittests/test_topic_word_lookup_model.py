import unittest

from mock import patch
import os
from ir_query_engine.match_models.topic_word_lookup_model import TopicWordLookupModelStruct
from ir_query_engine.common import DataStore

__author__ = 'Deyang'


class TopicWordLookupModelTestCase(unittest.TestCase):

    def setUp(self):
        self.data = [
            {
                "_question": "tell me about a16z",
                "_question_topic_words": ["a16z"],
                "answer": "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
                "answer_topic_words": ["a16z", "venture", "capital"],

            },
            {
                "_question": "who is the founder of a16z",
                "_question_topic_words": ["founder", "a16z"],
                "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
                "answer_topic_words": ["co-founded", "a16z"],
            },
            {
                "_question": "who is Steven Sinofsky",
                "_question_topic_words": ["Steven", "Sinofsky"],
                "answer": "Steven Sinofsky is a board partner at a16z.",
                "answer_topic_words": ["Steven", "Sinofsky", "board", "partner", "a16z"],
            },
        ]
        self.data_store = DataStore(self.data)
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_simmx_file_path = os.path.join(self.dir_path, 'test_topic_words.simmx')
        self.test_dict_file_path = os.path.join(self.dir_path, 'test_topic_words.dict')

    def tearDown(self):
        os.remove(self.test_simmx_file_path)
        os.remove(self.test_dict_file_path)

    @patch('ir_query_engine.match_models.topic_word_lookup_model.get_dict_path')
    @patch('ir_query_engine.match_models.topic_word_lookup_model.get_simmx_path')
    def test_get_topic_word_vec(self, mock_simmx_file_path, mock_dict_file_path):
        mock_simmx_file_path.return_value = self.test_simmx_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path

        model_struct = TopicWordLookupModelStruct.get_model(data_store=self.data_store)
        query_doc = "tell me about a16z. what is A16Z"
        vec = model_struct.get_topic_word_vec(query_doc)
        self.assertEqual(vec, [(0, 2)])
        self.assertEqual(model_struct.dictionary.get(vec[0][0]),
                         'a16z')

        query_doc = "who is the founder that co-founded a16z"
        vec = model_struct.get_topic_word_vec(query_doc)
        self.assertEqual(vec, [(0, 1), (3, 1), (4, 1)])
        self.assertEqual(model_struct.dictionary.get(vec[0][0]),
                         'a16z')
        self.assertEqual(model_struct.dictionary.get(vec[1][0]),
                         'founder')
        self.assertEqual(model_struct.dictionary.get(vec[2][0]),
                         'co-found')

    @patch('ir_query_engine.match_models.topic_word_lookup_model.get_dict_path')
    @patch('ir_query_engine.match_models.topic_word_lookup_model.get_simmx_path')
    def test_get_similarities_and_query(self, mock_simmx_file_path, mock_dict_file_path):
        mock_simmx_file_path.return_value = self.test_simmx_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        model_struct = TopicWordLookupModelStruct.get_model(data_store=self.data_store)

        query_doc = "Who are the founders of a16z?"
        compare_docs = [pair[0] for pair in self.data_store.topic_word_docs]
        results = model_struct.get_similarities(query_doc, compare_docs)

        results.sort(key=lambda p: p[1], reverse=True)
        self.assertEqual(self.data_store.topic_word_docs[results[0][0]][0],
                         "who is the founder of a16z")

        results = model_struct.query(raw_doc=query_doc)
        self.assertEqual(self.data_store.doc_set[results[0][0]],
                         "who is the founder of a16z")

    @patch('ir_query_engine.match_models.topic_word_lookup_model.get_dict_path')
    @patch('ir_query_engine.match_models.topic_word_lookup_model.get_simmx_path')
    def test_get_model(self, mock_simmx_file_path, mock_dict_file_path):
        mock_simmx_file_path.return_value = self.test_simmx_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        self.assertFalse(os.path.isfile(self.test_dict_file_path))
        self.assertFalse(os.path.isfile(self.test_simmx_file_path))

        new_model_struct = TopicWordLookupModelStruct.get_model(
            data_store=self.data_store)
        # assert file saves
        self.assertTrue(os.path.isfile(self.test_simmx_file_path))
        self.assertTrue(os.path.isfile(self.test_dict_file_path))

        # loading model
        loaded_model_struct = TopicWordLookupModelStruct.get_model()
        self.assertEqual(loaded_model_struct.dictionary,
                         new_model_struct.dictionary)


if __name__ == '__main__':
    unittest.main()
