import unittest
from mock import patch
import os
import glob
from ir_query_engine.rank_match_models.topic_word_lookup_feature.topic_word_lookup_model import TopicWordLookupModelStruct
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.common import DataStore
from gensim.models.tfidfmodel import df2idf

__author__ = 'Deyang'


class TopicWordModelTestCase(unittest.TestCase):

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
        self.test_md_file_path = os.path.join(self.dir_path, 'test_topic_words.set')

    def tearDown(self):
        os.remove(self.test_md_file_path)

    @patch('ir_query_engine.rank_match_models.topic_word_lookup_feature.topic_word_lookup_model.get_set_path')
    def test_get_topic_word_vec(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path

        tfidf_model_struct = TfIdfModelStruct.get_model(data_store=self.data_store, save=False, regen=True)
        model_struct = TopicWordLookupModelStruct.get_model(
            tfidf_model_struct,
            data_store=self.data_store)
        query_doc = "tell me about a16z"
        vec = model_struct.get_topic_word_vec(query_doc)
        for tokenid, value in vec:
            if value:
                self.assertEqual(tfidf_model_struct.dictionary.get(tokenid),
                                 'a16z')

        query_doc = "who is the founder"
        vec = model_struct.get_topic_word_vec(query_doc)
        for tokenid, value in vec:
            if value:
                self.assertEqual(tfidf_model_struct.dictionary.get(tokenid),
                                 'founder')


    @patch('ir_query_engine.rank_match_models.topic_word_lookup_feature.topic_word_lookup_model.get_set_path')
    def test_get_similarities(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        tfidf_model_struct = TfIdfModelStruct.get_model(data_store=self.data_store, save=False, regen=True)
        model_struct = TopicWordLookupModelStruct.get_model(
            tfidf_model_struct,
            data_store=self.data_store)

        query_doc = "Who are the managers of a16z?"
        compare_docs = [pair[0] for pair in self.data_store.topic_word_docs]
        results = model_struct.get_similarities(query_doc, compare_docs)

        results.sort(key=lambda p: p[1], reverse=True)
        ranked_docs = [t[0] for t in results]
        self.assertEqual(ranked_docs, [0, 3, 2, 1, 5, 4])

    @patch('ir_query_engine.rank_match_models.topic_word_lookup_feature.topic_word_lookup_model.get_set_path')
    def test_get_model(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        self.assertFalse(os.path.isfile(self.test_md_file_path))

        tfidf_model_struct = TfIdfModelStruct.get_model(data_store=self.data_store, save=False, regen=True)
        new_model_struct = TopicWordLookupModelStruct.get_model(
            tfidf_model_struct,
            data_store=self.data_store)
        # assert file saves
        self.assertTrue(os.path.isfile(self.test_md_file_path))

        # loading model
        loaded_model_struct = TopicWordLookupModelStruct.get_model(tfidf_model_struct)
        self.assertEqual(loaded_model_struct.topic_word_set,
                         new_model_struct.topic_word_set)


if __name__ == '__main__':
    unittest.main()
