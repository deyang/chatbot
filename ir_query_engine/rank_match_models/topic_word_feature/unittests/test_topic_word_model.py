import unittest
from mock import patch
import os
import glob
from ir_query_engine.rank_match_models.topic_word_feature.topic_word_model import TopicWordModelStruct, POS_TAG_LABELS
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.common import DataStore
from gensim.models.tfidfmodel import df2idf

__author__ = 'Deyang'


class TopicWordModelTestCase(unittest.TestCase):

    def setUp(self):
        topic_word_docs = [
            ("tell me about a16z", ["a16z"]),
            ("a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
             ["a16z", "venture", "capital"]),
            ("who is the founder of a16z", ["founder", "a16z"]),
            ("Marc Andreessen and Ben Horowitz co-founded a16z.", ["co-founded", "a16z"]),
            ("who is Steven Sinofsky", ["Steven", "Sinofsky"]),
            ("Steven Sinofsky is a board partner at a16z.", ["Steven", "Sinofsky", "board", "partner", "a16z"]),
        ]
        self.data_store = DataStore({})
        self.data_store.topic_word_docs = topic_word_docs
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_md_file_path = os.path.join(self.dir_path, 'test_lr.pkl')

    def tearDown(self):
        for f in glob.glob(os.path.join(self.dir_path, "test_lr.*")):
            os.remove(f)

    @patch('ir_query_engine.rank_match_models.topic_word_feature.topic_word_model.get_md_path')
    def test_extract_feature(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        model_struct = TopicWordModelStruct.get_model(data_store=self.data_store)
        self.assertEqual(model_struct._extract_feature(("tell me about a16z", "a16z")),
                         (1, df2idf(5, 6), 1, 1, 1, 0, 0, 0, POS_TAG_LABELS['NN']))
        self.assertEqual(model_struct._extract_feature(("tell me about a16z", "me")),
                         (1, df2idf(1, 6), 1, 1, 1, 0, 0, 0, POS_TAG_LABELS['PRP']))
        self.assertEqual(model_struct._extract_feature(("tell me about a16z", "Yo")),
                         (0, 10.0, 0, 0, 0, 0, 0, 0, 0))

    @patch('ir_query_engine.rank_match_models.topic_word_feature.topic_word_model.get_md_path')
    def test_predict(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        model_struct = TopicWordModelStruct.get_model(data_store=self.data_store)
        query_doc = "Who are the managers of a16z?"

        prob_vec = model_struct._predict_doc_as_vec(query_doc)
        self.assertEqual(len(prob_vec), 5)

        # validation
        for termid, prob in prob_vec:
            term = model_struct.tfidf_model_struct.dictionary.get(termid)
            feature_vec = model_struct._extract_feature((query_doc, term))
            single_predicted_prob = model_struct.predict_one_word(feature_vec)
            self.assertAlmostEqual(prob, single_predicted_prob)

    @patch('ir_query_engine.rank_match_models.topic_word_feature.topic_word_model.get_md_path')
    def test_get_similarities(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        model_struct = TopicWordModelStruct.get_model(data_store=self.data_store)

        query_doc = "Who are the managers of a16z?"
        compare_docs = [pair[0] for pair in self.data_store.topic_word_docs]
        results = model_struct.get_similarities(query_doc, compare_docs)
        results.sort(key=lambda p: p[1], reverse=True)

        ranked_docs = [t[0] for t in results]
        self.assertEqual(ranked_docs, [2, 0, 3, 1, 5, 4])

    @patch('ir_query_engine.rank_match_models.topic_word_feature.topic_word_model.get_md_path')
    def test_get_model(self, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path

        new_model_struct = TopicWordModelStruct.get_model(data_store=self.data_store)
        # assert file saves
        self.assertTrue(os.path.isfile(self.test_md_file_path))

        # loading model
        local_data_store = DataStore({})
        local_data_store.doc_set = [pair[0] for pair in self.data_store.topic_word_docs]
        tfidf_model_struct = TfIdfModelStruct.get_model(data_store=local_data_store, save=False, regen=True)
        loaded_model_struct = TopicWordModelStruct.get_model(tfidf_model_struct=tfidf_model_struct)

        # assert model identity
        vec = (0, 10.0, 0, 0, 0, 0, 0, 0, 0)
        self.assertAlmostEqual(loaded_model_struct.predict_one_word(vec),
                               new_model_struct.predict_one_word(vec))

        # regen
        self.data_store.topic_word_docs = self.data_store.topic_word_docs[1:]
        regen_model_struct = TopicWordModelStruct.get_model(data_store=self.data_store, regen=True)
        self.assertNotAlmostEqual(loaded_model_struct.predict_one_word(vec),
                                  regen_model_struct.predict_one_word(vec))

if __name__ == '__main__':
    unittest.main()
