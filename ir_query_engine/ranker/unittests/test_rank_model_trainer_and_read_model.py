import unittest
import os
from mock import patch
from ir_query_engine.ranker.ranking import RankModelTrainer, read_model_from_file, LinearSVMRankModel, RBFSVMRankModel

__author__ = 'Deyang'


class RankModelTrainerTestCase(unittest.TestCase):

    def setUp(self):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_train_data_file_path = os.path.join(self.dir_path, '..', 'svm_rank', 'example', 'train.dat')
        self.test_model_file_path = os.path.join(self.dir_path, '..', 'svm_rank', 'example', 'model')

    def tearDown(self):
        os.remove(self.test_model_file_path)

    @patch('ir_query_engine.ranker.ranking.get_train_data_path')
    @patch('ir_query_engine.ranker.ranking.get_md_path')
    def test_train_linear_model(self, mock_get_md, mock_get_train_data):
        mock_get_md.return_value = self.test_model_file_path
        mock_get_train_data.return_value = self.test_train_data_file_path
        self.assertFalse(os.path.isfile(self.test_model_file_path))

        trainer = RankModelTrainer(RankModelTrainer.RANK_SVM_KERNEL_TYPE_LINEAR, c=3)
        trainer.train_model()

        self.assertTrue(os.path.isfile(self.test_model_file_path))
        rank_model = read_model_from_file()
        self.assertTrue(isinstance(rank_model, LinearSVMRankModel))

    @patch('ir_query_engine.ranker.ranking.get_train_data_path')
    @patch('ir_query_engine.ranker.ranking.get_md_path')
    def test_train_rbf_based_model(self, mock_get_md, mock_get_train_data):
        mock_get_md.return_value = self.test_model_file_path
        mock_get_train_data.return_value = self.test_train_data_file_path
        self.assertFalse(os.path.isfile(self.test_model_file_path))

        trainer = RankModelTrainer(RankModelTrainer.RANK_SVM_KERNEL_TYPE_RBF, c=3)
        trainer.train_model()

        self.assertTrue(os.path.isfile(self.test_model_file_path))
        rank_model = read_model_from_file()
        self.assertTrue(isinstance(rank_model, RBFSVMRankModel))


class ReadModelTestCase(unittest.TestCase):

    def setUp(self):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_model_file_path = os.path.join(self.dir_path, '..', 'svm_rank', 'example', 'model')

    def tearDown(self):
        os.remove(self.test_model_file_path)

    @patch('ir_query_engine.ranker.ranking.get_md_path')
    def test_read_linear_model(self, mock_get_md):
        mock_get_md.return_value = self.test_model_file_path
        with open(self.test_model_file_path, 'w') as f:
            contents = """SVM-light Version V6.20
0 # kernel type
3 # kernel parameter -d
1 # kernel parameter -g
1 # kernel parameter -s
1 # kernel parameter -r
# kernel parameter -u
6 # highest feature index
8 # number of training documents
2 # number of support vectors plus 1
0 # threshold b, each following line is a SV (starting with alpha*y)
1 1:1.521512 2:-0.057497051 3:-0.52151203 4:-0.17125149 5:0.96401501 #"""
            f.write(contents)

        rank_model = read_model_from_file()
        self.assertTrue(isinstance(rank_model, LinearSVMRankModel))
        self.assertEqual(rank_model.threshold, 0.0)
        self.assertEqual(rank_model.weight_vec, [1.521512, -0.057497051, -0.52151203, -0.17125149, 0.96401501])

    @patch('ir_query_engine.ranker.ranking.get_md_path')
    def test_read_rbf_model(self, mock_get_md):
        mock_get_md.return_value = self.test_model_file_path
        with open(self.test_model_file_path, 'w') as f:
            contents = """SVM-light Version V6.20
2 # kernel type
3 # kernel parameter -d
0.5 # kernel parameter -g
1 # kernel parameter -s
1 # kernel parameter -r
empty# kernel parameter -u
6 # highest feature index
9 # number of training documents
7 # number of support vectors plus 1
0 # threshold b, each following line is a SV (starting with alpha*y)
0.10001180341368905157839463981873 1:0 2:1 3:1 4:0.5 5:0 #
-0.10001180341368905157839463981873 1:1 2:0 3:0 4:0.40000001 5:1 #
-0.10001180341368905157839463981873 1:1 2:1 3:0 4:0.30000001 5:0 #
0.10001180341368905157839463981873 1:0 2:0 3:1 4:0.1 5:1 #
-0.30003541024106716861297172727063 1:0 2:1 3:1 4:0.5 5:0 #
0.30003541024106716861297172727063 1:1 2:0 3:0 4:0.40000001 5:1 #"""
            f.write(contents)

        rank_model = read_model_from_file()
        self.assertTrue(isinstance(rank_model, RBFSVMRankModel))
        self.assertEqual(rank_model.threshold, 0.0)
        self.assertAlmostEqual(rank_model.alphays,
                               [0.10001180341368905, -0.10001180341368905, -0.10001180341368905, 0.10001180341368905,
                                -0.30003541024106717, 0.30003541024106717])
        self.assertAlmostEqual(rank_model.svs,
                               [[0.0, 1.0, 1.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.40000001, 1.0],
                                [1.0, 1.0, 0.0, 0.30000001, 0.0], [0.0, 0.0, 1.0, 0.1, 1.0],
                                [0.0, 1.0, 1.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.40000001, 1.0]])
        self.assertEqual(rank_model.gamma, 0.5)

if __name__ == '__main__':
    unittest.main()
