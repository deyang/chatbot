import unittest
from ir_query_engine.rank_match_models.topic_word_feature.training import DocWordAnalyzer, POS_TAG_LABELS

__author__ = 'Deyang'


class DocWordAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        self.test_doc = """
The first Republican to serve as governor of the state, he became known as the father of the Republican Party in Kentucky.
After a well-received speech seconding the presidential nomination of Ulysses S. Grant at the 1880 Republican National Convention, he was nominated for governor in 1887.
He lost the general election that year, but won in 1895, capitalizing on divisions in the Democratic Party over the issue of free silver.
His term was marked by political struggles and violence?
He advanced the status of black citizens, but was unable to enact much of his reform agenda over a hostile Democratic majority.
He was elected by the state legislature to the U.S. Senate in 1907, when voting was deadlocked and the Democratic candidate, outgoing Governor J. C. W. Beckham, refused to withdraw in favor of a compromise candidate.
Bradley's opposition to Prohibition made him palatable to some Democratic legislators, and after two months of balloting, four of them crossed party lines to elect him. His career in the Senate was largely undistinguished.
"""

    def test_analyze(self):
        a = DocWordAnalyzer()

        self.assertEqual(a.analyze(self.test_doc, "republican"),
                         (2, 1, 0, 1, 1, 0, POS_TAG_LABELS['NNP']))
        self.assertEqual(a.analyze(self.test_doc, "party"),
                         (3, 1, 0, 1, 1, 0, POS_TAG_LABELS['NNP']))
        self.assertEqual(a.analyze(self.test_doc, "his"),
                         (3, 0, 1, 0, 0, 0, POS_TAG_LABELS['PRP$']))
        self.assertEqual(a.analyze(self.test_doc, "Derek"),
                         (0, 0, 0, 0, 0, 0, 0))

    def test_cache(self):
        a = DocWordAnalyzer()
        a.analyze(self.test_doc, "party")
        self.assertEqual(len(a.cache), 1)
        self.assertTrue(self.test_doc in a.cache)


if __name__ == '__main__':
    unittest.main()
