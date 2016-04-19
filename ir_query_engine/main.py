from optparse import OptionParser

from common import load_data
import retrieve_match_models.tf_idf_feature.transform as tf_idf_transform

__author__ = 'Deyang'


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--data_file', dest='data_file',
                      action='store',
                      default=None,
                      help='Input data file')
    parser.add_option('', '--load_tf_idf', dest='load_tf_idf',
                      action='store_true',
                      default=False,
                      help='Load TF-IDF model only')
    parser.add_option('-r', '--regen', dest='regen',
                      action='store_true',
                      default=False,
                      help='Force to re gen/train models')

    (options, args) = parser.parse_args()

    if options.data_file:
        data_store = load_data(options.data_file)

    if options.load_tf_idf:
        tf_idf_model_struct = tf_idf_transform.get_model(data_store=data_store, regen=options.regen)
        results = tf_idf_model_struct.query(raw_doc="Mark Zuckerberg established Facebook")
        print results
        print data_store.doc_set[results[0][0]]
        print data_store.doc_set[results[1][0]]
