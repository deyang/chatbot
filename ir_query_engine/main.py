from optparse import OptionParser

from common import load_data

__author__ = 'Deyang'


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--data_file', dest='data_file',
                      action='store',
                      default=None,
                      help='Input data file')
    parser.add_option('', '--load_tf_idf', dest='load_tf_idf',
                      action='store_true',
                      default=None,
                      help='Load TF-IDF model only')
    parser.add_option('-r', '--reload', dest='reload',
                      action='store_true',
                      default=None,
                      help='Force to reload/train models')

    (options, args) = parser.parse_args()

    if options.data_file:
        data_store = load_data(options.data_file)
        print data_store