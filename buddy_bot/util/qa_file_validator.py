from optparse import OptionParser
import json
import os.path
import sys


__author__ = 'Deyang'


def validate_qa_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        skipped_lines = [l for l in lines if not l.strip().startswith('#')]
        content = ''.join(skipped_lines)
        try:
            json.loads(content)
            print "Congrats! Validation of %s passed" % file_name
            return True
        except:
            with open("%s_dump.log" % file_name, 'w') as o:
                o.write(content)
            print "Validation of %s failed! Please copy the content in %s_dump.log and paste it to http://www.jsoneditoronline.org. You will see what is broken" % \
                  (file_name, file_name)
            return False

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-i', '--input_file', dest='input_file',
                      action='store',
                      default=None,
                      help='Input qa file to validate')

    (options, args) = parser.parse_args()

    if options.input_file is None:
        print "missing file path"
        sys.exit(1)
    if not os.path.isfile(options.input_file):
        print "cannot find the file"
        sys.exit(1)

    ret = validate_qa_file(options.input_file)
    if ret:
        sys.exit(0)
    else:
        sys.exit(1)
