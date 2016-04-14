from optparse import OptionParser
import json
import os.path
import sys


__author__ = 'Deyang'


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

    with open(options.input_file, 'r') as f:
        lines = f.readlines()
        skipped_lines = [l for l in lines if not l.strip().startswith('#')]
        content = ''.join(skipped_lines)
        try:
            json.loads(content)
            print "Congrats! Validation passed"
        except:
            with open("dump.log", 'w') as o:
                o.write(content)
            print "Validation failed! Please copy the content in dump.log and paste it to http://www.jsoneditoronline.org. You will see what is broken"