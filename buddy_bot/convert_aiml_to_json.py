from optparse import OptionParser
import re
import html2text
import json
import pprint

__author__ = 'Deyang'


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-i', '--input_file', dest='input_file',
                      action='store',
                      default=None,
                      help='Input AIML file')
    parser.add_option('-o', '--out_file', dest='out_file',
                      action='store',
                      default=None,
                      help='Output file of conversation in json')

    (options, args) = parser.parse_args()

    with open(options.input_file, 'r') as f:
        content = f.read()
    print content
    results = re.findall("<category><pattern>(.*?)</pattern>\s?<template>(.*?)</template>", content)

    print results
    data = []
    for pair in results:
        question = pair[0].lower().strip("\n  _*")
        answer = html2text.html2text(pair[1]).strip("\n  _*")
        data.append([question, answer])

    with open(options.out_file, 'w') as o:
        o.write(json.dumps(data, indent=4))
