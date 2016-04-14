import glob
from qa_file_validator import validate_qa_file

__author__ = 'Deyang'

if __name__ == '__main__':
    for file in glob.glob("data/*.qa"):
        validate_qa_file(file)



