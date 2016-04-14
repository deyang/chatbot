import glob
import sys
from qa_file_validator import validate_qa_file

__author__ = 'Deyang'

if __name__ == '__main__':
    ret = True
    for file in glob.glob("data/*.qa"):
        ret &= validate_qa_file(file)
    if ret:
        sys.exit(0)
    else:
        sys.exit(1)



