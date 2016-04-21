# chatbot
[![Build Status](https://travis-ci.org/deyang/chatbot.svg?branch=master)](https://travis-ci.org/deyang/chatbot)

## Generate training data

* First crawl data from a16z.com (optional since there's already a data.json copy checked in)
```
cd /path/to/project/root
PYTHONPATH=`pwd` python bin/crawl_a16z_website.py
```

* Then run the other script to generate training data
```
PYTHONPATH=`pwd` python bin/generate_qa_pairs.py
```
