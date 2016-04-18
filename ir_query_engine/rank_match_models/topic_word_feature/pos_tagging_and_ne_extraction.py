import nltk

__author__ = 'Deyang'


nltk.download('maxent_ne_chunker')
nltk.download('tagsets')

# part of speech
doc_ex = """
Ten year's working. The first Republican to serve as governor of the state, he became known as the father of the Republican Party in Kentucky.
After a well-received speech seconding the presidential nomination of Ulysses S. Grant at the 1880 Republican National Convention, he was nominated for governor in 1887.
He lost the general election that year, but won in 1895, capitalizing on divisions in the Democratic Party over the issue of free silver.
His term was marked by political struggles and violence?
He advanced the status of black citizens, but was unable to enact much of his reform agenda over a hostile Democratic majority.
He was elected by the state legislature to the U.S. Senate in 1907, when voting was deadlocked and the Democratic candidate, outgoing Governor J. C. W. Beckham, refused to withdraw in favor of a compromise candidate.
Bradley's opposition to Prohibition made him palatable to some Democratic legislators, and after two months of balloting, four of them crossed party lines to elect him. His career in the Senate was largely undistinguished.
"""

sentences = nltk.sent_tokenize(doc_ex)

tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
print tagged_sentences


tagdict = nltk.data.load("help/tagsets/upenn_tagset.pickle")
POS_TAG_LABELS = dict()
for i, tag in enumerate(sorted(tagdict.keys())):
    POS_TAG_LABELS[tag] = i

print POS_TAG_LABELS



chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)



def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(t)
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

entity_names = []
for tree in chunked_sentences:
    # Print results per sentence
    # print extract_entity_names(tree)
    print tree
    entity_names.extend(extract_entity_names(tree))

# Print all entity names
print entity_names