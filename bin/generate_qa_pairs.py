import json
import random
from qagen.data_provider.knowledge_data_providers import *
from qagen.data_provider.qa_concept_providers import *
from qagen.qa.generators import DefaultQAPairGenerator
from qagen.training_label.training_label_generator import TrainingLabelGenerator

# data_provider = WebCrawlerKnowledgeDataProvider()
data_provider = JsonFileKnowledgeDataProvider('data.json')
concept_provider = DefaultQAConceptProvider(data_provider)

qa_generator = DefaultQAPairGenerator(data_provider)

qa_pairs = []
for qa_concept in concept_provider.get_all_qa_concepts():
    qa_pairs.extend(qa_generator.generate_qa_pairs_about_concept(qa_concept))

print '%d QA pairs collected.' % len(qa_pairs)


is_for_training = True

if is_for_training:
    output_path = 'qa_pairs_labeled.json'
    print 'Generating training label...'
    labeler = TrainingLabelGenerator(qa_pairs)
    output_qa_pairs = labeler.generate_qa_pairs_with_training_label()

else:
    output_path = 'qa_pairs_unlabeled.json'
    output_qa_pairs = qa_pairs

print 'Dumping QA pairs to JSON file...'
qa_pairs_json_data = [qa_pair.to_json_dict(is_for_training=is_for_training) for qa_pair in output_qa_pairs]
qa_json_str = json.dumps(qa_pairs_json_data, indent=2, sort_keys=True)
with open(output_path, 'w') as output_file:
    output_file.write(qa_json_str)
print 'All QA pairs dumped at ' + output_path

