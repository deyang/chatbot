import json
import random
from optparse import OptionParser
import os
from qagen.data_provider.knowledge_data_providers import *
from qagen.data_provider.qa_concept_providers import *
from qagen.qa.generators import DefaultQAPairGenerator
from qagen.training_label.training_label_generator import TrainingLabelGenerator


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('', '--max_companies', dest='max_companies',
                      action='store',
                      default=None,
                      help='Maximum companies when trimming down companies. Default: None, no trimming')
    parser.add_option('', '--max_jobs', dest='max_jobs',
                      action='store',
                      default=3,
                      help='Maximum jobs per company when trimming down jobs. Default: 3')
    parser.add_option('', '--disable_label', dest='disable_label',
                      action='store_true',
                      default=False,
                      help='Disable labeled data i.e. not for training')

    (options, args) = parser.parse_args()

    # data_provider = WebCrawlerKnowledgeDataProvider()
    data_provider = JsonFileKnowledgeDataProvider(os.path.join(os.path.dirname(__file__), 'data.json'))

    if options.max_companies:
        max_companies = int(options.max_companies)
        data_provider.trim_down_companies(max_company=max_companies)
    if options.max_jobs:
        max_jobs = int(options.max_jobs)
        data_provider.trim_down_jobs(max_job_per_company=max_jobs)

    concept_provider = DefaultQAConceptProvider(data_provider)
    qa_generator = DefaultQAPairGenerator(data_provider)

    qa_pairs = []
    for qa_concept in concept_provider.get_all_qa_concepts():
        qa_pairs.extend(qa_generator.generate_qa_pairs_about_concept(qa_concept))

    print '%d QA pairs collected.' % len(qa_pairs)

    is_for_training = not options.disable_label

    if is_for_training:
        output_path = os.path.join(os.path.dirname(__file__), 'qa_pairs_labeled.json')
        print 'Generating training label...'
        labeler = TrainingLabelGenerator(qa_pairs, concept_provider)
        output_qa_pairs = labeler.generate_qa_pairs_with_training_label()

    else:
        output_path = os.path.join(os.path.dirname(__file__), 'qa_pairs_unlabeled.json')
        output_qa_pairs = qa_pairs

    print 'Dumping QA pairs to JSON file...'
    qa_pairs_json_data = [qa_pair.to_json_dict(is_for_training=is_for_training) for qa_pair in output_qa_pairs]
    qa_json_str = json.dumps(qa_pairs_json_data, indent=2, sort_keys=True)
    with open(output_path, 'w') as output_file:
        output_file.write(qa_json_str)
    print 'All QA pairs dumped at ' + output_path
