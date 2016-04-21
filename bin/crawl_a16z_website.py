import sys
import json

from qagen.data_provider.knowledge_data_providers import WebCrawlerKnowledgeDataProvider
from qagen.knowledge.json_converter import EntityJsonConverter

# if len(sys.argv) != 2:
#     raise Exception('Insufficient or unexpected input. Example: python crawl_a16z_website.py /path/to/output/file')
#
# output_file_path = sys.argv[1]
output_file_path = 'data.json'

data_provider = WebCrawlerKnowledgeDataProvider()

entity_map = dict()
for entity_type in data_provider.get_all_entity_types():
    entity_type_name = entity_type.__name__
    entity_instances = data_provider.get_all_instances_of_type(entity_type)
    entity_map[entity_type_name] = [EntityJsonConverter.to_json_dict(entity_instance)
                                    for entity_instance in entity_instances]

serialized = json.dumps(entity_map, indent=2, sort_keys=True)

print 'Dumping data into JSON file...'
with open(output_file_path, 'w') as output_file:
    output_file.write(serialized)

print 'JSON data saved in ' + output_file_path
