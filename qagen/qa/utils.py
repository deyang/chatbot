import re
from qagen.knowledge.entities import BaseEntity


def make_context_map(*args):
    for arg in args:
        if not isinstance(arg, BaseEntity):
            raise Exception('Arguements must be instances of entity')

    return {entity_instance.__class__.__name__: entity_instance.property_value_map['name']
            for entity_instance in args}


def tokenize_sentence(sentence):
    return filter(None, re.compile('[^A-Za-z0-9]').split(sentence))


def intersect_lists(list_a, list_b):
    return list(set(list_a) & set(list_b))

