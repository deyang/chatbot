import re


def make_context_map(*args):
    """Actually returns a list. Sorry"""
    from qagen.knowledge.entities import BaseEntity
    for arg in args:
        if not isinstance(arg, BaseEntity):
            raise Exception('Arguements must be instances of entity')

    return [{entity_instance.__class__.__name__: entity_instance.get_entity_id()} for entity_instance in args]


def tokenize_sentence(sentence):
    return filter(None, re.compile('[^A-Za-z0-9]').split(sentence))


def intersect_lists(list_a, list_b):
    return list(set(list_a) & set(list_b))


def construct_job_search_url(company_id='%', location_id='%', function_id='%', page_number=1):
    """
    for simplicity, all ids are str type
    """
    return 'http://portfoliojobs.a16z.com/careers_home.php?Company=%s&Function=%s&Location=%s&p=%d' \
           % (company_id, function_id, location_id, page_number)
