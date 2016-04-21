import urllib2
import json
import re
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from bs4 import BeautifulSoup
from urlparse import urlparse, parse_qs

from qagen.knowledge.entities import *
from qagen.knowledge.json_converter import EntityJsonConverter, ENTITY_JSON_ID, ENTITY_JSON_RELATIONS,\
    ENTITY_JSON_RELATION_REF_ENTITY_TYPE, ENTITY_JSON_RELATION_REF_ENTITY_ID, ENTITY_JSON_RELATION_REF_ENTITY_IDS


class KnowledgeDataProvider(object):

    def __init__(self):
        self.entity_map = {
            A16Z: [],
            Company: [],
            Investor: [],
            Job: [],
        }

    def add_entity(self, entity_instance):
        self.entity_map[entity_instance.__class__].append(entity_instance)

    def get_all_entity_types(self):
        return self.entity_map.keys()

    def get_all_instances_of_type(self, entity_class):
        return self.entity_map[entity_class]

    def get_entity_instance(self, entity_class, entity_id):
        for entity_instance in self.get_all_instances_of_type(entity_class):
            if entity_instance.get_entity_id() == entity_id:
                return entity_instance
        return None


class WebCrawlerKnowledgeDataProvider(KnowledgeDataProvider):

    def __init__(self):
        super(WebCrawlerKnowledgeDataProvider, self).__init__()
        self.__crawl_all_entity_data()
        self.__reconstruct_entity_relations()

    def __crawl_all_entity_data(self):

        print 'Initializing company:A16Z...'
        a16z = A16Z({
            'name': 'Andreessen Horowitz',
            'founder': 'Marc Andreessen and Ben Horowitz',
            'location': 'Melo Park, California',
            'website': 'a16z.com',
            'type of business': 'venture capital',
            'business model': None,
            'stage': None,
            'contact info': 'http://a16z.com/about/contact/',
        })
        self.add_entity(a16z)

        print 'Crawling http://a16z.com/portfolio/...'
        print 'Companies in venture/growth stage...'
        for div in self.__find_company_data_divs_from_url('http://a16z.com/portfolio/'):
            self.add_entity(self.__parse_company_data(div))
        print 'Companies in seed stage...'
        for div in self.__find_seed_company_data_divs_from_url('http://a16z.com/portfolio/'):
            self.add_entity(self.__parse_seed_company_data(div))

        print 'Crawling http://a16z.com/about/team/'
        team_member_groups = self.__collect_team_member_profile_urls('http://a16z.com/about/team/')
        for team_member_group_info in team_member_groups:
            for investor_url in team_member_group_info['urls']:
                self.add_entity(self.__parse_investor_data(investor_url, team_member_group_info['role_description']))

        print 'Crawling http://portfoliojobs.a16z.com/...'
        for job in self.__crawl_all_jobs():
            self.add_entity(job)

    def __reconstruct_entity_relations(self):

        print 'Reconstructing entity relations...'

        the_a16z = self.get_all_instances_of_type(A16Z)[0]
        all_companies = self.get_all_instances_of_type(Company)
        all_investors = self.get_all_instances_of_type(Investor)

        print 'A16Z -> COMPANYs'
        the_a16z.relation_value_map['portfolio companies'] = all_companies
        print 'A16Z -> INVESTORs'
        the_a16z.relation_value_map['people'] = all_investors
        # a16z -> POSTs
        # a16z -> PODCASTs

        # JOB -> a COMPANY, COMPANY -> JOBs
        for job_listing in self.get_all_instances_of_type(Job):
            company_name = job_listing.property_value_map['company name']
            matched_company_entity = None
            # search for different variation of the company name
            for company_entity in self.get_all_instances_of_type(Company):
                if self.__compare_company_name(company_entity.property_value_map['name'], company_name):
                    matched_company_entity = company_entity
                    matched_company_entity.add_job(job_listing)
                    job_listing.relation_value_map['company'] = matched_company_entity
                    break
            if not matched_company_entity:
                print 'Cannot find company named %s to associate with' % company_name

    @staticmethod
    def __find_company_data_divs_from_url(url):
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        return soup.findAll('div', 'company')

    @staticmethod
    def __parse_company_data(div):
        name = div.select_one('.meta > li:nth-of-type(1) > h2').next_sibling.strip()
        founder = div.select_one('.meta > li:nth-of-type(2) > h2').next_sibling.strip()
        location = div.select_one('.meta > li:nth-of-type(3) > h2').next_sibling.strip()
        website = div.select_one('.meta > li:nth-of-type(4) > a')['href'].strip()
        type_of_business = div.select_one('.meta > li:nth-of-type(5) > h2').next_sibling.strip()
        business_model = 'to consumer' if 'consumer' in div['class'] else 'to enterprise'

        name_normalized = name.replace(u'\u00a0', ' ')

        return Company({
            'name': name_normalized,
            'founder': founder,
            'location': location,
            'website': website,
            'type of business': type_of_business,
            'business model': business_model,
            'stage': 'venture',
        })

    @staticmethod
    def __find_seed_company_data_divs_from_url(url):
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        return soup.findAll('div', 'company-seed')

    @staticmethod
    def __parse_seed_company_data(div):
        website = div.select_one('.img-wrapper')['href'].strip()

        return Company({
            'name': WebCrawlerKnowledgeDataProvider.__guess_company_name_by_website(website),
            'founder': None,
            'location': None,
            'website': website,
            'type of business': None,
            'business model': None,
            'stage': 'seed',
        })

    @staticmethod
    def __guess_company_name_by_website(url):
        domain_name_no_suffix = urlparse(url).netloc.split('.')[-2]
        # meetearnest.com
        if domain_name_no_suffix.lower().startswith('meet'):
            domain_name_no_suffix = domain_name_no_suffix[len('meet'):]
        terms = domain_name_no_suffix.split('-')
        terms_capitalized = [term.capitalize() for term in terms]
        return ' '.join(terms_capitalized)

    @staticmethod
    def __collect_team_member_profile_urls(url):
        """returns list of urls grouped by role of investor"""
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')

        result = []
        for team_section_div in soup.select('#main .team-section'):
            role_description = WebCrawlerKnowledgeDataProvider.__parse_role_description(team_section_div)
            urls = WebCrawlerKnowledgeDataProvider.__parse_team_member_profile_urls(team_section_div)
            result.append({
                'role_description': role_description,
                'urls': urls
            })

        return result

    @staticmethod
    def __parse_role_description(div):
        text = div.select_one('.team-heading').get_text().strip().lower()
        if text.endswith('.'):
            text = text[0:-1]
        return text

    @staticmethod
    def __parse_team_member_profile_urls(div):
        return [a_tag['href'] for a_tag in div.select('.team-member > a')]

    @staticmethod
    def __parse_investor_data(page_url, role_description):
        print 'Crawling %s...' % page_url

        page = urllib2.urlopen(page_url)
        soup = BeautifulSoup(page, 'html.parser')

        investor_data_div = soup.select_one('.team-member-info')
        name = investor_data_div.select_one('.team-member-name').get_text()
        photo_url = investor_data_div.select_one('img.team-member-photo')['src']
        linkedin_url = None
        for a_tag_in_bio in investor_data_div.select('.team-member-bio a[href]'):
            url = a_tag_in_bio['href']
            if 'linkedin.com' in url:
                linkedin_url = url
                print 'Found LinkedIn profile: ' + linkedin_url

        return Investor({
            'name': name,
            'role': role_description,
            'picture': photo_url,
            'profile': page_url,
            'linkedin': linkedin_url,
        })

    def __crawl_all_jobs(self):
        page = urllib2.urlopen('http://portfoliojobs.a16z.com/')
        soup = BeautifulSoup(page, 'html.parser')

        self.company_name_id_mapping = self.__parse_company_name_to_id_mapping(soup)
        self.city_name_id_mapping = self.__parse_city_name_to_id_mapping(soup)
        function_name_id_mapping = self.__parse_function_name_to_id_mapping(soup)

        result = []

        # NOTE: we are able to infer the id of a company of a location,
        # but not on function since such data is missing in the job listing section.
        # To handle this, we need to manually search all the possible functions and keep track
        # of the function we are currently searching with
        for function_name, function_id in function_name_id_mapping.iteritems():
            result.extend(self.__crawl_all_jobs_under_function(function_name, function_id))

        return result

    @staticmethod
    def __parse_company_name_to_id_mapping(soup):
        return WebCrawlerKnowledgeDataProvider.__parse_name_to_id_mapping(
            soup, '#jobs-filter-options select[name="Company"] option')

    @staticmethod
    def __parse_city_name_to_id_mapping(soup):
        location_mapping = WebCrawlerKnowledgeDataProvider.__parse_name_to_id_mapping(
            soup, '#jobs-filter-options select[name="Location"] option')
        return {
            location_name.split(',')[-1].strip(): location_id
            for location_name, location_id in location_mapping.iteritems()
        }

    @staticmethod
    def __parse_function_name_to_id_mapping(soup):
        return WebCrawlerKnowledgeDataProvider.__parse_name_to_id_mapping(
            soup, '#jobs-filter-options select[name="Function"] option')

    @staticmethod
    def __parse_name_to_id_mapping(soup, option_selector):
        result = {}
        for option in soup.select(option_selector):
            option_name = option.get_text()
            option_id = option['value']
            if option_id.isdigit():
                result[option_name] = option_id
        return result

    def __crawl_all_jobs_under_function(self, function_name, function_id):

        first_page_url = construct_job_search_url(function_id=function_id, page_number=1)
        first_page = BeautifulSoup(urllib2.urlopen(first_page_url), 'html.parser')

        page_number_link = first_page.select_one('a.page-numbers')

        if page_number_link:
            # there is at least one search result
            total_page_number = int(parse_qs(urlparse(page_number_link['href']).query)['tpages'][0])
            print '%d pages found for function: %s...' % (total_page_number, function_name)

            result = []
            for page_number in range(1, total_page_number + 1):
                search_url = construct_job_search_url(function_id=function_id, page_number=page_number)
                result.extend(self.__extract_jobs_in_page(search_url, function_name, function_id))
            return result
        else:
            print 'No job found for function: %s...' % function_name
            return []

    def __extract_jobs_in_page(self, url, function_name, function_id):

        soup = BeautifulSoup(urllib2.urlopen(url), 'html.parser')

        result = []
        for job_div in soup.select('table.job_listings tbody tr'):
            title = job_div.select_one('td.title-job > a').get_text()
            job_id = job_div.select_one('td.title-job > a')['href'].split('=')[-1]
            company_name = job_div.select_one('td.name-company > a').get_text()
            location_name = job_div.select_one('td.location > a').get_text()
            city_name = location_name.split(',')[0]

            # company id is required
            company_id = self.company_name_id_mapping.get(company_name)
            if not company_id:
                print 'Invalid job listing. Company name not found: %s' % company_name
                continue
            # location id is optional since some of the job descriptions use special format
            location_id = self.city_name_id_mapping.get(city_name)

            result.append(Job({
                'job_id': job_id,
                'title': title,
                'company name': company_name,
                'company_id': company_id,
                'location': location_name,
                'location_id': location_id,
                'function': function_name,
                'function_id': function_id,
            }))

        print '%d jobs found in %s' % (len(result), url)
        return result

    @staticmethod
    def __compare_company_name(name_a, name_b):
        non_char_regex = '[^a-zA-Z0-9]'
        name_a_norm = name_a.lower().capitalize()
        name_b_norm = name_b.lower().capitalize()
        if name_a_norm == name_b_norm:
            return True
        if name_a_norm == re.compile(non_char_regex).split(name_b_norm)[0]:
            return True
        if name_b_norm == re.compile(non_char_regex).split(name_a_norm)[0]:
            return True
        if name_a_norm == re.sub(non_char_regex, '', name_b_norm):
            return True
        if name_b_norm == re.sub(non_char_regex, '', name_a_norm):
            return True
        return False


class JsonFileKnowledgeDataProvider(KnowledgeDataProvider):

    def __init__(self, file_path):
        super(JsonFileKnowledgeDataProvider, self).__init__()
        with open(file_path, 'r') as input_file:
            json_data = json.load(input_file)
            # unmarshall the entity properties before reconstructing relations
            self.__load_entity_property_data_from_dict(json_data)
            self.__load_entity_relations_from_dict(json_data)

    def __load_entity_property_data_from_dict(self, json_dict):
        for known_entity_type in self.entity_map:
            entity_json_list = json_dict.get(known_entity_type.__name__)
            if entity_json_list:
                for entity_data in entity_json_list:
                    self.add_entity(EntityJsonConverter.load_from_json_dict(entity_data))

    def __load_entity_relations_from_dict(self, json_dict):
        for known_entity_type in self.entity_map:
            for entity_data in json_dict.get(known_entity_type.__name__):
                # search the already-materialized instance
                entity_id = entity_data[ENTITY_JSON_ID]
                entity_instance = self.get_entity_instance(known_entity_type, entity_id)

                for relation_name, relation_ref_dat in entity_data[ENTITY_JSON_RELATIONS].iteritems():
                    related_entity_class_name = relation_ref_dat[ENTITY_JSON_RELATION_REF_ENTITY_TYPE]
                    related_entity_class = find_entity_class_by_name(related_entity_class_name)
                    related_entity_id = relation_ref_dat.get(ENTITY_JSON_RELATION_REF_ENTITY_ID)
                    related_entity_ids = relation_ref_dat.get(ENTITY_JSON_RELATION_REF_ENTITY_IDS)

                    # search the related entities
                    if related_entity_id:
                        entity_instance.relation_value_map[relation_name] = \
                            self.get_entity_instance(related_entity_class, related_entity_id)
                    elif related_entity_ids:
                        entity_instance.relation_value_map[relation_name] = \
                            [self.get_entity_instance(related_entity_class, single_id) for single_id in related_entity_ids]
                    else:
                        raise Exception('Neither %s or %s is found in the relation reference'
                                        % (ENTITY_JSON_RELATION_REF_ENTITY_ID, ENTITY_JSON_RELATION_REF_ENTITY_IDS))


def construct_job_search_url(company_id='%', location_id='%', function_id='%', page_number=1):
    """
    for simplicity, all ids are str type
    """
    return 'http://portfoliojobs.a16z.com/careers_home.php?Company=%s&Function=%s&Location=%s&p=%d' \
              % (company_id, function_id, location_id, page_number)


