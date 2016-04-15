from chatterbot import ChatBot
from wit import message
import re
import os
import json
from util.util import get_logger

__author__ = 'Deyang'

bot_logger = get_logger('bot')
wit_token = 'QUCDCX7MQX4FLYGONBEYLGDHKSTIUFTQ'


# identity_intent = [conversation[0] for conversation in identity_data]
# ask_company_intent = [conversation[0] for conversation in ask_company_data]
# ask_customer_intent = [conversation[0] for conversation in ask_customer_data]
# ask_doc_intent = [conversation[0] for conversation in ask_doc_data]
# ask_price_intent = [conversation[0] for conversation in ask_price_data]
# ask_product_intent = [conversation[0] for conversation in ask_product_data]
# ask_story_intent = [conversation[0] for conversation in ask_story_data]


GREETING_INTENT = 'greetings'
IDENTITY_INTENT = 'identity'
TRY_INTENT = 'try'
EMAIL_INTENT = 'email'
INSULT_AND_SEX_INTENT = 'insult_and_sex'
ASK_PRICE_INTENT = 'ask_price'
ASK_CUSTOMER_INTENT = 'ask_customer'
ASK_TEAM_INTENT = 'ask_team'
ASK_INTENT_PATTERN = re.compile('ask.*')


def load_data(file_name):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, '..', 'data', file_name)
    fd = open(filepath, 'r')
    lines = fd.readlines()
    skipped_lines = [l for l in lines if not l.strip().startswith('#')]
    content = ''.join(skipped_lines)
    return json.loads(content)


class DummyEmailBot(object):
    def get_response(self, input):
        return "Thanks! Please sit tight. I've notified a member of the team"


class Bots(object):

    def _install_bot(self, intent, data):
        new_bot = self.get_new_bot("%s_bot" % intent)
        for conversation in data:
            new_bot.train(conversation)
        self.set_readonly(new_bot)
        self.intent_to_bot_dict[intent] = new_bot

    def __init__(self):
        self.intent_to_bot_dict = dict()
        # load data
        identity_data = load_data('identity.qa')
        ask_company_data = load_data('company.qa')
        ask_customer_data = load_data('customer.qa')
        ask_doc_data = load_data('doc.qa')
        ask_price_data = load_data('price.qa')
        ask_product_data = load_data('product.qa')
        ask_story_data = load_data('story.qa')
        ask_beta_data = load_data('beta.qa')
        ask_integration_data = load_data('integration.qa')
        ask_team_data = load_data('team.qa')
        additional_greeting = load_data('greeting.qa')
        try_data = load_data('try.qa')
        insult_and_sex_data = load_data('insult_and_sex.qa')

        self.general_bot = self.get_new_bot('general_bot')
        self.general_bot.train(
            "chatterbot.corpus.english",
        )
    
        greeting_bot = self.get_new_bot('greeting_bot')
        greeting_bot.train("chatterbot.corpus.english.greetings")
        for conversation in additional_greeting:
            greeting_bot.train(conversation)
        self.set_readonly(greeting_bot)
        self.intent_to_bot_dict[GREETING_INTENT] = greeting_bot

        self._install_bot(IDENTITY_INTENT, identity_data)
        self._install_bot(TRY_INTENT, try_data)
        self._install_bot(INSULT_AND_SEX_INTENT, insult_and_sex_data)
        self._install_bot(ASK_PRICE_INTENT, ask_price_data)
        self._install_bot(ASK_CUSTOMER_INTENT, ask_customer_data)
        self._install_bot(ASK_TEAM_INTENT, ask_team_data)

        self.company_bot = self.get_new_bot('company_bot')
        all_data = ask_company_data + ask_doc_data + ask_story_data + \
            ask_product_data + ask_beta_data + ask_integration_data
        for conversation in all_data:
            self.company_bot.train(conversation)
        self.set_readonly(self.company_bot)

        self.email_bot = DummyEmailBot()

    @staticmethod
    def get_new_bot(name):
        return ChatBot(
            name,
            logic_adapters=[
                "chatterbot.adapters.logic.ClosestMatchAdapter",
                "chatterbot.adapters.logic.ClosestMeaningAdapter"
            ],
            io_adapters=[
                "chatterbot.adapters.io.NoOutputAdapter"
            ],
            database=name + "_database.db"
        )

    @staticmethod
    def set_readonly(bot):
        for adapter in bot.storage_adapters:
            adapter.read_only = True

    def _select_bot(self, in_msg):
        # hack based on html. No need to query wit.ai
        if 'mailto:' in in_msg:
            intent = EMAIL_INTENT
            confidence = 1.0
        else:
            resp = message(wit_token, in_msg)
            try:
                intent = resp['outcomes'][0]['intent']
                confidence = resp['outcomes'][0]['confidence']
                bot_logger.info("Predicting intent: %s" % intent)
                bot_logger.info("Predicting confidence: %s" % confidence)
            except Exception as e:
                bot_logger.error("Wit exception: %s" % e)
                bot_logger.error("Wit reps: %s" % resp)

        if confidence < 0.3:
            return self.general_bot
        if intent in self.intent_to_bot_dict:
            return self.intent_to_bot_dict[intent]

        if ASK_INTENT_PATTERN.match(intent):
            return self.company_bot
        else:
            return self.general_bot

    def chat(self, question):
        bot = self._select_bot(question)
        answer = bot.get_response(question)
        bot_logger.info("In msg: <%s> and out msg: <%s>" % (question, answer))
        return answer

if __name__ == '__main__':
    bots = Bots()
    while True:
        in_msg = raw_input()
        out_msg = bots.chat(in_msg)
        print "Buddy bot :> %s" % out_msg
