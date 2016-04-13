from chatterbot import ChatBot
from wit import message
import re
import os
import json

__author__ = 'Deyang'

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
ASK_INTENT_PATTERN = re.compile('ask.*')


def load_data(file_name):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, '..', 'data', file_name)
    fd = open(filepath, 'r')
    lines = fd.readlines()
    skipped_lines = [l for l in lines if not l.startswith('#')]
    content = ''.join(skipped_lines)
    return json.loads(content)


class DummyEmailBot(object):
    def get_response(self, input):
        return "Thanks! Please sit tight. I've notified a member of the team"


class Bots(object):
    def __init__(self):
        # load data
        identity_data = load_data('identity.qa')
        ask_company_data = load_data('company.qa')
        ask_customer_data = load_data('customer.qa')
        ask_doc_data = load_data('doc.qa')
        ask_price_data = load_data('price.qa')
        ask_product_data = load_data('product.qa')
        ask_story_data = load_data('story.qa')
        additional_greeting = load_data('greeting.qa')
        try_data = load_data('try.qa')

        self.general_bot = self.get_new_bot('general_bot')
        self.general_bot.train(
            "chatterbot.corpus.english",
        )
    
        self.greeting_bot = self.get_new_bot('greeting_bot')
        self.greeting_bot.train("chatterbot.corpus.english.greetings")
        for conversation in additional_greeting:
            self.greeting_bot.train(conversation)

        self.identity_bot = self.get_new_bot('identity_bot')
        for conversation in identity_data:
            self.identity_bot.train(conversation)
        self.set_readonly(self.identity_bot)

        self.company_bot = self.get_new_bot('self.company_bot')
        all_data = ask_company_data + ask_customer_data + ask_price_data + ask_doc_data + ask_story_data + ask_product_data

        for conversation in all_data:
            self.company_bot.train(conversation)

        self.set_readonly(self.company_bot)

        self.try_bot = self.get_new_bot('try_bot')
        for conversation in try_data:
            self.try_bot.train(conversation)
        self.set_readonly(self.try_bot)

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

    def select_bot(self, in_msg):
        # hack based on html. No need to query wit.ai
        if 'mailto:' in in_msg:
            intent = EMAIL_INTENT
            confidence = 1.0
        else:
            resp = message(wit_token, in_msg)
            intent = resp['outcomes'][0]['intent']
            confidence = resp['outcomes'][0]['confidence']
        print intent
        print confidence
        if confidence < 0.2:
            return self.general_bot
        if intent == GREETING_INTENT:
            return self.greeting_bot
        elif intent == IDENTITY_INTENT:
            return self.identity_bot
        elif intent == EMAIL_INTENT:
            return self.email_bot
        elif intent == TRY_INTENT:
            return self.try_bot
        elif ASK_INTENT_PATTERN.match(intent):
            return self.company_bot
        else:
            return self.general_bot

    def chat(self, question):
        bot = self.select_bot(question)
        answer = bot.get_response(question)
        print "in: <%s> and out: <%s>" % (question, answer)
        return answer

if __name__ == '__main__':
    bots = Bots()
    while True:
        in_msg = raw_input()
        out_msg = bots.chat(in_msg)
        print "Buddy bot :> %s" % out_msg
