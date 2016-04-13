from intercom import Intercom, Conversation
import os

__author__ = 'Deyang'

Intercom.app_id = os.environ.get('INTERCOM_APP_ID')
Intercom.app_api_key = os.environ.get('INTERCOM_APP_API_KEY')

ADMIN_DEREK_ID = 426663
ADMIN_BOT_ID = 426928

SUBSCRIBED_TOPIC = 'conversation.user.replied'


def reply_to_user(conversation_id, reply_msg):
    conversation = Conversation.find(id=conversation_id)
    conversation.reply(
        type='admin', id=str(ADMIN_BOT_ID),
        message_type='comment', body=reply_msg)


def parse_notification_and_should_reply(notification):
    print "Raw notification: %s" % notification
    try:
        conversation_id = notification['data']['item']['id']
        assignee = notification['data']['item']['assignee']
        topic = notification['topic']
        if topic != SUBSCRIBED_TOPIC:
            return None
        elif assignee['type'] != 'nobody_admin':
            if assignee['type'] == 'admin':
                if assignee['id'] != ADMIN_BOT_ID:
                    return None
            else:
                return None

        msg = notification['data']['item']['conversation_parts']['conversation_parts'][0]['body']
        if msg.startswith('<p>'):
            msg = msg[3:]
        if msg.endswith('</p>'):
            msg = msg[:-4]
        return conversation_id, msg
    except Exception as e:
        print e
        return None, None