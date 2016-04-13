from intercom import Intercom, Conversation
import os
import html2text
from util.util import get_logger

__author__ = 'Deyang'

intercomm_logger = get_logger('intercomm')

Intercom.app_id = os.environ.get('INTERCOM_APP_ID')
Intercom.app_api_key = os.environ.get('INTERCOM_APP_API_KEY')

ADMIN_DEREK_ID = '426663'
ADMIN_BOT_ID = '426928'

SUBSCRIBED_REPLY_TOPIC = 'conversation.user.replied'
SUBSCRIBED_CREATE_TOPIC = 'conversation.user.created'


def reply_to_user(conversation_id, reply_msg):
    conversation = Conversation.find(id=conversation_id)
    conversation.reply(
        type='admin', id=str(ADMIN_BOT_ID),
        message_type='comment', body=reply_msg)


def parse_notification_and_should_reply(notification):
    intercomm_logger.debug("Raw notification: %s" % notification)
    try:
        conversation_id = notification['data']['item']['id']
        assignee = notification['data']['item']['assignee']
        topic = notification['topic']
        if topic != SUBSCRIBED_CREATE_TOPIC and topic != SUBSCRIBED_REPLY_TOPIC:
            return None, None
        elif assignee['type'] != 'nobody_admin':
            if assignee['type'] == 'admin':
                if assignee['id'] != ADMIN_BOT_ID:
                    return None, None
            else:
                return None, None
        if topic == SUBSCRIBED_REPLY_TOPIC:
            msg = notification['data']['item']['conversation_parts']['conversation_parts'][0]['body']
        else:
            msg = notification['data']['item']['conversation_message']['body']
        if msg is None or len(msg) == 0:
            return None, None
        msg = html2text.html2text(msg).strip()
        intercomm_logger.debug("Msg from notification" % msg)
        return conversation_id, msg
    except Exception as e:
        intercomm_logger.error("Parsing notification exception: %s" % e)
        return None, None
