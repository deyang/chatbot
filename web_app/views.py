__author__ = 'Deyang'
from flask import render_template, request, make_response, redirect, session, flash, jsonify
from web_app import app
from buddy_bot.main import Bots
from integations.intercomm import parse_notification_and_should_reply, reply_to_user

bots = Bots()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/message', methods=['POST'])
def chat_api():
    in_msg = request.form.get("msg", "")
    out_msg = bots.chat(in_msg)
    return make_response(jsonify(response=out_msg), 201)


@app.route('/intercomm_webhook', methods=['POST'])
def intercomm_webhook():
    app.logger.info("Intercomm webhook triggered")
    notification = request.json
    conversation_id, in_msg = parse_notification_and_should_reply(notification)

    if conversation_id:
        out_msg = bots.chat(in_msg)
        reply_to_user(conversation_id, out_msg)
    return make_response(jsonify(status=0), 201)


@app.route('/smooch_webhook', methods=['POST'])
def smooch_webhook():
    print "inside smooch webhook"
    print request.json
    return make_response(jsonify(status=0), 201)

