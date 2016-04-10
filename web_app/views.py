__author__ = 'Deyang'
from flask import render_template, request, make_response, redirect, session, flash, jsonify
from web_app import app
from buddy_bot.main import chat


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/message', methods=['POST'])
def chat_api():
    in_msg = request.form.get("msg", "")
    out_msg = chat(in_msg)
    return make_response(jsonify(response=out_msg), 201)
