from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from question_similarity1 import get_similar_question_answer
from document_qa1 import get_document_answer

app = Flask(__name__, static_url_path='/static')

cors = CORS(app, resources={
    r"*": {
        "origins": "*",
        "allow_headers": "*",
        "methods": "GET,HEAD,POST,OPTIONS,PUT,DELETE"
    }
})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data['userQuestion']
    chatbot_response, similarity = get_similar_question_answer(user_question)
    if similarity < 0.5:
        chatbot_response = get_document_answer(user_question)
    return jsonify({'chatbotResponse': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)