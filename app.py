import dspy
import os
from flask import Flask, request, jsonify, render_template
from redisretriever import DSPythonicRMClient
from config.llm import TOGETHER_API_URL, TOGETHER_API_KEY, TOGETHER_MODEL_ID, USE_INST_TEMPLATE, DSPY_INPUT_DESC, DSPY_OUTPUT_DESC
from config.vectordb import REDIS_URL, REDIS_PORT, NUM_DOCS_RETURNED


retriever = DSPythonicRMClient(REDIS_URL, REDIS_PORT, NUM_DOCS_RETURNED)
together = dspy.Together(model=TOGETHER_MODEL_ID)
together.use_inst_template = USE_INST_TEMPLATE

dspy.settings.configure(lm=together, rm=retriever)

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc=DSPY_INPUT_DESC)
    question = dspy.InputField()
    answer = dspy.OutputField(desc=DSPY_OUTPUT_DESC)

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

rag = RAG()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Welcome to use the chat API! Please ask me questions about insurance products.'

@app.route('/chat', methods=['GET', 'POST'])
def api():
    # Get JSON data
    data = request.get_json()

    if 'question' not in data:
        return jsonify({"message": "No question provided"}), 400
    else:
        question = data['question']
        # print(type(question), question) # for debugging
        context = rag(question).context
        answer = rag(question).answer
        # Return JSON data
        return jsonify({
            "question": question,
            "context": context,
            "answer": answer
        }), 200

@app.route('/chatwebsite', methods=['GET'])
def chatwebsite():
    return render_template('chatwebsite.html')

if __name__ == '__main__':
    app.run(port=5002, debug=True)
