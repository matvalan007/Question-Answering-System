from transformers import pipeline
from fastapi import FastAPI

app = FastAPI()

model_checkpoint = r"C:\Users\USER\OneDrive\Desktop\QA_model\model"
question_answerer = pipeline("question-answering", model=model_checkpoint)


#ans = question_answerer(question=question, context=context)

@app.post("/question-answering")
def get_answer(context : str, question : str):
    # Use the loaded model for question answering
    result = question_answerer(question=question, context=context)
    return result['answer']

