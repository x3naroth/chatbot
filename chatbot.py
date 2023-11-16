import os
import PyPDF2
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Suppress transformers and PyTorch warnings
warnings.filterwarnings("ignore")

# Declare pdf_path as a global variable
pdf_path = r'route to pdf'

# Load PDF content
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page_num in range(len(pdf_reader.pages)):
            pdf_text += pdf_reader.pages[page_num].extract_text()
    print("Length of extracted text:", len(pdf_text))
    return pdf_text

# Answer questions using transformers pipeline
def answer_question(question, context, model, num_answers=3):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1
    answer = tokenizer.decode(inputs["input_ids"][0, answer_start:answer_end])

    # Get additional answers
    n_best_size = min(num_answers, len(outputs.start_logits))
    answers = []
    for i in range(n_best_size):
        start = outputs.start_logits[i].argmax().item()
        end = outputs.end_logits[i].argmax().item() + 1
        answers.append(tokenizer.decode(inputs["input_ids"][0, start:end]))
    
    return answers

# Main function
def main():
    global pdf_path  # Declare pdf_path as global
    pdf_text = extract_text_from_pdf(pdf_path)

    # Load the model separately to avoid warnings
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    while True:
        user_question = input("Ask a question (type 'exit' to end): ")
        if user_question.lower() == 'exit':
            break
        answers = answer_question(user_question, pdf_text, model, num_answers=3)
        print("Answers:")
        for i, answer in enumerate(answers):
            print(f"{i+1}. {answer}")

if __name__ == "__main__":
    main()
