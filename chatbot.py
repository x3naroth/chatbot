import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load PDF content
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    print("Length of extracted text:", len(text))
    return text

# Process and store document vectors
def create_document_vectors(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = [sent.strip() for sent in text.split('.') if sent]
    vectors = model.encode(sentences)
    return model, sentences, vectors

# Answer questions using document vectors
def answer_question(question, sentences, vectors):
    question_vector = model.encode(question)
    similarities = cosine_similarity([question_vector], vectors)[0]
    most_similar_idx = similarities.argmax()
    return sentences[most_similar_idx]


# Main function
def main():
    pdf_path = r'pdf path route'
    pdf_text = extract_text_from_pdf(pdf_path)
    model, sentences, vectors = create_document_vectors(pdf_text)  # Asegúrate de tener 'model' aquí

    while True:
        user_question = input("Ask a question (type 'exit' to end): ")
        if user_question.lower() == 'exit':
            break
        answer = answer_question(user_question, sentences, vectors)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
