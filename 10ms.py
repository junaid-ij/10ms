import re
import fitz
from bnlp import BasicTokenizer
from bnlp import NLTKTokenizer
import nltk
from sentence_transformers import SentenceTransformer, util


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()


def bangla_token(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    bnltk = NLTKTokenizer()
    sentence_tokens = bnltk.sentence_tokenize(text)
    cleaned_texts = [text.replace('\n', ' ') for text in sentence_tokens]
    return cleaned_texts


def find_closest_match(query, sentences):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # Supports Bangla
    embeddings = model.encode(sentences + [query], convert_to_tensor=True)

    query_embedding = embeddings[-1]
    sentence_embeddings = embeddings[:-1]

    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    best_idx = scores.argmax()
    return sentences[best_idx]

def main():
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    query = input("আপনার প্রশ্ন লিখুন: ")

    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)


    print("Tokenizing sentences...")
    sentences = bangla_token(raw_text)

    print("Finding closest sentence...")
    best_match = find_closest_match(query, sentences)

    print("\nClosest matching sentence:")
    print(best_match)


if __name__ == "__main__":
    main()





