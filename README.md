1. install the requirements from requirements.txt
2. Used tools: Jupyter Notebook

QnA
1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
   ans: PyMuPDF -> fitz.
   formatting challange: Bangla texts are broken. Couldn't fix it.
2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
   ans: bnlp toolkit - > sentence_tokenize
3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
   ans: SentenceTransformer('distiluse-base-multilingual-cased-v1')
4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
   ans:
5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
   ans: 
6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
   ans: the results does not seem to be relevant. points of improvement are,
   - better text extraction, tokenizing
   - removing irrelevant words and informations such as , multiple question, page number etc.
   - better embedding model , a well trained model
