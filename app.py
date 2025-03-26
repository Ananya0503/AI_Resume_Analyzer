from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import faiss
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests
app.secret_key = os.getenv("FLASK_SECRET_KEY", "defaultsecretkey")

# Upload configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Google Gemini API Key
GENAI_API_KEY = ("YOUR_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("Google Gemini API key is missing in the .env file")
genai.configure(api_key=GENAI_API_KEY)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def split_text(text, chunk_size=500, overlap=50):
    """Splits text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def get_gemini_response(prompt):
    """Generates a response using Google Gemini API."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "No response generated."
    except Exception as e:
        logger.error(f"Error with Google Gemini API: {e}")
        return "Error generating response from Gemini."


def embed_chunks(chunks):
    """Creates embeddings for text chunks using SentenceTransformer and FAISS."""
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def search_similar_chunks(index, embeddings, query, chunks, k=3):
    """Finds the top-k similar chunks based on similarity search."""
    query_embedding = embedding_model.encode([query])
    _, top_k_indices = index.search(query_embedding, k)
    return [chunks[i] for i in top_k_indices[0]]


def analyze_resume(chunks, query):
    """Performs resume analysis using Gemini AI and FAISS."""
    index, embeddings = embed_chunks(chunks)
    relevant_chunks = search_similar_chunks(index, embeddings, query, chunks)
    combined_text = "\n\n".join(relevant_chunks)
    prompt = f"{query}\n\nResume Content:\n{combined_text}"
    return get_gemini_response(prompt)


@app.route('/')
def home():
    """Render the upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and analysis."""
    if 'file' not in request.files:
        flash("No file part")
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process Resume
            resume_text = extract_text_from_pdf(file_path)
            chunks = split_text(resume_text)

            # Queries for analysis
            queries = {
                "summary": "Provide a detailed summary of the resume:",
                "strengths": "Analyze and explain the strengths of this resume:",
                "weaknesses": "Analyze the weaknesses of this resume and suggest improvements:",
                "job_suggestions": "Based on this resume, suggest job roles that can be applied on LinkedIn:"
            }

            # Generate responses
            response = {key: analyze_resume(chunks, query) for key, query in queries.items()}

            flash("File uploaded and analyzed successfully!")
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error processing resume: {e}")
            flash("An error occurred during analysis. Please try again.")
            return redirect(url_for('home'))

    flash("Invalid file type. Only PDFs are allowed.")
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
