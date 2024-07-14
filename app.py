import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
#from transformers import AutoModelForCausalLM
import torch
import base64

# Model and tokenizer
checkpoint = "LaMini-Flan-T5"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', offload_folder='offload_weights', use_safetensors=True, torch_dtype=torch.float32)

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts += text.page_content
    return final_texts

def llmpipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=1500,
        min_length=50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]["summary_text"]
    return result

def qa_pipeline(filepath, question):
    input_text = file_preprocessing(filepath)
    qa_pipe = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        min_length=50
    )
    result = qa_pipe(question=question, context=input_text)
    result = result["answer"]
    return result

import os
import tempfile

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
import os
import tempfile

# ... (rest of the code remains the same)

def main():
    st.title("PDF Summarization & Question ")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            # Create a temporary file to store the uploaded file
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, uploaded_file.name)
                with open(tmp_file, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Use the temporary file path
                summarized_text = llmpipeline(tmp_file)
                st.write(summarized_text)
                
                question = st.text_input("Ask a question about the PDF:")
                if question:
                    answer = qa_pipeline(tmp_file, question)
                    st.write("Answer:", answer)

                displayPDF(tmp_file)

if __name__ == "__main__":
    main()

