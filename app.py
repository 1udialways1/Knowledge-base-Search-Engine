import streamlit as st
from utils import extract_text_from_pdf, chunk_text, create_vector_store
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv
from pydantic import SecretStr
import time
import numpy as np
import sys

sys.path.append('C:/Users/uddip/PycharmProjects/pythonProject16/venv/lib/site-packages')

# NEW: Import HumanMessage for proper LLM usage
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in your .env file.")
GROQ_API_KEY = SecretStr(api_key)

st.title("Hello! –  AI PDF Companion ")

# Initialize chat history, stats, and vector store state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'is_generating' not in st.session_state:
    st.session_state['is_generating'] = False
if 'timings' not in st.session_state:
    st.session_state['timings'] = []
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'all_text' not in st.session_state:
    st.session_state['all_text'] = ""
if 'summary' not in st.session_state:
    st.session_state['summary'] = ""

# --- UPLOAD & PROCESS MULTIPLE PDFs ---
uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type="pdf")
all_texts = []

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        st.info(f"Extracting text from {uploaded_file.name}...")
        text = extract_text_from_pdf(pdf_path)
        all_texts.append(text)
        st.success(f"Text extracted from {uploaded_file.name}!")
    st.session_state['all_text'] = "\n\n".join(all_texts)

    st.info("Chunking all PDFs...")
    all_chunks = chunk_text(st.session_state['all_text'])
    st.success(f"Total chunks: {len(all_chunks)}")

    st.info("Creating vector store (may take a minute)...")
    vector_store = create_vector_store(all_chunks)
    st.session_state['vector_store'] = vector_store
    st.success("Vector store ready!")

    # --- GENERATE SUMMARY FOR ALL PDFs ---
    st.info("Generating summary for all uploaded PDFs...")
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
    summary_prompt = "Provide a concise summary for the following text:\n" + st.session_state['all_text'][:4000]
    summary_response = llm([HumanMessage(content=summary_prompt)])
    if hasattr(summary_response, 'content'):
        summary = summary_response.content
    else:
        summary = str(summary_response)
    st.session_state['summary'] = summary
    st.success("Summary generated!")

# --- SHOW SUMMARY IF AVAILABLE ---
if st.session_state['summary']:
    st.markdown("### 📄 Summary of Your PDFs")
    st.info(st.session_state['summary'])

# --- QA SECTION ---
if st.session_state['vector_store']:
    st.markdown("---")
    st.subheader("Ask questions about your PDFs")

    col1, col2 = st.columns([8, 1])
    with col1:
        user_question = st.text_input(
            "Type your question:",
            key="user_question",
            label_visibility="collapsed"
        )
    with col2:
        submit = st.button("➡️")

    if st.session_state['is_generating']:
        st.warning("⏳ Generating answer, please wait...")

    if submit and st.session_state.get('user_question', ''):
        st.session_state['is_generating'] = True

        with st.spinner("⏳ Generating answer..."):
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state['vector_store'].as_retriever()
            )
            t0 = time.perf_counter()
            answer = qa.invoke(st.session_state['user_question'])
            t1 = time.perf_counter()
            response_time = t1 - t0
            answer_str = answer["result"]
            st.session_state['chat_history'].append({
                "question": st.session_state['user_question'],
                "answer": answer_str,
                "time": response_time
            })
            st.session_state['timings'].append(response_time)

        st.session_state['is_generating'] = False
        st.session_state.pop("user_question")
        st.rerun()

    # Display chat history with inline copy button + feedback
    for i, chat in enumerate(reversed(st.session_state['chat_history'])):
        answer_id = f"answer-{i}"
        button_id = f"copy-btn-{i}"
        feedback_id = f"copy-feedback-{i}"
        st.markdown(f"""
        <div style='background-color:#1976D2; border-radius:12px; padding:1em; margin-block-end:1em; 
                    box-shadow: 0 2px 8px #0002; color:white; position: relative;'>
            <b style='color:#FFC107;'>You:</b> {chat['question']}<br><br>
            <b style='color:#00E676;'>NoteMate:</b>
            <div id='{answer_id}' style='font-size:1.1em; margin-block-end: 8px; white-space: pre-wrap;'>
                {chat['answer']}
            </div>
            <button id="{button_id}" style="position:absolute; inset-block-start:10px; inset-inline-end:10px; background:rgba(255,255,255,0.1); border:none; 
                      color:#fff; cursor:pointer; font-size:16px;" title="Copy answer" onclick="
                const text = document.getElementById('{answer_id}').innerText;
                navigator.clipboard.writeText(text).then(() => {{
                    const fb = document.getElementById('{feedback_id}');
                    fb.style.opacity = '1';
                    setTimeout(() => {{ fb.style.opacity = '0'; }}, 1500);
                }});
            ">📋</button>
            <div id="{feedback_id}" style="color:#fff; position:absolute; inset-block-start:35px; inset-inline-end:10px; opacity:0; transition: opacity 0.3s;">
                Copied!
            </div>
            <div style='font-size:0.9em;color:#ccc; margin-block-start:8px;'>⏱️ {chat['time']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    # Performance Summary
    if len(st.session_state['timings']) > 3:
        arr = np.array(st.session_state['timings'])
        st.markdown("### 📊 Performance Summary")
        st.write(
            f"- Average response: {arr.mean():.2f}s\n"
            f"- Median response: {np.median(arr):.2f}s\n"
            f"- 95th percentile: {np.percentile(arr, 95):.2f}s"
        )
