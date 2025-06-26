import streamlit as st
import pandas as pd
import requests
import os
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import fitz
import re
import matplotlib
matplotlib.use("Agg")
import os
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    import streamlit as st
    st.error("API key not found. Please set TOGETHER_API_KEY as a repository variable.")
    st.stop()


MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

def split_response(response):
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"(plt\..*?)(?=\n\n|$)", response, re.DOTALL)
    explanation_only = re.sub(r"```(?:python)?\n.*?```", "", response, flags=re.DOTALL)
    return explanation_only.strip(), code_blocks



def analyze_with_maverick(prompt, file_content=None, file_type=None):
    messages = [{"role": "user", "content": prompt}]

    if file_content:
        if file_type == "image":
            messages[0]["content"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{file_content}"}},
                {"type": "text", "text": prompt}
            ]
        else:
            messages[0]["content"] += f"\n\n[Attached {file_type} content:]\n{file_content}"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.together.ai/v1/chat/completions",
                             json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"API Error {response.status_code}: {response.text}"

def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text[:5000]

def process_file(uploaded_file):
    file_type = uploaded_file.type.split('/')[-1]

    if file_type in ["csv", "vnd.ms-excel"]:
        df = pd.read_csv(uploaded_file)
        return df.head(100).to_string(), "data"

    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        return df.head(100).to_string(), "data"

    elif file_type == "plain":
        content = uploaded_file.read().decode("utf-8")
        return content[:1250], "text"

    elif file_type == "pdf":
        text = extract_pdf_text(uploaded_file)
        return text, "text"

    elif file_type in ["png", "jpeg", "jpg"]:
        image = Image.open(uploaded_file)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_img, "image"

    return "Unsupported file type", "text"


st.title("LLaMA-4 Maverick Data Analyst")

uploaded_file = st.file_uploader(
    "Upload a document or image",
    type=["csv", "txt", "xlsx", "pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_preview, file_type = process_file(uploaded_file)

    st.subheader("File Preview")
    if file_type == "data" or file_type == "text":
        st.code(file_preview, language="text")
    elif file_type == "image":
        decoded_img = base64.b64decode(file_preview)
        st.image(decoded_img)

    st.markdown("---")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.text_input("🔎 Ask a question about the file:", key="input_query")

    if user_query:
        with st.spinner("Analyzing..."):
            response = analyze_with_maverick(
                prompt = f"""
                              You are a data analyst using data give to you as plain text.

                              - DO NOT use `pd.read_csv()` or attempt to read from a file.
                              - Use the given data as plain text to you for all data operations and visualizations.
                              - Provide a brief explanation **before** the code block.
                              - Return the full Python code in a properly formatted Markdown code block (using ```python).
                              - If the task involves plotting, use matplotlib or seaborn, and call `plt.show()` or use `plt.gcf()` at the end.
                              - If the data is image,remove any information that user will not require or will not be able to process as human, just answer what he asks
                              Here is the preview of the data:

                              {file_preview}

                              User query:
                              {user_query}
                              """
                              ,
                file_content=file_preview,
                file_type=file_type
            )
        explanation, code_blocks = split_response(response)
        if code_blocks:
          for i, code in enumerate(code_blocks):
              st.subheader(f"Code Block {i+1}")
              st.code(code, language="python")
              try:
                  exec(code, globals())
                  st.pyplot(plt.gcf())
                  plt.clf()
                  st.success("Executed successfully")
              except Exception as e:
                  st.error(f"Execution error: {e}")
          if explanation:
              st.markdown("Explanation")
              st.write(explanation)
        else:
            st.markdown("Response")
            st.write(response)


        st.session_state.history.append((user_query, response))


    if st.session_state.history:
        st.markdown("### 💬 Chat History")
        for i, (q, a) in enumerate(st.session_state.history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Model:** {a}")
            st.markdown("---")
