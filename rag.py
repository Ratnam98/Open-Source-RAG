import os , chromadb, transformers , torch , base64 , time , textwrap
from langchain.document_loaders import PyPDFDirectoryLoader , PyPDFLoader , PDFMinerLoader
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsClusteringFilter,EmbeddingsRedundantFilter
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings , HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from chroma_config import client_settings

st.set_page_config(layout="wide")
device=torch.device("cuda")

persist_directory="DB_DIR"

model_name="mistralai/Mistral-7B-Instruct-v0.2"
model_config = transformers.AutoConfig.from_pretrained(model_name)

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    model_name="BAAI/bge-large-en"
    model_kwargs={'device':'cpu'}
    encode_kwargs={'normalize_embeddings':True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=client_settings)
    db.persist()
    db=None

@st.cache_resource
def llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "!"
    tokenizer.padding_side = "right"
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
    )

    config = {
    'max_new_tokens': 4096,
    'repetition_penalty': 1.1,
    'temperature': 0.01,
    'top_k': 50,
    'top_p': 0.9
    }
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,config=config
    )
    from transformers import pipeline
    text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.01,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=4096
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return mistral_llm

prompt_template = """Based on the documents provided, identify the most relevant information to answer the question comprehensively.
Consider the context and specifics of each document to ensure the answer is accurate and directly applicable to the user's query.

Context: {context}
Question: {question}

To ensure the answer is helpful and structured, follow these guidelines:
1. Clearly identify the document or source that best addresses the question.
2. From that source, meticulously extract the pertinent information.
3. Lay out the answer as a detailed, step-by-step guide. Start each step with a verb, indicating an action to be taken or an insight to be understood, and include the cause or reasoning behind each step when applicable.
4. Conclude with a summary of the key points, reinforcing the solution's effectiveness and applicability to the question.
5. Give me the whole answer.
6. look into the heading of the data and match the headings along with the name of the cabinet.
Ensure each step is concise yet thorough, facilitating easy understanding and implementation.

Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    model_name="BAAI/bge-large-en"
    model_kwargs={'device':'cpu'}
    encode_kwargs={'normalize_embeddings':True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
    db = Chroma(persist_directory="DB_DIR", embedding_function = embeddings, client_settings=client_settings)
    retriever = db.as_retriever()
    llm_chain_1=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,chain_type='stuff',verbose=True,memory=memory,combine_docs_chain_kwargs={"prompt":prompt},get_chat_history=lambda  h: h)
    return llm_chain_1

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['answer']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data 
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/Ratnam98'>Aman Ratnam </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2= st.columns([1,2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            if user_input:
                answer = process_answer({'question': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            if st.session_state["generated"]:
                display_conversation(st.session_state)
        
if __name__ == "__main__":
    main()