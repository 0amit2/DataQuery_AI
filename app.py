import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# title
st.title('DataQuery AI')

# input
query=st.text_input('Ask your Query!')



# Retrive LLM model using gemini
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.7,  
    max_tokens=None,
)

# Load the pdf
pdf_path=r'./Data.pdf'
loader=PyPDFLoader(pdf_path)
documents=loader.load()

# Split into Text
splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
docs=splitter.split_documents(documents)

# Divide text into Embeddings and Store in vector database
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectordb=Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
vectordb.persist()

retriever=vectordb.as_retriever(search_kwargs={'k':10})

# Create a PromptTemplate
prompt_template=PromptTemplate(
    template="""
    you are a helpful **Data Scientist**.
    Explain data science concept short & simple way.

    Context rules:
    -Use only the provided CONTEXT chunks from the retriever.
    -Explain in a clear and professional tone.
    -Do not make up information beyond the context.
    -Do not tell As a Data Scientist in answer
    -if the answer is not availabe in the context, say "I don't know".

    CONTEXT:
    {context}

    QUESTION:
    {input}

    Answer as a Data Scientist
    """,
    input_variables=['context','input']
)

# Create Retrival Chain
chain_QA = create_stuff_documents_chain(llm, prompt_template)
Chain = create_retrieval_chain(retriever, chain_QA)

# Output
if st.button("Sumit"):
    chain_QA = create_stuff_documents_chain(llm, prompt_template)
    Chain = create_retrieval_chain(retriever, chain_QA)
    response = Chain.invoke({"input":query})
    st.markdown('Answer')
    st.write(response['answer'])