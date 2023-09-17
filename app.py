from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os

from htmlTemplates import css, bot_template, user_template
import streamlit as st
from PyPDF2 import PdfFileReader


def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf_doc in pdf_docs:
        with open(pdf_doc, "rb") as f:
            pdf_reader = PdfFileReader(f)
            for page in pdf_reader.pages:
                text += page.extractText()
    return text


def get_ppt_text(ppt_docs):
    text = ""
    for ppt in ppt_docs:
        ppt_reader = ppt.read()
        for slide in ppt_reader.slides:
            for shape in slide.shapes:
                if shape.has_text():
                    text += shape.text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={'device': 'cpu'},
                                               encode_kwargs={'normalize_embeddings': True})

    try:
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    except IndexError:
        print("The chunks list is empty or the first element in the chunks list is not a list of strings.")
        return None

    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm_ = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                          huggingfacehub_api_token="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",
                          model_kwargs={"temperature": 0.5, "max_length": 4096})
    memory = ConversationBufferMemory()
    memory_key = 'chat_history',
    return_messages = True
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Wrap the string value in a list
model_name = ["hkunlp/instructor-xl"]
model_name = model_name[0]

embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})


if __name__ == '__main__':
    # Example usage:

    chunks = ["This is the first sentence.", "This is the second sentence."]
    vectorstore = get_vectorstore(chunks)

    if vectorstore is not None:
        # Use the vectorstore
        pass



def init_session_state():
    if not hasattr(st.session_state, 'init_conversation'):
        st.session_state.init_conversation = False


# Your other code here...

def handle_userinput(user_question):
    # Initialize session_state if it hasn't been initialized
    init_session_state()

    # Check if init_conversation is False, then initialize the conversation
    if not st.session_state.init_conversation:
        # Perform the conversation initialization here
        # ...
        st.session_state.init_conversation = True  # Mark as initialized

    # Rest of your code...


def main():
    load_dotenv()

    st.header("DocsPro :knife:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)
        ppt_docs = st.file_uploader(
            "Upload your PPTs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                pdf_docs = st.file_uploader("Upload PDF documents", type=".pdf", accept_multiple_files=True)
                ppt_docs = st.file_uploader("Upload PPT documents", type=".ppt", accept_multiple_files=True)
                raw_text = get_pdf_text(pdf_docs) + get_ppt_text(ppt_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
else:
    pass
