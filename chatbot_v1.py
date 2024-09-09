# a bot that answers questions about the given case reports.
import os
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document

import streamlit as st

# Custom Text Document Loader
class TextLoader:
    def __init__(self, filepath=None):
        self.filepath = filepath

    def loadTxt(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise FileNotFoundError('File not found at', self.filepath)

        # Returning a Document object
        return Document(
            page_content=text,
            metadata={"source": self.filepath}
        )

    @staticmethod
    def loadDir(dirpath):
        documents = []
        for f in os.listdir(dirpath):
            if f.endswith('.txt'):
                fpath = os.path.join(dirpath, f)

                ob = TextLoader(fpath)
                documents.append(ob.loadTxt())
        return documents
# set up langchain backend
chat_model = ChatGroq(
    model='llama3-8b-8192',
    temperature=0.2,
    api_key='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'
)
# Set up function to load data


@st.cache_resource
def load_dir(fpath):
   # loaders = [DirectoryLoader(fpath)]
    docs = TextLoader().loadDir('/Users/samyucktha/PycharmProjects/one/pgms/data')
    index = VectorstoreIndexCreator(
        embedding=JinaEmbeddings(jina_api_key='jina_97335e7338a24fcbbbad954f2bb4b94bMgV1uZhMhUoPm1FzqHekUbIz2I7e', model_name='jina-clip-v1'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_documents(docs)
    return index


vecdb = load_dir('/Users/samyucktha/PycharmProjects/one/pgms/data')

# Creating a QnA chain?
qna = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type='stuff',
    retriever=vecdb.vectorstore.as_retriever(),
    input_key='question'
)

# set up app
st.title('AskMe')

# session state message variable to hold old messages:
if 'messages' not in st.session_state:
    st.session_state.messages = []

#diplay chat history:
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Enter your query here')

chain = chat_model | StrOutputParser()

# if user hits enter then
if prompt:

    # Display prompt
    st.chat_message('user').markdown(prompt)

    # store user prompt in state:
    st.session_state.messages.append({ 'role':'user', 'content':prompt})

    # generate response with chat_model
    response = qna.run(prompt)
    st.chat_message('bot').markdown(response)
    st.session_state.messages.append({'role':'bot', 'content':response})
