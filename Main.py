import os
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
import openai
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

from langchain import PromptTemplate, LLMChain
#from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space

from pix2tex.cli import LatexOCR
from PIL import Image

# To run the WebUI , type the following in the terminal
# cd "C:\Users\Admin\Desktop\Python Projects\AI model"
# streamlit run main.py

api_key = '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'
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

embeddings = OpenAIEmbeddings()
GeoF1db = FAISS.load_local("Embeddings/Secondary/Geography", embeddings, index_name="GeografiF1")
SejF1db = FAISS.load_local("Embeddings/Secondary/Sejarah", embeddings, index_name="SejarahF1")

from streamlit_extras.app_logo import add_logo

#sidebar
with st.sidebar:
    #add_logo("gallery/TitanMathLogo.png", height=300) # cannot add logo on local disk
    st.title('👨‍🏫 AI Personal Tutor ')
    st.markdown('''
    ## About
    This app is an AI Peronal Tutor that answers any questions regarding the following subjects.
    ''')
    add_vertical_space(1)

    # Initialize state variables for subject and level
    st.session_state['subject'] = []
    st.session_state['level'] = []

    # Radio button to select subject and level
    st.header("Select Subject :")
    st.session_state['subject'] = st.radio(" ",
        ('Math', 'Science', 'Sejarah', 'Geography', 'English', 'BM'))
    
    st.header("Select Level :")
    st.session_state['level'] = st.selectbox(" ",
        ('Form 1', 'Form 2', 'Form 3'))
       
    
    # Select which vectordb to used based on subjects and level chosen
    if st.session_state['subject'] == 'Sejarah' and st.session_state['level'] == 'Form 1':
        currentdb = SejF1db
    elif st.session_state['subject'] == 'Geography' and st.session_state['level'] == 'Form 1':
        currentdb = GeoF1db
    else:
        currentdb = GeoF1db

    add_vertical_space(1)
    if st.button('Clear Chat'):
        st.session_state['generated'] = []
        st.session_state['past'] = []
    
    # st.write("Current subject " + subject)

    # Copyright
    add_vertical_space(5)
    st.write('Made by Titan Math')

def main():
    st.header("Your Personal AI Tutor 🤖")

    # Load embeddings
    # embeddings = OpenAIEmbeddings()
    # db = FAISS.load_local("faiss_index", embeddings)

    #if 'generated' not in st.session_state:
    #    st.session_state['subject'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Setting up chat memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_history = []

    # Accept user question
    query = st.text_input("Ask question:") 
    #st.write(query)

    # input a question via camera
    #if st.session_state['subject'] == 'Math':
        #camera_image_input = st.camera_input("Take the equation: ")
        #img = Image.open('Equation_Image_Sample.png')
        #model = LatexOCR()
        #equation = model(img)
    #elif st.session_state['subject'] != 'Math':
        #camera_image_input = st.camera_input("Take a picture of your question: ")
        #img = Image.open('Equation_Image_Sample.png')
        #model = LatexOCR()
        #equation = model(img)

    prompt_template  = """You are a helpful tutor that answers questions. he following is a 
    friendly conversation between a human and an AI. The AI is talkative and provides lots 
    of specific details from its context. If the AI does not know the answer to a question, 
    it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.
    
    Relevant Information:
    {history}
    
    Human:{question}. 
    AI: translates English to Malay."""

    PROMPT = PromptTemplate(
    input_variables=["history", "question"], template=prompt_template 
)

    if query:
        # Search similarity from vector db indexes
        docs = currentdb.similarity_search(query=query, k=3)
        #st.write(query)   

        chain_type_kwargs = {"prompt": PROMPT}
        llm = OpenAI(model_name='gpt-3.5-turbo',temperature=0)
        qa = ConversationalRetrievalChain.from_llm(
        #qa = RetrievalQA.from_chain_type(
            llm,  
            currentdb.as_retriever(), 
            memory=memory,
            #chain_type_kwargs=chain_type_kwargs
        )
       
        # implement search distance
        vectordbkwargs = {"search_distance": 0.9}

        # Perform Query
        with get_openai_callback() as cb:
            result = qa({"question": query, "chat_history": chat_history}) #, "vectordbkwargs": vectordbkwargs})
            print(cb)
            #st.latex(r'''{\frac{12y+18y^{2}}{3z^{2}-15z}}\times{\frac{z}{y}}''')

        #st.write(result["answer"])
        #message(result["answer"])

        st.session_state.past.append(query)
        st.session_state.generated.append(result["answer"])

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i),avatar_style='bottts',seed=13 )
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')

    # Store the messages
    # history = ChatMessageHistory()
    # dicts = messages_to_dict(history.messages)

if __name__ =='__main__':
    main()