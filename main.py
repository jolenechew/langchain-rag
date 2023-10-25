from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain


import streamlit as st
from streamlit_chat import message
from utils import *
from indexing import *
from dotenv import load_dotenv
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"] = "Chatbot"

load_dotenv()

# get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


st.title("Migrant Workers AI Bot")
chat_history = []

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )


llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 openai_api_key=OPENAI_API_KEY, temperature=0)

# retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 5})
retriever = vectordb.as_retriever(
    search_type="mmr", search_kwargs={"k": 10})


# Define template prompt
template = """Use the following pieces of context to answer the question at the end. Always say Thanks for asking at the end of the answer.
{context}
Question: {question}
Helpful Answer in english:"""

prompt_template = PromptTemplate.from_template(template)

# Execute chain
qa = ConversationalRetrievalChain.from_llm(
    llm,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
    verbose=True,
    memory=st.session_state.buffer_memory,
)

# conversation = ConversationChain(
#     memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# container for chat history
response_container = st.container()
# container for text box (user input)
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input", value="")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            # refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            # context = find_match(refined_query)
            # print(context)
            result = qa({"question": query, "chat_history": chat_history})
            chat_history.extend([(query, result["answer"])])
            response = result["answer"]
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):

            # display chat response using steamlit message function
            message(st.session_state['responses'][i], key=str(i))

            # to see conveersation flow
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i],
                        is_user=True, key=str(i) + '_user')
