from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma

# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import streamlit as st
from streamlit_chat import message
from utils import *
from indexing import *
from dotenv import load_dotenv
import os

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from translate import Translator


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"] = "Chatbot"

analyzer = SentimentIntensityAnalyzer()

load_dotenv()

# get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


st.title("Migrant Workers AI Bot")
chat_history = []

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hello! Ask me anything about migrant workers' medical, dental and mental health converage in Singapore, as well as any barriers that migrant workers face! \n\n হ্যালো! সিঙ্গাপুরে অভিবাসী কর্মীদের চিকিৎসা, ডেন্টাল এবং মানসিক স্বাস্থ্য কনভারেজ, সেইসাথে অভিবাসী শ্রমিকরা যে কোন বাধার সম্মুখীন হয় সে সম্পর্কে আমাকে কিছু জিজ্ঞাসা করুন! \n\n 你好！请向我询问有关新加坡外籍劳工的医疗、牙科和心理健康状况的任何问题, 以及外籍劳工面临的任何障碍！\n\n வணக்கம்! சிங்கப்பூரில் புலம்பெயர்ந்த தொழிலாளர்களின் மருத்துவம், பல் மருத்துவம் மற்றும் மனநல சுகாதாரம் மற்றும் புலம்பெயர்ந்த தொழிலாளர்கள் எதிர்கொள்ளும் தடைகள் பற்றி என்னிடம் எதையும் கேளுங்கள்!"]

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
                 openai_api_key=OPENAI_API_KEY, temperature=0, request_timeout=120, max_tokens=1000)

# retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 5})

retriever = vectordb.as_retriever(
    search_type="mmr", search_kwargs={"k": 10})


# Define template prompt
template = """Use the following pieces of context to answer the question at the end. Always say Thanks for asking at the end of the answer.
{context}
Question: {question}"""

prompt_template = PromptTemplate.from_template(template)

# Execute chain
qa = ConversationalRetrievalChain.from_llm(
    llm,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
    verbose=True,
    memory=st.session_state.buffer_memory
)

# conversation = ConversationChain(
#     memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# container for chat history
response_container = st.container()
# container for text box (user input)
textcontainer = st.container()


with textcontainer:
    query = st.chat_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            
            result = qa({"question": query, "chat_history": chat_history})
            chat_history.extend([(query, result["answer"])])
            response = result["answer"]

            # for language detection & translation
            language = detect(query)
            print(language)
            

            # for sentiment analysis
            vs = analyzer.polarity_scores(query)
            print(vs)
            polarity_score = vs['compound']
            print(polarity_score)
            sentiment_message = ""
            if -0.3 <= polarity_score < -0.1:
                sentiment_message = "\n\nIt seems like you are feeling down and stressed about this matter. Please take a rest and go for a walk in the park or participate in outdoor activities to unwind and feel better! It's important that you prioritize your health and safety, and know that whatever problems you face can be solved! I hope you feel better soon!"
            elif -0.6 <= polarity_score < -0.3:
                sentiment_message = "\n\nIt seems that you are currently feeling very upset, depressed, or stressed about this matter. Please remember to take care of yourself and your mental health, and know that it is okay to feel this way! You can try to take a break from work, go for a walk in the park, or participate in outdoor activities to unwind and feel better! If you need someone to talk to, you can call the Healthserve's hotline at +65 3129 5000. They are an organization dedicated to help migrant workers. I hope you feel better soon!"
            elif polarity_score < -0.6:
                sentiment_message = "\n\nI can see that you're currently feeling really depressed and stressed, and I can only imagine how tough that must be. It's okay to have those feelings, everyone goes through difficult times. I just want you to know that you don't have to face this alone. If you ever want to talk, vent, or just have someone listen, I'm here for you. Remember, it's okay to ask for help and lean on your loved ones during these times. You are not a burden, and your well-being matters. Please take care of yourself, even in the smallest ways, like taking a deep breath or doing something you enjoy. If you need someone to talk to, you can call the Healthserve's hotline at +65 3129 5000. Sending you a warm virtual hug ❤️"
            print(sentiment_message)
            response += sentiment_message

            translator = Translator(to_lang=language)
            translation = translator.translate(response)
            response = translation
            print(response)

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
