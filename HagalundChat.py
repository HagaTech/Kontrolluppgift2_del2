import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
import os
from groq import Groq

load_dotenv()

# Ställ in din API-nyckel för ChatGroq
#os.environ["OPENAI_API_KEY"] = "din-chatgroq-api-nyckel"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client=Groq(api_key=GROQ_API_KEY)

# Ladda och dela upp text från dokumentet
def load_and_split_text(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Skapa en vektorbas för att hitta relevanta svar
def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #return FAISS.from_documents(docs, embeddings)
    return InMemoryVectorStore.from_documents(chunks, embeddings)

# LLM-modell via ChatGroq
def load_chatgroq_llm():
    #return ChatOpenAI(model="llama-3.3-70b-versatile", temperature=0)
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5, api_key= GROQ_API_KEY)

# Ladda restauranginformationen
file_path = "hagalunds_café.txt"
chunks = load_and_split_text(file_path)

vector_db = create_vector_db(chunks)

retriever=vector_db.as_retriever()

# Skapa en QA-kedja med LangChain
qa_chain = RetrievalQA.from_chain_type(llm=load_chatgroq_llm(), retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="Hagalunds Café", page_icon="🌿", layout="wide")
st.markdown("""
<style>
    body {
        background-color: #f8f5ec;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #f8f5ec;
    }
    .stApp {
        background-color: #f8f5ec;
    }
    .menu-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
            
    /* Sidopanelens bakgrund */
    [data-testid="stSidebar"] {
        background-color: #4a7c59;
    }

    /* Justera textfärg i sidopanelen för kontrast */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Styla knappen i sidopanelen */
    button[kind="secondary"] {
        background-color: #4a7c59 !important;
        color: #4a7c59 !important;
        border: 2px solid #ffffff !important;
        font-weight: bold;
        border-radius: 8px;
    }

    /* Hover-effekt för knappen */
    button[kind="secondary"]:hover {
        border: 2px solid #000000 !important;
    }
            
    [data-testid="stSidebar"] input[type="text"] {
    color: black !important;
    background-color: #ffffff !important;
    border: 1px solid #dddddd;
    border-radius: 6px;
    }
                       
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #e9f5ec;
        color: #4a7c59;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    /* Justera bakgrund och färg på flikarna */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f7f4;  /* mjuk bakgrund */
        border-radius: 8px;
        padding: 0.25rem;
        gap: 0.5rem;
    }

    /* Stil på varje flik */
    .stTabs [data-baseweb="tab"] {
        color: #2e4735;
        font-weight: bold;
        background-color: #ffffff;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease-in-out;
        border: 1px solid #cfe7dd;
    }

    /* Aktiv flik */
    .stTabs [aria-selected="true"] {
        background-color: #cfe7dd;  /* aktiv bakgrund - grön ton */
        color: #1e3529;
        border: 2px solid #9fcbb5;
    }            
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Breddar själva sidopanelen */
section[data-testid="stSidebar"] {
    width: 400px !important;
}

/* Gör att main-content flyttas åt höger så det inte överlappar */
section[data-testid="stSidebar"] > div {
    width: 400px !important;
}
</style>
""", unsafe_allow_html=True)

# -------- SIDEBAR: Frågespalt --------
#st.sidebar.image("hagalund_logo.png", width=120)
#view_choice = st.sidebar.selectbox("📋 Välj vad du vill se:", ["🥗 Meny", "📖 Om caféet"])
#with st.sidebar.expander("📋"):
#    view_choice = st.sidebar.radio("", ["🥗 Meny", "📖 Om caféet"])

st.sidebar.markdown("### ❓ Har du en fråga om caféet eller menyn?")
user_question = st.sidebar.text_input("Skriv din fråga här:")

if st.sidebar.button("🍓 Få svar") and user_question:
    with st.sidebar:
        with st.spinner("Vi letar bland våra trädgårdsanteckningar..."):
            answer = qa_chain.run(user_question)
        st.success(answer)

# -------- Header --------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("hagalund_logo.png", width=200)  # Byt ut till din bildfil eller URL
with col_title:
    st.title("🌿 Hagalunds Café")
#with st.expander("📋"):
    #view_choice = st.radio("", ["🥗 Meny", "📖 Om caféet"])
    tabs = st.tabs(["📖 Om caféet", "🥗 Meny", "🍹 Drycker"])
st.divider()

#st.title("Hagalunds Café - Fråga oss!")
#st.write("Ställ en fråga om restaurangen och menyn så svarar vi baserat på vår information.")

#user_question = st.text_input("Din fråga:")

#if st.button("Ställ fråga") and user_question:
#    answer = qa_chain.run(user_question)
#    st.write("### Svar:")
#    st.write(answer)


# -------- Meny --------
#if view_choice == "🥗 Meny":
with tabs[0]:  # Flik 2 - Om caféet
    st.markdown("**Välkommen till Hagalunds Café – en grön oas där vi serverar växtbaserad mat med råvaror direkt från vår egen trädgård. Vi tror på hållbarhet, kvalitet och att njuta av god mat i en naturnära miljö.**")
    st.markdown("Våra rätter är skapade med omsorg och kärlek för både smak och hälsa. Dessutom brygger vi vår egen kombucha med noggrant utvalda smaker. Slå dig ner i vår lummiga trädgård och njut av en stunds avkoppling!")

with tabs[1]:  # Flik 1 - Menyn
    st.markdown("### 🥬 Meny")
    st.markdown("""
    <div class="menu-box">
                
    🥗 **Grönskande Salladsskål** – En fräsch mix av nyskördade bladgrönsaker, grillad zucchini, tomater, örter och rostade pumpafrön. Serveras med en krämig tahinidressing.

    🥕 **Gårdens Grönsakssoppa** – Dagens soppa gjord på säsongens bästa grönsaker från trädgården. Serveras med vårt hembakta surdegsbröd.

    🍆 **Ugnsbakad Aubergine med Hasselnötskräm** – Aubergine rostad till perfektion, toppad med en len hasselnötskräm och örtsallad.

    🌮 **Trädgårdstacos** – Små majs- eller kikärtstortillas fyllda med grillade grönsaker, picklad rödlök och en kryddig bönröra.

    🍄 **Svamp- och Linsbiffar** – Serveras med rostad sötpotatis, vitlökssås och en fräsch örtsallad.

    🥒 **Fermenterad Lycka** – En tallrik med husets egna syrade grönsaker och surkål, perfekt som ett smakrikt tillbehör.

    </div>
    """, unsafe_allow_html=True)

#elif view_choice == "📖 Om caféet":
with tabs[2]:  # Flik 3 - Drycker
    st.markdown("### 🍹 Drycker")
    st.markdown("""
    Hos oss hittar du uppfriskande drycker gjorda med kärlek och omtanke:

    🌿 **Isad Örtinfusion** – Kallbryggda örter från trädgården som mynta, citronmeliss och kamomill – perfekt svalka!

    🍋 **Citronsvalka** – En klassisk lemonad med en twist av gurka och mynta.

    🍏 **Äppelmust** – Färskpressad must från lokala äpplen, serverad iskall.
    
    🍹 **Hagalunds Kombucha** – Vår egenbryggda kombucha, tillgänglig i olika smaker beroende på säsong. Fråga oss om dagens val!

    🍵 **Örtte från Trädgården** – En värmande kopp te med färska örter som mynta, citronmeliss och timjan.

    ☕ **Ekologiskt Kaffe** – Bryggt på noggrant utvalda, rättvisemärkta kaffebönor.

    🌿 **Smoothie på Säsongens Frukter & Bär** – En näringsrik och uppfriskande dryck med bär och grönsaker från vår trädgård.
    """)

st.markdown(
    """
    <div class="footer">
        🌼 Hagalunds Café – Hemlagat med kärlek från trädgården 🌼
    </div>
    """,
    unsafe_allow_html=True
)
