import os
import json
from typing import List
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
import pyttsx3
import queue

load_dotenv()

# Paths
DATA_PATH = "data/cleaned_knowledge_base.json"
FAISS_INDEX_PATH = "faiss_index"
PERSONALITY_PATH = "data/personality_rules.json"
GUARDIAN_INFO_PATH = "data/guardian_info.json"

# Guardrail keywords and aliases
SAFE_KEYWORDS = [
    "baby", "birth", "child", "sleep", "dream", "ear", "eye", "brain",
    "body", "heart", "health", "water", "sky", "sun", "moon", "earth",
    "wind", "plant", "animal", "bird", "fish", "weather", "day", "night",
    "stars", "bones", "food", "teeth", "color", "light", "dark", "eyesight",
    "smell", "touch", "hear", "sound", "laugh", "cry", "family", "mom",
    "dad", "school", "friend", "kindness", "honesty", "gratitude",
    "manners", "duas", "Islam", "Allah", "Prophet", "Quran",
    "hygiene", "clean", "cleanliness", "wash", "bath", "soap", "toothbrush",
    "teeth", "toilet", "wudu", "ablution", "tidy", "germs", "sanitary",
    "sneeze", "cough", "handwashing", "grooming", "shower",
    "respect", "obedience", "truth", "mercy", "patience", "forgiveness",
    "humility", "helping", "sharing", "eyes", "ears", "nose", "mouth",
    "hand", "skin", "hair", "feelings", "sad", "happy", "angry", "fear",
    "love", "mountains", "sea", "river", "ocean", "rain", "tree", "flower",
    "fruit", "seed", "cat", "dog", "camel", "horse", "bee", "ant", "butterfly",
    "drink", "milk", "eat", "clothes", "shoes", "home", "book", "toy",
    "travel", "parent", "brother", "sister", "teacher", "neighbour",
    "guest", "grandma", "grandpa", "alphabet", "numbers", "shape",
    "time", "why", "how", "what", "wazu", "duwa", "namaz", "janamaz",
    "guardian", "phone", "emergency", "contact"
]

# Helpers
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)  # Speed (optional)
    engine.say(text)
    engine.runAndWait()
def normalize_query(query):
    return query.strip().lower()

def fuzzy_match_keywords(query):
    match = process.extractOne(query, SAFE_KEYWORDS, scorer=fuzz.partial_ratio)
    if match:
        best, score, _ = match
        return best if score >= 80 else None
    return None

def is_question_safe(query):
    return fuzzy_match_keywords(query) is not None

def filter_documents_by_topic(docs: List[Document]) -> List[Document]:
    return [doc for doc in docs if any(kw in doc.page_content.lower() for kw in SAFE_KEYWORDS)]

def fallback_keyword_search(query: str, full_docs: List[Document]) -> List[Document]:
    return [doc for doc in full_docs if query.lower() in doc.page_content.lower()]

def call_openai_api(query: str, docs: List[Document], prompt_prefix: str) -> str:
    context_texts = "\n\n".join([doc.page_content for doc in docs])
    debug_log = f"\n--- FINAL PROMPT TO OPENAI ---\n{prompt_prefix}Only use the context below to answer the question. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context_texts}\n\nQuestion: {query}\nAnswer:"
    st.code(debug_log, language="markdown")  # Display the exact prompt
    llm = ChatOpenAI(temperature=0)
    return llm.invoke(debug_log).content
def load_json_documents(filepath: str) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    docs = []
    for item in raw_data:
        content_parts = []
        for k, v in item.items():
            if k not in ["category", "audio"]:  # skip raw audio from content
                content_parts.append(f"{k.capitalize()}: {v}")
        content = "\n".join(content_parts)
        metadata = {
            "category": item.get("category", "unknown"),
            "audio": item.get("audio", ""),
            "arabic": item.get("arabic", ""),
            "usage": item.get("usage", ""),
            "translation": item.get("translation", ""),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_guardian_info():
    if not os.path.exists(GUARDIAN_INFO_PATH):
        return {}
    with open(GUARDIAN_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_guardian_response(query: str, guardian_info: dict) -> str:
    for key in ["mom", "dad", "guardian", "emergency", "phone", "contact", "father", "mother", "parent"]:
        if key in query:
            return f"""
**👨‍👩‍👧 Guardian Info:**

- **Primary Guardian:** {guardian_info.get('guardian_name', 'N/A')} ({guardian_info.get('relationship', 'N/A')})
- **Phone:** {guardian_info.get('phone', 'N/A')}

**📞 Emergency Contact:**

- **Name:** {guardian_info.get('emergency_contact', {}).get('name', 'N/A')} ({guardian_info.get('emergency_contact', {}).get('relationship', 'N/A')})
- **Phone:** {guardian_info.get('emergency_contact', {}).get('phone', 'N/A')}

**🏡 Address:** {guardian_info.get('address', 'N/A')}
"""
    return ""

# FAISS Setup

def create_faiss_index():
    documents = load_json_documents(DATA_PATH)
    st.write("🔧 Creating embeddings and FAISS index...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    return db

def load_faiss_index():
    st.write("📂 Loading FAISS index from disk...")
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def load_personality():
    st.write("🧠 Loading personality rules...")
    with open(PERSONALITY_PATH, "r", encoding="utf-8") as f:
        personality = json.load(f)
    rules = "\n".join(personality.get("response_rules", []))
    style = personality.get("teaching_style", "")
    tone = personality.get("tone", "")
    return (
        f"You are a {tone} Islamic AI friend for children. "
        f"Your teaching style: {style}. "
        f"Follow these rules:\n{rules}\n\n"
    )

# Streamlit UI
st.set_page_config(page_title="My Islamic AI Friend", layout="centered")
st.title("🕌 My Islamic AI Companion")
st.write("Ask me anything about Islam, daily duas, or general knowledge!")

user_question = st.text_input("What would you like to know?")

if not os.path.exists(FAISS_INDEX_PATH):
    st.info("Creating AI knowledge base...")
    db = create_faiss_index()
else:
    db = load_faiss_index()

full_doc_list = load_json_documents(DATA_PATH)
custom_prompt_prefix = load_personality()
guardian_info = load_guardian_info()

if user_question:
    query = normalize_query(user_question)
    if not is_question_safe(query):
        st.warning("This question might be out of scope for this educational app.")
    else:
        guardian_answer = get_guardian_response(query, guardian_info)
        if guardian_answer:
            st.subheader("📞 Emergency Info")
            st.write(guardian_answer)
        else:
            retrieved_docs = db.similarity_search(query, k=2)
            filtered_docs = filter_documents_by_topic(retrieved_docs)
            if not filtered_docs:
                st.info("No direct match found in vector search. Using fallback keyword match...")
                filtered_docs = fallback_keyword_search(query, full_doc_list)
            if not filtered_docs:
                final_response = "I don't know."
            else:
                final_response = call_openai_api(query, filtered_docs, custom_prompt_prefix)
            st.subheader("📘 Answer:")
            # speak(final_response)
            st.write(final_response)
            st.subheader("📚 Retrieved Context")
            for i, doc in enumerate(filtered_docs, 1):
                st.markdown(f"**Result {i} | Category:** {doc.metadata.get('category', 'unknown')}")
                st.code(doc.page_content, language="markdown")
    if filtered_docs:
        try:
            audio_path = None
            for doc in filtered_docs:
                if "audio" in doc.metadata:
                    audio_path = doc.metadata["audio"]
                    arabic_text = doc.metadata["arabic"]
                    # st.write(f"Audio file found: {audio_path}")
                else:
                    try:
                        content_dict = {}
                        for line in doc.page_content.split("\n"):
                            if ":" in line:
                                key, value = line.split(":", 1)
                                content_dict[key.strip().lower()] = value.strip()
                        audio_path = content_dict.get("audio", "")
                        st.write(f"Extracted audio path: {audio_path}")
                    except Exception:
                        continue
                if audio_path and os.path.exists(audio_path):
                    break  # Found the first audio, stop looping

            # Play the audio if found
            if audio_path and os.path.exists(audio_path):
                st.subheader("🔊 Listen to the Dua")
                st.header("Dua in arabic:")
                st.write(arabic_text)
                st.audio(audio_path)
        except Exception as e:
                st.warning("Audio unavailable or format issue.")