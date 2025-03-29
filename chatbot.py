import streamlit as st
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load a more coherent model
MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

chatbot = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  
)

def get_response(user_input):
    try:
        # Generate coherent responses
        inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)
        
        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ✅ Streamlit UI Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="wide")

# ✅ Custom CSS for Styling (No Changes)
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
            margin: 0;
            padding: 0;
        }
        .title-container {
            text-align: center;
            padding-top: 10px;
            padding-bottom: 5px;
        }
        .title {
            font-size: 30px;
            font-weight: bold;
            color: #0078FF;
        }
        .chat-container {
            width: 80%;
            max-width: 700px;
            margin: auto;
            padding: 0px;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        }
        .user-message, .bot-message {
            border-radius: 20px;
            padding: 10px 15px;
            margin: 5px;
            width: fit-content;
            max-width: 75%;
        }
        .user-message {
            background-color: #0078FF;
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }
        .bot-message {
            background-color: #e6e6e6;
            color: black;
            align-self: flex-start;
            margin-right: auto;
        }
        .stChatInput input {
            border-radius: 25px;
            border: 2px solid #0078FF;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-container'><div class='title'>🤖 AI Chatbot</div></div>", unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# ✅ Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ✅ User input handling
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
    
    # ✅ Generate response with a delay for better UI experience
    with st.spinner("🤖 Thinking..."):
        time.sleep(1)  # Artificial delay for effect
        response = get_response(prompt)

    # ✅ Display bot response
    st.markdown(f"<div class='bot-message'>{response}</div>", unsafe_allow_html=True)
    
    # ✅ Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
