import streamlit as st
import time
from transformers import pipeline, AutoTokenizer

# Load a CPU-optimized model
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
chatbot = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device="cpu"  # Force CPU usage
)

def get_response(user_input):
    try:
        # Generate response directly from text input
        response = chatbot(
            user_input + tokenizer.eos_token,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Extract only the bot's response
        return response[0]['generated_text'].split(user_input)[-1].strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        /* Full Page Background */
        body {
            background-color: #f7f9fc;
            margin: 0;
            padding: 0;
        }

        /* Centering the Title */
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

        /* Chat Container */
        .chat-container {
            width: 80%;
            max-width: 700px;
            margin: auto;
            padding: 0px;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Chat Messages */
        .user-message, .bot-message {
            border-radius: 20px;
            padding: 10px 15px;
            margin: 5px;
            width: fit-content;
            max-width: 75%;
        }

        /* User Message (Right-Aligned) */
        .user-message {
            background-color: #0078FF;
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }

        /* Bot Message (Left-Aligned) */
        .bot-message {
            background-color: #e6e6e6;
            color: black;
            align-self: flex-start;
            margin-right: auto;
        }

        /* Chat Input */
        .stChatInput input {
            border-radius: 25px;
            border: 2px solid #0078FF;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Centered Title
st.markdown("<div class='title-container'><div class='title'>🤖 AI Chatbot</div></div>", unsafe_allow_html=True)

# Chat Container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# User input handling
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
    
    # Generate response with a delay for better UI experience
    with st.spinner("🤖 Thinking..."):
        time.sleep(1)  # Artificial delay for effect
        response = get_response(prompt)

    # Display bot response
    st.markdown(f"<div class='bot-message'>{response}</div>", unsafe_allow_html=True)
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
