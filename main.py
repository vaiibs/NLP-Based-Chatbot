import streamlit as st
import json
import numpy as np
from tensorflow import keras
import pickle
import random

# Load the intents file
with open("intents.json") as file:
    data = json.load(file)

# Initialize model and components only once
if 'model' not in st.session_state:
    st.session_state.model = keras.models.load_model('chat_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        st.session_state.tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        st.session_state.lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

# Streamlit interface
st.title("Chatbot Interface")

# Initialize conversation history in session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Function to predict the response
def get_bot_response(user_input):
    # Predict the tag for the input
    result = st.session_state.model.predict(keras.preprocessing.sequence.pad_sequences(
        st.session_state.tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))

    # Get the predicted tag
    tag = st.session_state.lbl_encoder.inverse_transform([np.argmax(result)])

    found_response = False
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            found_response = True
            return response

    # If no response is found
    if not found_response:
        return "Sorry, I didn't understand that."

# Display conversation history
for msg in st.session_state.conversation:
    if msg['role'] == 'user':
        st.write(f"**You:** {msg['message']}")
    else:
        st.write(f"**ChatBot:** {msg['message']}")

# Text input from user
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""  # Initialize session state for user input

user_input = st.text_area("You: ", value=st.session_state.user_input, height=100)
send_button = st.button('Send')

# If the user input is not empty and button is clicked, continue the conversation
if send_button and user_input:
    if user_input.lower() in ["quit", "exit"]:
        st.write("**ChatBot:** Goodbye! Chat session ended.")
        st.session_state.conversation.clear()  # Clear the conversation history
        st.session_state.user_input = ""  # Reset user input after ending
    else:
        response = get_bot_response(user_input)
        st.session_state.conversation.append({"role": "user", "message": user_input})
        st.session_state.conversation.append({"role": "chatbot", "message": response})
        
        # Clear the input field after message is sent
        st.session_state.user_input = ""  # Set user_input to an empty string
        st.rerun()  # Rerun to reset input field
