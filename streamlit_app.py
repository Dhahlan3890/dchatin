import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("suayptalha/FastLlama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("suayptalha/FastLlama-3.2-1B-Instruct")
    return tokenizer, model

tokenizer, model = load_model()

# Chatbot interface
def generate_response(user_input, tokenizer, model, max_length=512):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("FastLlama Chatbot")
st.write("A chatbot powered by FastLlama-3.2-1B-Instruct model.")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
with st.form(key="chat_form"):
    user_input = st.text_input("You:", value="", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Display user input in chat
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Generating response..."):
        bot_response = generate_response(user_input, tokenizer, model)
        st.session_state["messages"].append({"role": "bot", "content": bot_response})

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state["messages"] = []
