import streamlit as st
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Replace 'your_huggingface_token' with your actual Hugging Face access token
access_token = os.getenv('token')

# Initialize the tokenizer and model with the Hugging Face access token
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,
    use_auth_token=access_token
)
model.eval()  # Set the model to evaluation mode

# Initialize the inference client (if needed for other API-based tasks)
client = InferenceClient(token=access_token)

def respond(
    message: str,
    history: list[tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate a response for a multi-turn chat conversation."""
    # Prepare the messages in the correct format for the API
    messages = [{"role": "system", "content": system_message}]

    for user_input, assistant_reply in history:
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})

    messages.append({"role": "user", "content": message})

    response = ""

    # Stream response tokens from the chat completion API
    for message_chunk in client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message_chunk["choices"][0]["delta"].get("content", "")
        response += token
        yield response

# Streamlit App
st.title("Chat with Hugging Face Model")

# Sidebar for configuration options
st.sidebar.header("Configuration")
system_message = st.sidebar.text_input("System message", "You are a friendly Chatbot.")
max_tokens = st.sidebar.slider("Max new tokens", 1, 2048, 512, step=1)
temperature = st.sidebar.slider("Temperature", 0.1, 4.0, 0.7, step=0.1)
top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95, step=0.05)

# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
with st.form("chat_form"):
    user_input = st.text_input("Your message", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Append user input to history
    st.session_state.history.append((user_input, ""))

    # Generate response
    placeholder = st.empty()
    response_text = ""
    for response in respond(
        message=user_input,
        history=st.session_state.history,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        response_text = response
        placeholder.text(response_text)

    # Update conversation history with the response
    st.session_state.history[-1] = (user_input, response_text)

# Display conversation history
st.header("Conversation History")
for user_input, assistant_reply in st.session_state.history:
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Chatbot:** {assistant_reply}")
