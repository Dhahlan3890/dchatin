import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Show title and description.
st.title("🦒 Falcon-7B Chatbot")
st.write(
    "This chatbot uses the Falcon-7B model for generating responses. You can have a conversation with the AI below. "
    "This model is optimized for text generation tasks."
)

# Load the model and tokenizer.
@st.cache_resource  # Cache the model and tokenizer to avoid reloading on every run.
def load_model():
    model_name = "tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return tokenizer, text_gen_pipeline

tokenizer, text_gen_pipeline = load_model()

# Create a session state variable to store chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field.
if prompt := st.chat_input("Say something to the Falcon-7B chatbot!"):
    # Add user's message to the session state.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the Falcon-7B model.
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        try:
            # Generate the response with the pipeline.
            sequences = text_gen_pipeline(
                prompt,
                max_length=200,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            response_text = sequences[0]["generated_text"]

            # Display the response.
            response_container.markdown(response_text)

        except Exception as e:
            st.error(f"Error: {e}")

        # Store the assistant's response in the session state.
        st.session_state.messages.append({"role": "assistant", "content": response_text})
