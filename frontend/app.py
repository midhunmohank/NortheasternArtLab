import streamlit as st
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

STABLE_DIFFUSION_API_KEY = st.secrets["STABLE_DIFFUSION_API_KEY"]

st.set_page_config(page_title="Northeastern Art Lab", page_icon="🎨", layout="wide")

def load_prompter():
  prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

def generate(plain_text):
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=3, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = [output_text.replace(plain_text+" Rephrase:", "").strip() for output_text in output_texts]
    return res


 
def generate_image(prompt):
    url = 'https://stablediffusionapi.com/api/v3/text2img'
    payload = {
        "key": STABLE_DIFFUSION_API_KEY,
        "prompt": prompt,
        "negative_prompt": "",
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "20",
        "seed": None,
        "guidance_scale": 7.5,
        "safety_checker": "yes",
        "webhook": None,
        "track_id": None
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=payload)
    response = response.json()
    if response['status'] == 'success':
        return response['output'][0]
    else:
        return None
    
def main():
    st.header("Northeastern Art Lab")

    prompt = st.text_input("Enter a prompt")
    if prompt:
        st.subheader("Optimized Prompts")
        rephrased = generate(prompt)
        for r in list(range(len(rephrased))):
            st.button(rephrased[r], key=f"rephrased_{r}")

        if st.session_state.rephrased_0:
            st.subheader("Generated Image")
            image = generate_image(rephrased[0])
            if image:
                st.image(image)
            else:
                st.write("No image generated")
        
        if st.session_state.rephrased_1:
            st.subheader("Generated Image")
            image = generate_image(rephrased[1])
            if image:
                st.image(image)
            else:
                st.write("No image generated")

        if st.session_state.rephrased_2:
            st.subheader("Generated Image")
            image = generate_image(rephrased[2])
            if image:
                st.image(image)
            else:
                st.write("No image generated")

    


if __name__ == "__main__":
    main()





