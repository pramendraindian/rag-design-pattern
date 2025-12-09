import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
# Load environment variables from .env file
load_dotenv() 

# Access variables using os.getenv() or os.environ[]
MODEL_NAME = os.getenv("MODEL_NAME")
secret_key = os.getenv("HF_TOKEN")


# Use the variables in your application
print(f"Model Name: {MODEL_NAME}")
print(f"Secret Key: {secret_key}") 

#Login to Huggin face
login(token=secret_key)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"


# Check if tokenizer and model are already defined to avoid re-loading
# This block is moved here to ensure embed_text is defined before use
# Ideally, cell RyXoBpd6dUum should be executed first.

print("Loading Gemma model")

# 2. Load the model and tokenizer
  #model1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir="E:/ExtendedPrograms/CachedLibs",device_map="auto")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loaded Gemma model")


context = ['Huggin face is provides an interface to connect with llm']
user_query='What is Hugging Face in ai development?'

# 3. Prepare the input
# 4. Construct an enhanced prompt for the LLM
# Combine the user query with the retrieved context
llm_prompt = f"""
Based on the following information, answer the question:

---
Context:
{"\n".join(context)}
---
Question: {user_query}
Answer:"""

print(f"\nEnhanced LLM Prompt:\n{llm_prompt}\n")

# 5. Generate a response using the LLM (model1) with the enhanced prompt

#with torch.no_grad():
inputs = tokenizer(llm_prompt, return_tensors="pt").to(device) # Tokenize and move to device
print (inputs)
outputs = model.generate(
    **inputs,
    max_new_tokens=50, # Increased max_new_tokens for potentially longer answers
    do_sample=True,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id
)


decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nLLM Generated Response:")
print(decoded_output)

def getModel(ip_inputTokens):
    model.generate(
    **ip_inputTokens,
    max_new_tokens=50, # Increased max_new_tokens for potentially longer answers
    do_sample=True,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id
    )
    return model

def getTokenizer(prompt):
    return tokenizer(prompt, return_tensors="pt").to(device) # Tokenize and move to device
    #return tokenizer