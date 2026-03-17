import os
from dotenv import load_dotenv
import gradio as gr
from pathlib import Path
import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re



# import API key of OPENAI 
# the key should be saved as json format 
# e.g. {"OPENAI_API_KEY" : "sk-proj-xxxxxxxx"} in parent folder or anywhere in the workingspace


dir = Path(__file__).resolve().parent
keyFilePath = dir/"openAiKey.json"

with open(keyFilePath, "r",encoding="utf-8" ) as file:
    keyFile= json.load(file)

key = keyFile["OPENAI_API_KEY"]


# set up environment variable for OpenAI key
os.environ["OPENAI_API_KEY"] = key




# Initialize OpenAI Model 
llm = ChatOpenAI(model = "o4-mini")

prompt = PromptTemplate(
    input_variables=["user_input"],
    template = "You are a helpful assistant.\n\nQuestion: {user_input}"

)

# New LangChain pipeline style
chain = prompt | llm



# create function for clearning LLM result 
# e.g. \n(Newline (line break)), \u2009 (Thin space (Unicode)), \u200a (Hair space (Unicode)),  \\ (Escaped backslash), \uxxxx (Unicode character in hex)

def clean_response(text):
    # Make sure it's a string first
    if not isinstance(text, str):
        text = str(text)
    # Replace unwanted characters
    text = text.replace("\u2009", " ").replace("\u200a","  ")
    text = re.sub(r'\n{2,}', '\n\n', text)  # Replace multiple newlines with single newlines 

    return text.strip()


# Function to run LLM 


def ask_llm(user_input):
    result = chain.invoke({"user_input": user_input})
    return clean_response(result.content if hasattr(result, "content") else result)

# set up Gradio UI 

with gr.Blocks() as demo: 
    gr.Markdown("OPEN AI Chat Bot project --by Ivan Chiu ")

    with gr.Row():
        input_box = gr.Textbox(label = "input here", placeholder = "E.g. What is LangChain?", lines= 4)
    with gr.Row():
        submit_btn = gr.Button("Submit")
    
    output_box = gr.Textbox(label = "LLM : OPEN AI CHATGPT", lines= 12)

    submit_btn.click(fn = ask_llm, inputs= input_box, outputs= output_box)

# launch app 
demo.launch()
