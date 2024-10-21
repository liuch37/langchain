import ollama
import gradio as gr
import os
import subprocess

'''
# install ollama on Linux
print("Install ollama......")
download = subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'],
                          check=True, text=True)

# pull ollama
print("Pull LLaVA......")
pull = subprocess.run(['ollama', 'pull', 'llava'], check=True, text=True)
'''

def ask(message, history):
    if len(message["files"]) == 0:
        return "Please upload an image."
    else:
        # always analyze the last image
        with open(message["files"][-1]['path'], 'rb') as file:
            response = ollama.chat(
                model='llava',
                messages=[
                    {
                        'role': 'user',
                        'content': message["text"],
                        'images': [file.read()],
                    },
                ],
                stream=False,
            )

        return response['message']['content']

# lanuch gradio
gr.ChatInterface(
    ask,
    type="messages",
    multimodal=True,
    textbox=gr.MultimodalTextbox(placeholder="Ask me a question regarding your uploaded image.", container=False, scale=7),
    title="Vision Chatbot",
    description="Ask Vision Chatbot any question regarding your uploaded image",
    theme="soft",
    examples=[{"text": "Tell me about the image", "files": ["/Users/chun-haoliu/Documents/research/australia.jpg"]},
              {"text": "How many animals are in the image?", "files": ["/Users/chun-haoliu/Documents/research/cat.jpg"]}],    
).launch()