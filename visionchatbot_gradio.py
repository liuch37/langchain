import ollama
import gradio as gr
import os
import subprocess
import base64
from io import BytesIO
from PIL import Image

#from langchain_ollama import OllamaLLM

'''
# install ollama on Linux
print("Install ollama......")
download = subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'],
                          check=True, text=True)

# pull ollama
print("Pull LLaVA......")
pull = subprocess.run(['ollama', 'pull', 'llava'], check=True, text=True)
'''

#print("Initialize LLaVA......")
#llm = OllamaLLM(model="llava")

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def find_latestimagefile(history):
    for index in range(len(history) - 1, -1, -1):
        if type(history[index]['content']) == tuple and history[index]['content'][-1].lower()[-4:] in ['.jpg', '.jpeg', '.png']:
            return history[index]['content'][-1]
    return None

def ask(message, history):
    #print(history)
    if len(message["files"]) == 0 and find_latestimagefile(history) == None:
        return "Please upload one image."
    else:
        if message["files"]:
            # use the current image
            file_path = message["files"][-1]['path']
        else:
            # use the latest image
            file_path = find_latestimagefile(history)
        with open(file_path, 'rb') as file:
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
        '''
        # TO BE REFACTORED AND DEBUG
        image = open(message["files"][-1]['path'], "rb").read()
        llm_with_image_context = llm.bind(images=[image])
        response = llm_with_image_context.invoke(message["text"])

        return response
        '''

# lanuch gradio
gr.ChatInterface(
    ask,
    type="messages",
    multimodal=True,
    textbox=gr.MultimodalTextbox(placeholder="Ask me a question regarding your uploaded image.", container=False, scale=7),
    title="Vision Chatbot",
    description="Ask Vision Chatbot any question regarding your uploaded image",
    theme="soft",
    examples=[{"text": "Tell me about the image", "files": ["images/australia.jpg"]},
              {"text": "How many animals are in the image?", "files": ["images/cat_dog.jpg"]}],
).launch(share=True)