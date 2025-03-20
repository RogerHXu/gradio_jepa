from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv
import torch
import gradio as gr
import os
import shutil
import jepa_classifier
import yaml
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

load_dotenv()
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv('NVIDIA_LLAMA_API_KEY')
)

def video_saver(video_path):
    if video_path is None:
        return "No video provided"
    os.makedirs("dataset", exist_ok=True)
    save_path = os.path.join("dataset", "input_video.mp4")
    shutil.copy(video_path, save_path)
    return "video saved successfully"


#create an api call nvidia hosted llama 3.3 LLM
def llm_api_request(robot_spec, action):
    prompt = f'A robot has the following specs:{robot_spec}, and it has just observed this action: {action}. Is it in a safe situation? \
        What would be the next course of action the robot should take?'

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=300,
        stream=True
    )
    strs = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            strs += chunk.choices[0].delta.content
    return strs

def generate_response(text_input, vinput):
    save_msg = video_saver(vinput)
    print(save_msg)
    #output = llm_api_request(text_input, "hello")
    return load_jepa_predictor("config_and_stuff/vith16_k400_16x8x3.yaml")

#create Gradio interface
def create_interface():
    vid = gr.Video(label="Video Source")
    text_input=gr.Textbox(lines=3, placeholder="Robot specifications")
    interface = gr.Interface(
        fn=generate_response,
        inputs=[text_input, vid],
        outputs="text",
        title="V-JEPA with Llama 3.3",
        description="Assess situation and robot safety"
    )
    interface.launch(share=True, debug=True)

def load_jepa_predictor(fname):
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    predicted_label, confidence_score = jepa_classifier.main(params)
    print("AHHHHHHHHHHHHHHHHHHHHHH Predicted Label:", predicted_label)
    return predicted_label, confidence_score

if __name__ == "__main__":
    create_interface()


