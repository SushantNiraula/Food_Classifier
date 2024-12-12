

# import streamlit as st
# from PIL import Image
# import torch
# from transformers import AutoModelForImageClassification, AutoImageProcessor
# from joblib import load
# import pandas as pd
# import os
# from groq import Groq
# from dotenv import load_dotenv
# load_dotenv()

# # Define the InferencePipeline class
# class InferencePipeline:
#     def __init__(self, model_path, model_name):
#         self.model = AutoModelForImageClassification.from_pretrained(model_path)
#         self.image_processor = AutoImageProcessor.from_pretrained(model_name)

#     def predict(self, image):
#         preprocessed_image = self.image_processor(image, return_tensors="pt")["pixel_values"]
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(preprocessed_image)
#             predicted_class = outputs.logits.argmax(-1).item()
#         return predicted_class

# # Load the inference pipeline from joblib
# inference_pipeline = load("inference_pipeline.joblib")

# # Define the class labels (should match what was used during training)
# class_names = ['biryani', 'boiled_egg', 'caesar_salad', 'chatamari', 'chhoila', 'chicken curry', 'chicken wing', 'chocolate cake', 'chow_mein', 'cupcake', 'dalbhat', 'dhindo', 'dumpling', 'egg_roll', 'fried Rice', 'fruitcake', 'gundruk', 'icecream', 'kheer', 'momo', 'nacho', 'ommelete', 'pizza', 'poached egg', 'ramen', 'samosa', 'sandwich', 'sausage_roll', 'scrambled_eggs', 'sekuwa', 'selroti', 'springroll', 'sushi']  # Replace with your actual class labels

# # Function to get nutritional values from Llama 70B using Groq
# def get_nutritional_info(food_item):
#     # Initialize the Groq client
    
#     client = Groq('GROQ_API_KEY')
    
#     # Send the message to the Llama 70B model to get nutritional info
#     completion = client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[{"role": "system", "content": f"Provide the nutritional information for {food_item}. Try to format as a table. Keep the response as short and concise as possible. At last tell a healthy tip."}],
#         temperature=1,
#         max_tokens=1024,
#         top_p=1,
#         stream=True,
#         stop=None,
#     )
    
#     # Process the model's response
#     nutrition_info = ""
#     for chunk in completion:
#         nutrition_info += chunk.choices[0].delta.content or ""
    
#     # Return the nutrition info (assuming it's returned as a string)
#     return nutrition_info

# # Streamlit UI
# st.title("Image Classifier Using Vision Transformer (ViT)")
# st.write("Upload an image to classify it and get nutritional values.")

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Open and display the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

#     # Add a button for prediction
#     if st.button("Classify Image"):
#         # Make a prediction
#         predicted_class_idx = inference_pipeline.predict(image)
#         predicted_class = class_names[predicted_class_idx]  # Map to human-readable label
        
#         st.write(f"Prediction: {predicted_class}")
        
#         # Get nutritional information
#         nutritional_info = get_nutritional_info(predicted_class)
        
#         if nutritional_info:
#             # Display the nutritional information
#             st.write("Nutritional Information:")
#             st.write(nutritional_info)
# else:
#     st.write("Please upload an image to continue.")



import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import openai
from dotenv import load_dotenv

# Retrieve the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Define the InferencePipeline class
class InferencePipeline:
    def __init__(self, model_path, model_name):
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    def predict(self, image):
        preprocessed_image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(preprocessed_image)
            predicted_class = outputs.logits.argmax(-1).item()
        return predicted_class

# Load the inference pipeline from joblib
# (Make sure this file exists and is properly saved)
# inference_pipeline = load("inference_pipeline.joblib")

# Define the class labels (should match what was used during training)
class_names = ['biryani', 'boiled_egg', 'caesar_salad', 'chatamari', 'chhoila', 'chicken curry', 'chicken wing', 'chocolate cake', 'chow_mein', 'cupcake', 'dalbhat', 'dhindo', 'dumpling', 'egg_roll', 'fried Rice', 'fruitcake', 'gundruk', 'icecream', 'kheer', 'momo', 'nacho', 'ommelete', 'pizza', 'poached egg', 'ramen', 'samosa', 'sandwich', 'sausage_roll', 'scrambled_eggs', 'sekuwa', 'selroti', 'springroll', 'sushi']

# Function to get nutritional values using OpenAI's GPT
def get_nutritional_info(food_item):
    try:
        # Send the message to GPT to get nutritional info
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can also try "gpt-4-32k" for larger context window
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Provide the nutritional information for {food_item}. Format it as a table. Include key nutrients like calories, protein, fat, carbohydrates, and fiber."}
            ]
        )

        # Extract response
        nutrition_info = response['choices'][0]['message']['content']
        return nutrition_info

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Image Classifier Using Vision Transformer (ViT)")
st.write("Upload an image to classify it and get nutritional values.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Add a button for prediction
    if st.button("Classify Image"):
        # Make a prediction (use your model for classification)
        predicted_class_idx = inference_pipeline.predict(image)
        predicted_class = class_names[predicted_class_idx]  # Map to human-readable label
        
        st.write(f"Prediction: {predicted_class}")
        
        # Get nutritional information from GPT
        nutritional_info = get_nutritional_info(predicted_class)
        
        if nutritional_info:
            # Display the nutritional information
            st.write("Nutritional Information:")
            st.write(nutritional_info)
else:
    st.write("Please upload an image to continue.")
