import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


model = ResNet50(weights='imagenet')


def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    predictions = model.predict(img_expanded)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions


st.set_page_config(page_title="Image Recognition",
                   page_icon=":camera_with_flash:",
                   layout='wide')

st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: linear-gradient(to bottom, #ff7f00, #ff00ff);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center;'>Image Recognition &#128247;</h1>
    """,
    unsafe_allow_html=True
)



st.sidebar.title("Navigation :sparkles:")
page = st.sidebar.selectbox("Select a page 	:page_with_curl:", ["Home", "About"])


if page == "Home":
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image",width=300)

        
        if st.button("Predict 	:mag:"):
            with st.spinner("Predicting..."):
                try:
                    predictions = predict_image(image)
                    st.success("Prediction completed! :white_check_mark:")
                    
                    
                    st.subheader("Predictions:")
                    for pred in predictions:
                        label = pred[1]
                        confidence = pred[2]
                        st.write(f"<h3>{label}</h3>", unsafe_allow_html=True)
                        st.write(f"<p style='font-size: 14px;'>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
                except:
                    st.write("CAN NOT PREDICT")


elif page == "About":
    

    introduction = """
    # **Image Recognition System**

    ### About Us \
    
    Welcome to our image recognition web application! We have developed this platform to provide users with a seamless and intuitive experience in recognizing objects within uploaded images. Powered by advanced deep learning techniques and the ResNet50 model, our system offers accurate and real-time predictions. Simply upload an image of your choice, and our AI algorithms will process it, identifying objects and providing confidence scores for each prediction. Our web-based application features a user-friendly interface, ensuring easy navigation and efficient interaction. With a commitment to delivering high-quality results, we strive to empower users with valuable insights and enhance their understanding of the visual world. Explore our application and unlock the potential of AI in image recognition.

    ### What is Image Recognition?

    As the name suggests, image recognition is the ability of software or a computer system to recognize people, objects, places, and actions in an image. It uses artificial intelligence and a trained set of algorithms to identify the process and analyze the content of an image.

    We humans are blessed with excellent vision processing abilities. It does not take us much effort to differentiate between a cat and a car. However, this process is quite different for computers. They require deep machine learning algorithms to perform the same task. The algorithms are designed based on counting the number of pixels, identifying borders of objects by measuring shades of colors, and estimating spatial relation between different elements.

    Considering the latest advancements in the field of machine learning and the growing potential of computer vision, image recognition has taken the world by storm. The technology is highly versatile and has opened up a whole new dimension of possibilities for businesses and marketers.

    It is being used to perform a wide variety of machine-oriented visual tasks like guiding autonomous robots, labeling image content, performing an image-related search, setting accident avoidance systems, and designing self-driving cars, etc.

    A basic image recognition algorithm includes the following:

    - Optical Character Recognition
    - Patten and Gradient Matching
    - Face Recognition
    - License Plate Matching
    - Scene Identification

    ### The Early Days of Computer Vision

    Computer vision was recognized as a field in the 1960s with an aim to mimic human vision. It was an effort to ask computers what they see and how they see using the process of image analysis. This technology is the predecessor of the artificially intelligent image recognition system.

    Before the technology got automated and artificially intelligent, all kind of image analysis ranging from MRIs, x-rays, and high-resolution space photography was done manually.

    As the field evolved, algorithms kept on getting more intelligent and started solving individual challenges. Over time, they got better at this job by repeating the same task numerous times.

    Fast forward to the current technology, deep learning techniques have made a lot of progress. They are now able to program supercomputers in a way that they can train themselves, make improvements over time, and offer their capabilities to online applications like cloud-based apps.

    ### How Image Recognition Works

    The world of computers consists of numbers. They see every image as a 2-d array of numbers, which is called pixels. Now, in order to teach computer algorithms to recognize objects in an image, different techniques have come into existence. However, the most popular one is to use Convolutional Neural Networks to filter images from a series of artificial neuron layers.

    The neural networks were designed explicitly for the purpose of image processing and image recognition. But, the Convolutional Neural Network has a bit different architecture than a simple Neural Network.

    A regular neural network processes input by passing it through different series of layers. Every layer consists of a set of neurons that are connected to all the neurons in the layer before. Then comes the final-connected layer – the output layer – that shows predictions.

    On the other hand, in a Convolutional Neural Network, the layers are set up in three different dimensions: width, height, and depth. Further, all the neurons in each layer are not connected to each other but just a small region of them. In the end, the output only contains a single probability vector score, which is organized along the depth dimension.


    """
    st.markdown(introduction)
