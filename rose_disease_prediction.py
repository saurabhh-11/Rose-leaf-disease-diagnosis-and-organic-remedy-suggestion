import streamlit as st

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Rose Disease Detection",
    page_icon="🌹",
    layout="wide",
    initial_sidebar_state="expanded"
)

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import textwrap
import requests # Added for downloading the model file

# Try importing OpenCV, if not available, use PIL for image processing
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    st.warning("OpenCV not available, using PIL for image processing")

# Constants
MODEL_FILENAME = "custom_cnn_rose_disease_model.h5"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
# Direct URL to the raw model file in your GitHub repository
MODEL_DOWNLOAD_URL = "https://raw.githubusercontent.com/saurabhh-11/Rose-leaf-disease-diagnosis-and-organic-remedy-suggestion/main/custom_cnn_rose_disease_model.h5"

# Language dictionaries
LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी',
    'mr': 'मराठी'
}

UI_TEXT = {
    'en': {
        'title': '🌹 Rose Disease Detection System',
        'subtitle': 'For Farmers and Gardeners',
        'about': 'About',
        'instructions': 'Instructions',
        'tips': 'Tips',
        'upload_image': 'Upload Image',
        'choose_image': 'Choose an image...',
        'predict': '🔍 Predict Disease',
        'prediction_results': 'Prediction Results',
        'info': 'Information',
        'gallery': 'Disease Gallery',
        'common_conditions': 'Common rose conditions and their symptoms:',
        'supported': 'Supported Categories:',
        'tips_content': '• Take photos in good lighting\n• Focus on the affected area\n• Include both healthy and affected parts\n• Keep the image clear and steady',
        'best_practices_content': '• Check multiple leaves if possible\n• Take photos from different angles\n• Ensure the leaf is well-lit\n• Avoid shadows and glare',
        'no_image': 'Please upload an image to continue.',
        'confidence': 'Confidence',
        'desc': 'Description:',
        'recommend': 'Recommendations:',
        'instruction_1': '1. Upload Image',
        'instruction_1_content': '• Select a clear image of the rose leaf\n• Ensure good lighting\n• Focus on the affected area',
        'instruction_2': '2. 🔍 Predict Disease',
        'instruction_2_content': '• Click the predict button\n• Wait for analysis\n• View results',
        'instruction_3': '3. Prediction Results',
        'instruction_3_content': '• Check disease identification\n• Read description\n• Follow recommendations',
        'photo_tips': '📸 Photo Tips',
        'best_practices': '🔍 Best Practices'
    },
    'hi': {
        'title': '🌹 गुलाब रोग पहचान प्रणाली',
        'subtitle': 'किसानों और बागवानों के लिए',
        'about': 'परिचय',
        'instructions': 'निर्देश',
        'tips': 'सुझाव',
        'upload_image': 'छवि अपलोड करें',
        'choose_image': 'एक छवि चुनें...',
        'predict': '🔍 रोग पहचानें',
        'prediction_results': 'परिणाम',
        'info': 'जानकारी',
        'gallery': 'रोग गैलरी',
        'common_conditions': 'गुलाब की सामान्य स्थितियाँ और उनके लक्षण:',
        'supported': 'समर्थित श्रेणियाँ:',
        'tips_content': '• अच्छी रोशनी में फोटो लें\n• प्रभावित क्षेत्र पर ध्यान केंद्रित करें\n• स्वस्थ और प्रभावित दोनों भाग शामिल करें\n• छवि स्पष्ट रखें',
        'best_practices_content': '• यदि संभव हो तो कई पत्तियों की जांच करें\n• विभिन्न कोणों से फोटो लें\n• पत्ती अच्छी तरह से रोशन हो\n• छाया और चकाचौंध से बचें',
        'no_image': 'कृपया आगे बढ़ने के लिए एक छवि अपलोड करें।',
        'confidence': 'विश्वास',
        'desc': 'विवरण:',
        'recommend': 'सिफारिशें:',
        'instruction_1': '1. छवि अपलोड करें',
        'instruction_1_content': '• गुलाब की पत्ती की स्पष्ट छवि चुनें\n• अच्छी रोशनी सुनिश्चित करें\n• प्रभावित क्षेत्र पर ध्यान केंद्रित करें',
        'instruction_2': '2. 🔍 रोग पहचानें',
        'instruction_2_content': '• पहचान बटन पर क्लिक करें\n• विश्लेषण का इंतजार करें\n• परिणाम देखें',
        'instruction_3': '3. पहचान परिणाम',
        'instruction_3_content': '• रोग की पहचान जांचें\n• विवरण पढ़ें\n• सिफारिशों का पालन करें',
        'photo_tips': '📸 फोटो सुझाव',
        'best_practices': '🔍 सर्वोत्तम प्रथाएं'
    },
    'mr': {
        'title': '🌹 गुलाब रोग ओळख प्रणाली',
        'subtitle': 'शेतकरी आणि माळ्यांसाठी',
        'about': 'परिचय',
        'instructions': 'सूचना',
        'tips': 'टीप',
        'upload_image': 'प्रतिमा अपलोड करा',
        'choose_image': 'प्रतिमा निवडा...',
        'predict': '🔍 रोग ओळखा',
        'prediction_results': 'परिणाम',
        'info': 'माहिती',
        'gallery': 'रोग गॅलरी',
        'common_conditions': 'गुलाबाच्या सामान्य समस्या आणि त्यांची लक्षणे:',
        'supported': 'समर्थित श्रेण्या:',
        'tips_content': '• चांगल्या प्रकाशात फोटो घ्या\n• प्रभावित भागावर लक्ष केंद्रित करा\n• निरोगी आणि प्रभावित दोन्ही भाग समाविष्ट करा\n• प्रतिमा स्पष्ट ठेवा',
        'best_practices_content': '• शक्य असल्यास अनेक पाने तपासा\n• वेगवेगळ्या कोनातून फोटो घ्या\n• पान चांगल्या प्रकाशात असल्याची खात्री करा\n• सावली आणि चकाकी टाळा',
        'no_image': 'कृपया पुढे जाण्यासाठी प्रतिमा अपलोड करा.',
        'confidence': 'विश्वास',
        'desc': 'वर्णन:',
        'recommend': 'शिफारसी:',
        'instruction_1': '1. प्रतिमा अपलोड करा',
        'instruction_1_content': '• गुलाबाच्या पानाची स्पष्ट प्रतिमा निवडा\n• चांगला प्रकाश सुनिश्चित करा\n• प्रभावित भागावर लक्ष केंद्रित करा',
        'instruction_2': '2. 🔍 रोग ओळखा',
        'instruction_2_content': '• ओळखण्याच्या बटनावर क्लिक करा\n• विश्लेषणाची वाट पहा\n• परिणाम पहा',
        'instruction_3': '3. ओळखण्याचे परिणाम',
        'instruction_3_content': '• रोगाची ओळख तपासा\n• वर्णन वाचा\n• शिफारसींचे पालन करा',
        'photo_tips': '📸 फोटो टीप',
        'best_practices': '🔍 सर्वोत्तम पद्धती'
    }
}

# Disease info in all languages
DISEASE_INFO = {
    'en': {
        'healthy': {
            'name': 'Healthy',
            'description': 'The rose leaf appears healthy with no signs of disease.',
            'remedy': '• Continue regular maintenance\n• Monitor for any changes\n• Maintain proper watering schedule\n• Keep good air circulation',
            'icon': '✅'
        },
        'downy_mildew': {
            'name': 'Downy Mildew',
            'description': 'Downy mildew is a fungal disease that appears as yellow patches on leaves with grayish-white mold underneath.',
            'remedy': '• Apply neem oil spray\n• Use baking soda solution (1 tbsp per gallon of water)\n• Improve air circulation\n• Remove infected leaves',
            'icon': '🍄'
        },
        'powdery_mildew': {
            'name': 'Powdery Mildew',
            'description': 'Powdery mildew appears as white powdery spots on leaves and stems.',
            'remedy': '• Spray with milk solution (1 part milk to 9 parts water)\n• Apply neem oil\n• Use baking soda spray\n• Prune affected areas',
            'icon': '❄️'
        },
        'black_spot': {
            'name': 'Black Spot',
            'description': 'Black spot causes black spots with yellow halos on leaves, leading to defoliation.',
            'remedy': '• Apply neem oil\n• Use baking soda solution\n• Remove infected leaves\n• Improve air circulation',
            'icon': '⚫'
        },
        'rose_slug': {
            'name': 'Rose Slug',
            'description': 'Rose slugs are sawfly larvae that skeletonize leaves.',
            'remedy': '• Handpick larvae\n• Apply neem oil\n• Use insecticidal soap\n• Encourage natural predators',
            'icon': '🐛'
        },
        'rose_mosaic': {
            'name': 'Rose Mosaic',
            'description': 'Rose mosaic virus causes yellow patterns on leaves.',
            'remedy': '• Remove infected plants\n• Use virus-free planting material\n• Maintain plant health\n• Control aphids',
            'icon': '🎨'
        },
        'rose_rust': {
            'name': 'Rose Rust',
            'description': 'Rose rust appears as orange powdery spots on leaves and stems.',
            'remedy': '• Apply neem oil\n• Use sulfur-based fungicide\n• Remove infected leaves\n• Improve air circulation',
            'icon': '🟠'
        }
    },
    'hi': {
        'healthy': {
            'name': 'स्वस्थ',
            'description': 'गुलाब की पत्ती स्वस्थ है और किसी रोग का कोई संकेत नहीं है।',
            'remedy': '• नियमित देखभाल जारी रखें\n• किसी भी बदलाव की निगरानी करें\n• उचित सिंचाई बनाए रखें\n• अच्छा वायु संचार रखें',
            'icon': '✅'
        },
        'downy_mildew': {
            'name': 'डाउनी मिल्ड्यू',
            'description': 'डाउनी मिल्ड्यू एक फफूंदजनित रोग है जो पत्तियों पर पीले धब्बों के रूप में दिखाई देता है, जिनके नीचे ग्रे-सफेद फफूंदी होती है।',
            'remedy': '• नीम का तेल स्प्रे करें\n• बेकिंग सोडा घोल (1 टेबलस्पून प्रति गैलन पानी) का उपयोग करें\n• वायु संचार सुधारें\n• संक्रमित पत्तियाँ हटाएँ',
            'icon': '🍄'
        },
        'powdery_mildew': {
            'name': 'पाउडरी मिल्ड्यू',
            'description': 'पाउडरी मिल्ड्यू पत्तियों और तनों पर सफेद पाउडर जैसे धब्बों के रूप में दिखाई देता है।',
            'remedy': '• दूध का घोल (1 भाग दूध, 9 भाग पानी) स्प्रे करें\n• नीम का तेल लगाएँ\n• बेकिंग सोडा स्प्रे करें\n• प्रभावित हिस्से काटें',
            'icon': '❄️'
        },
        'black_spot': {
            'name': 'ब्लैक स्पॉट',
            'description': 'ब्लैक स्पॉट पत्तियों पर काले धब्बे और पीले घेरे बनाता है, जिससे पत्तियाँ झड़ जाती हैं।',
            'remedy': '• नीम का तेल लगाएँ\n• बेकिंग सोडा घोल का उपयोग करें\n• संक्रमित पत्तियाँ हटाएँ\n• वायु संचार सुधारें',
            'icon': '⚫'
        },
        'rose_slug': {
            'name': 'रोज स्लग',
            'description': 'रोज स्लग पत्तियों को कंकाल जैसा बना देते हैं।',
            'remedy': '• लार्वा को हाथ से हटाएँ\n• नीम का तेल लगाएँ\n• कीटनाशक साबुन का उपयोग करें\n• प्राकृतिक शत्रुओं को प्रोत्साहित करें',
            'icon': '🐛'
        },
        'rose_mosaic': {
            'name': 'रोज मोज़ेक',
            'description': 'रोज मोज़ेक वायरस पत्तियों पर पीले पैटर्न बनाता है।',
            'remedy': '• संक्रमित पौधों को हटाएँ\n• वायरस-रहित पौध सामग्री का उपयोग करें\n• पौधों को स्वस्थ रखें\n• एफिड्स को नियंत्रित करें',
            'icon': '🎨'
        },
        'rose_rust': {
            'name': 'रोज रस्ट',
            'description': 'रोज रस्ट पत्तियों और तनों पर नारंगी पाउडर जैसे धब्बों के रूप में दिखाई देता है।',
            'remedy': '• नीम का तेल लगाएँ\n• सल्फर-आधारित फफूंदनाशक का उपयोग करें\n• संक्रमित पत्तियाँ हटाएँ\n• वायु संचार सुधारें',
            'icon': '🟠'
        }
    },
    'mr': {
        'healthy': {
            'name': 'निरोगी',
            'description': 'गुलाबाची पाने निरोगी आहेत आणि कोणत्याही रोगाचे लक्षण नाही.',
            'remedy': '• नियमित देखभाल सुरू ठेवा\n• कोणतेही बदल लक्षात घ्या\n• योग्य पाणी देणे सुरू ठेवा\n• चांगला वायुवीजन ठेवा',
            'icon': '✅'
        },
        'downy_mildew': {
            'name': 'डाउनी मिल्ड्यू',
            'description': 'डाउनी मिल्ड्यू हा एक बुरशीजन्य रोग आहे जो पानांवर पिवळ्या ठिपक्यांसारखा दिसतो आणि खाली राखाडी-श्वेत बुरशी असते.',
            'remedy': '• नीम तेलाचा फवारा करा\n• बेकिंग सोडा द्रावण (1 टेबलस्पून प्रति गॅलन पाणी) वापरा\n• वायुवीजन सुधारवा\n• संक्रमित पाने काढा',
            'icon': '🍄'
        },
        'powdery_mildew': {
            'name': 'पावडरी मिल्ड्यू',
            'description': 'पावडरी मिल्ड्यू पानांवर आणि खोडांवर पांढरे पावडर सारखे डाग निर्माण करतो.',
            'remedy': '• दूध द्रावण (1 भाग दूध, 9 भाग पाणी) फवारणी करा\n• नीम तेल लावा\n• बेकिंग सोडा फवारणी करा\n• प्रभावित भाग कापा',
            'icon': '❄️'
        },
        'black_spot': {
            'name': 'ब्लॅक स्पॉट',
            'description': 'ब्लॅक स्पॉट पानांवर काळे डाग आणि पिवळे वर्तुळे निर्माण करतो, ज्यामुळे पाने गळतात.',
            'remedy': '• नीम तेल लावा\n• बेकिंग सोडा द्रावण वापरा\n• संक्रमित पाने काढा\n• वायुवीजन सुधारवा',
            'icon': '⚫'
        },
        'rose_slug': {
            'name': 'रोज स्लग',
            'description': 'रोज स्लग पाने कंकालासारखी करतात.',
            'remedy': '• अळ्या हाताने काढा\n• नीम तेल लावा\n• कीटकनाशक साबण वापरा\n• नैसर्गिक शत्रूंना प्रोत्साहन द्या',
            'icon': '🐛'
        },
        'rose_mosaic': {
            'name': 'रोज मोझेक',
            'description': 'रोज मोझेक विषाणू पानांवर पिवळे नमुने तयार करतो.',
            'remedy': '• संक्रमित झाडे काढा\n• विषाणू-मुक्त लागवड साहित्य वापरा\n• झाडे निरोगी ठेवा\n• एफिड्स नियंत्रित करा',
            'icon': '🎨'
        },
        'rose_rust': {
            'name': 'रोज रस्ट',
            'description': 'रोज रस्ट पानांवर आणि खोडांवर नारिंगी पावडर सारखे डाग निर्माण करतो.',
            'remedy': '• नीम तेल लावा\n• सल्फर-आधारित बुरशीनाशक वापरा\n• संक्रमित पाने काढा\n• वायुवीजन सुधारवा',
            'icon': '🟠'
        }
    }
}

# Add custom CSS
st.markdown("""
<style>
    /* Sidebar styling */
    .sidebar-container {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar-header {
        color: #2c3e50;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .instruction-step {
        margin-bottom: 1rem;
    }
    
    .instruction-title {
        color: #4CAF50;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .instruction-content {
        color: #2c3e50;
        font-size: 0.9rem;
        margin-left: 1rem;
    }
    .instruction-content ul,
    .tips-list {
        list-style-type: disc;
        padding-left: 1.2rem;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .instruction-content li,
    .tips-list li {
        margin-bottom: 0.2rem;
        line-height: 1.4;
    }
    
    .tips-section {
        color: #2c3e50;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .tips-header {
        margin-bottom: 0.5rem;
    }
    
    .tips-list {
        margin-left: 1rem;
        padding-left: 0.5rem;
    }
    
    .tips-subheader {
        margin: 1rem 0 0.5rem 0;
    }

    /* Prediction Results Styling */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .prediction-title-h3 {
        margin-bottom: 1rem;
    }

    .prediction-confidence-text {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    .prediction-confidence-desc {
        font-weight: 600;
    }

    /* Disease Information Styling */
    .info-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .info-description-container {
        margin-bottom: 2rem;
    }

    .info-heading {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .info-description-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }

    .info-recommendations-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
    }

    /* Other Possible Conditions Styling */
    .other-prediction-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }

    .other-prediction-text {
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    try:
        st.info(f"Checking for model file at: {MODEL_PATH}")
        st.info(f"Current script directory: {os.path.dirname(os.path.abspath(__file__))}")
        st.info(f"Current working directory (os.getcwd()): {os.getcwd()}")
        st.info(f"Files in current working directory: {os.listdir('.')}") # List files in CWD

        # Check if model file exists locally
        if not os.path.exists(MODEL_PATH):
            st.warning("Model file not found locally. Attempting to download...")
            try:
                with st.spinner("Downloading model (this may take a moment)...
" \
                                f"Downloading from: {MODEL_DOWNLOAD_URL}
" \
                                f"Saving to: {MODEL_PATH}"):
                    response = requests.get(MODEL_DOWNLOAD_URL, stream=True)
                    response.raise_for_status() # Raise an exception for HTTP errors
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                st.success("Model downloaded successfully!")
                st.info(f"Files in script directory after download: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}") # List files in script directory
                st.info(f"Is model file now present? {os.path.exists(MODEL_PATH)}")

            except requests.exceptions.RequestException as req_err:
                st.error(f"Error downloading model: {req_err}")
                st.error("Please ensure the model file is accessible from the provided URL and that the Streamlit app has write permissions.")
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred during download: {e}")
                return None

        # Check if file is empty after ensuring it exists (either locally or after download)
        if os.path.getsize(MODEL_PATH) == 0:
            st.error("Model file is empty. Please check if the file is corrupted or download failed.")
            return None

        # Try to load the model
        try:
            # Disable GPU usage to avoid potential issues
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Load model with custom_objects to handle potential compatibility issues
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            # Compile the model with specific settings
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Set memory growth to avoid OOM errors
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            st.success("Model loaded successfully!")
            return model
            
        except Exception as model_error:
            st.error(f"Error loading model: {str(model_error)}")
            st.error("This might be due to:")
            st.error("1. Incompatible TensorFlow version")
            st.error("2. Corrupted model file")
            st.error("3. Model architecture mismatch")
            return None

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def preprocess_image(image):
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in RGB format
        if len(image.shape) == 2:  # Grayscale
            if USE_CV2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = np.stack((image,) * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            if USE_CV2:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                image = image[:, :, :3]
        
        # Resize image
        if USE_CV2:
            img = cv2.resize(image, (224, 224))
        else:
            img = np.array(Image.fromarray(image).resize((224, 224)))
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def get_disease_name(index):
    disease_mapping = {
        0: 'healthy',
        1: 'downy_mildew',
        2: 'powdery_mildew',
        3: 'black_spot',
        4: 'rose_slug',
        5: 'rose_mosaic',
        6: 'rose_rust'
    }
    return disease_mapping.get(index, 'unknown')

def get_confidence_class(confidence):
    if confidence >= 0.90:
        return "confidence-very-high"
    elif confidence >= 0.80:
        return "confidence-high"
    elif confidence >= 0.70:
        return "confidence-medium"
    elif confidence >= 0.60:
        return "confidence-low"
    else:
        return "confidence-very-low"

def get_confidence_description(confidence):
    if confidence >= 0.90:
        return "Very High Confidence"
    elif confidence >= 0.80:
        return "High Confidence"
    elif confidence >= 0.70:
        return "Medium Confidence"
    elif confidence >= 0.60:
        return "Low Confidence"
    else:
        return "Very Low Confidence"

def get_confidence_color(confidence):
    if confidence >= 0.90:
        return "#28a745"  # Green
    elif confidence >= 0.80:
        return "#17a2b8"  # Blue
    elif confidence >= 0.70:
        return "#ffc107"  # Yellow
    elif confidence >= 0.60:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red

def display_prediction_results(prediction, lang, DISEASES):
    T = UI_TEXT[lang]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(prediction[0])[-3:][::-1]
    top_3_confidences = prediction[0][top_3_idx]
    top_3_diseases = [get_disease_name(idx) for idx in top_3_idx]
    
    # Main prediction
    main_disease = top_3_diseases[0]
    main_confidence = top_3_confidences[0]
    confidence_class = get_confidence_class(main_confidence)
    confidence_desc = get_confidence_description(main_confidence)
    confidence_color = get_confidence_color(main_confidence)
    
    # Display main prediction
    st.markdown(textwrap.dedent(f"""<div class="prediction-box"
        style="border-left: 5px solid {confidence_color};"
    >
        <h3 class="prediction-title-h3"
            style="color: {confidence_color};"
        >
            {DISEASES[main_disease]['icon']} {DISEASES[main_disease]['name']}
        </h3>
        <div class="prediction-confidence-text">
            <strong>{T['confidence']}:</strong> {main_confidence:.2%}
        </div>
        <p class="prediction-confidence-desc"
            style="color: {confidence_color};"
        >
            {confidence_desc}
        </p>
    </div>"""), unsafe_allow_html=True)
    
    # Generate list items for recommendations dynamically
    remedy_lines = [item.strip().lstrip('•') for item in DISEASES[main_disease]['remedy'].split('\n•') if item.strip()]
    remedy_html = "".join([f"<li>{line}</li>" for line in remedy_lines])

    st.markdown(f"""<div class="info-box">
<div class="info-description-container">
<h4 class="info-heading">{T['desc']}</h4>
<div class="instruction-content">{DISEASES[main_disease]['description']}</div>
</div>

<div class="info-recommendations-container" style="border-left: 4px solid #4CAF50;">
<h4 class="info-heading">{T['recommend']}</h4>
<div class="instruction-content">
<ul>
{remedy_html}
</ul>
</div>
</div>
</div>""", unsafe_allow_html=True)
    
    # Display other possible predictions
    if len(top_3_diseases) > 1:
        st.markdown("### Other Possible Conditions")
        for i, (disease, conf) in enumerate(zip(top_3_diseases[1:], top_3_confidences[1:]), 1):
            st.markdown(textwrap.dedent(f"""<div class="other-prediction-box"
                style="border-left: 3px solid {get_confidence_color(conf)};"
            >
                <p class="other-prediction-text">
                    <strong>{i}.</strong> {DISEASES[disease]['icon']} {DISEASES[disease]['name']} 
                    ({conf:.2%})
                </p>
            </div>"""), unsafe_allow_html=True)

def main():
    # Language selector with improved styling
    st.sidebar.markdown("""
    <style>
    .language-selector {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .sidebar-header {
        margin-top: 0;
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Language selector
    lang = st.sidebar.selectbox('🌐 Select Language / भाषा चुनें / भाषा निवडा', 
                              list(LANGUAGES.keys()), 
                              format_func=lambda x: LANGUAGES[x])
    
    T = UI_TEXT[lang]
    DISEASES = DISEASE_INFO[lang]

    st.title(T['title'])
    st.subheader(T['subtitle'])
    
    # Sidebar with improved instructions and tips in dropdowns
    st.sidebar.markdown(f'<div class="sidebar-header">{T["about"]}</div>', unsafe_allow_html=True)
    st.sidebar.write(f"""
    {T['supported']}
    - {DISEASES['healthy']['icon']} {DISEASES['healthy']['name']}
    - {DISEASES['downy_mildew']['icon']} {DISEASES['downy_mildew']['name']}
    - {DISEASES['powdery_mildew']['icon']} {DISEASES['powdery_mildew']['name']}
    - {DISEASES['black_spot']['icon']} {DISEASES['black_spot']['name']}
    - {DISEASES['rose_slug']['icon']} {DISEASES['rose_slug']['name']}
    - {DISEASES['rose_mosaic']['icon']} {DISEASES['rose_mosaic']['name']}
    - {DISEASES['rose_rust']['icon']} {DISEASES['rose_rust']['name']}
    """)
    
    # Instructions dropdown
    with st.sidebar.expander(f"📋 {T['instructions']}", expanded=False):
        instruction_1_items = [f"<li>{item.strip()}</li>" for item in T['instruction_1_content'].split('\n•') if item.strip()]
        instruction_2_items = [f"<li>{item.strip()}</li>" for item in T['instruction_2_content'].split('\n•') if item.strip()]
        instruction_3_items = [f"<li>{item.strip()}</li>" for item in T['instruction_3_content'].split('\n•') if item.strip()]
        
        instructions_html = (
            f'<div class="instruction-step">'
            f'<div class="instruction-title">{T["instruction_1"]}</div>'
            f'<div class="instruction-content">'
            f'<ul>'
            f'{"".join(instruction_1_items)}'
            f'</ul>'
            f'</div>'
            f'</div>'
            f'<div class="instruction-step">'
            f'<div class="instruction-title">{T["instruction_2"]}</div>'
            f'<div class="instruction-content">'
            f'<ul>'
            f'{"".join(instruction_2_items)}'
            f'</ul>'
            f'</div>'
            f'</div>'
            f'<div class="instruction-step">'
            f'<div class="instruction-title">{T["instruction_3"]}</div>'
            f'<div class="instruction-content">'
            f'<ul>'
            f'{"".join(instruction_3_items)}'
            f'</ul>'
            f'</div>'
            f'</div>'
        )
        st.markdown(instructions_html, unsafe_allow_html=True)
    
    # Tips dropdown
    with st.sidebar.expander(f"💡 {T['tips']}", expanded=False):
        tips_content_items = [f"<li>{item.strip()}</li>" for item in T['tips_content'].split('\n•') if item.strip()]
        best_practices_items = [f"<li>{item.strip()}</li>" for item in T['best_practices_content'].split('\n•') if item.strip()]
        
        tips_html = (
            f'<div class="instruction-step">'
            f'<div class="instruction-title">{T["photo_tips"]}</div>'
            f'<div class="instruction-content">'
            f'<ul>'
            f'{"".join(tips_content_items)}'
            f'</ul>'
            f'</div>'
            f'</div>'
            f'<div class="instruction-step">'
            f'<div class="instruction-title">{T["best_practices"]}</div>'
            f'<div class="instruction-content">'
            f'<ul>'
            f'{"".join(best_practices_items)}'
            f'</ul>'
            f'</div>'
            f'</div>'
        )
        st.markdown(tips_html, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f'<div class="disease-title">{T["upload_image"]}</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(T['choose_image'], type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=T['choose_image'], use_column_width=True)
                if st.button(T['predict']):
                    with st.spinner("Analyzing image..."):
                        # Load model first
                        model = load_model()
                        if model is None:
                            st.error("Could not load the model. Please check the error messages above.")
                            return

                        # Process image
                        processed_img = preprocess_image(image)
                        if processed_img is None:
                            st.error("Could not process the image. Please try a different image.")
                            return

                        # Make prediction
                        try:
                            prediction = model.predict(processed_img)
                            # Display prediction results with clean HTML structure
                            display_prediction_results(prediction, lang, DISEASES)
                        except Exception as pred_error:
                            st.error(f"Error making prediction: {str(pred_error)}")
                            st.error("This might be due to:")
                            st.error("1. Incompatible model output format")
                            st.error("2. Image preprocessing issues")
                            st.error("3. Model architecture mismatch")
            except Exception as img_error:
                st.error(f"Error processing uploaded image: {str(img_error)}")
                st.error("Please try uploading a different image.")
        else:
            st.info(T['no_image'])
    with col2:
        st.markdown(f'<div class="disease-title">{T["gallery"]}</div>', unsafe_allow_html=True)
        st.write(T['common_conditions'])
        for disease, info in DISEASES.items():
            with st.expander(f"{info['icon']} {info['name']}"):
                remedy_items_gallery = [f"<li>{item.strip().lstrip('•')}</li>" for item in info['remedy'].split('\n•') if item.strip()]
                remedy_html_gallery = "".join(remedy_items_gallery)

                st.markdown(f"""<div class="instruction-step">
<div class="instruction-title">📝 Description</div>
<div class="instruction-content">{info['description']}</div>
</div>

<div class="instruction-step">
<div class="instruction-title">💡 Recommendations</div>
<div class="instruction-content">
<ul>
{remedy_html_gallery}
</ul>
</div>
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
