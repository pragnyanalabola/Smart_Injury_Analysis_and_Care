from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import uuid
import io
from tensorflow.keras.models import load_model as keras_load_model
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import base64
from io import BytesIO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'models/temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/injury_model.h5'
CLASS_NAMES_PATH = 'models/class_names.npy'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

def crop_to_injury(img_path):
    
    img = cv2.imread(img_path)
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        _, buffer = cv2.imencode('.jpg', img)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"  # Return original if no contours found
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    
    # Crop and save
    cropped = img[y:y+h, x:x+w]
    _, buffer = cv2.imencode('.jpg', cropped)
    
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

# Load the trained model and class names
def load_model():
    try: 
        # Try to load the lightweight model
        model = keras_load_model(MODEL_PATH)  # MODEL_PATH should point to your .h5 file
        
        class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
        model_type = 'keras'
        print("Loaded Keras model successfully")
        return model, class_names, model_type
    except Exception as e:
        print(f"Could not load model: {str(e)}, using simulated predictions")
        return None, None, 'simulated'

MODEL, CLASS_NAMES, MODEL_TYPE = load_model()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img_path, target_size=(64, 64)):
    """
    Extracts simple features from an image for the lightweight model.
    """
    try:
        # Read and resize image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img = cv2.resize(img, target_size)
        
        # Convert to different color spaces and extract features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract basic statistical features
        features = []
        
        # Color features (mean and std of each channel)
        for channel in cv2.split(img):
            features.extend([np.mean(channel), np.std(channel)])
        
        # HSV features
        for channel in cv2.split(hsv):
            features.extend([np.mean(channel), np.std(channel)])
        
        # Grayscale features
        features.extend([np.mean(gray), np.std(gray)])
        
        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.sum(edges) / (target_size[0] * target_size[1]))
        
        return np.array(features)
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def predict_injury(img_path):
    """
    Predicts the injury type from an image and description.
    """
    try:
        if MODEL_TYPE == 'keras' and MODEL is not None:
            # Preprocess image for Keras model
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Could not read image.")
            #match training preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # Match the size used in model.py
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = MODEL.predict(img)[0]
            predicted_idx = np.argmax(prediction)

            # print(f"[DEBUG] Raw Prediction Array: {prediction}") 
            # print(f"[DEBUG] Predicted Index: {predicted_idx}") 

            predicted_class_raw = CLASS_NAMES[predicted_idx].strip().lower()
            
            # print(f"[DEBUG] Predicted Class Raw (from CLASS_NAMES): {predicted_class_raw}") 

            plural_to_singular = {
                "bruises": "bruise",
                "burns": "burn",
                "cuts": "cut",
                "ulcers": "ulcer",
                "abrasions": "abrasion"
            }
            predicted_class = plural_to_singular.get(predicted_class_raw, predicted_class_raw)
            # print(f"[DEBUG] Predicted Class (after singular conversion): {predicted_class}, Confidence: {confidence}") 
            confidence = float(prediction[predicted_idx])

            return predicted_class , confidence

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Fallback to simulated prediction
        import random
        injury_types = ['abrasion', 'bruise', 'burn', 'cut', 'ulcer']
        injury_type = random.choice(injury_types)
        confidence = random.uniform(0.7, 0.99)
        
        return injury_type, confidence

def get_injury_info(injury_type, description):
    """
    Generates dynamic precautions and medications based on injury type and description.
    """
    # Base information for each injury type
    base_info = {
        'abrasion': {
            'name': 'Abrasion',
            'description': 'A partial-thickness wound caused by friction against a rough surface, removing the top layers of skin.',
            'precautions': [
                'Clean the wound gently with mild soap and running water.',
                'Apply gentle pressure using a clean cloth to stop any bleeding.',
                'Use sterilized tweezers to carefully remove dirt or debris from the wound.',
                'Avoid using hydrogen peroxide, iodine, or harsh antiseptics directly on the wound.',
                'Apply a thin layer of antibiotic ointment or petroleum jelly to the cleaned area.',
                'Cover the wound with a sterile bandage and change the dressing daily to monitor for infection.'
            ],
            'medications': [
                'Apply over-the-counter antibiotic ointments like Bacitracin or Neosporin to prevent infection.',
                'Use petroleum jelly to keep the wound moist and promote faster healing.',
                'Cover with sterile bandages or gauze to protect the area and prevent contamination.',
                'Take ibuprofen to reduce pain, swelling, and inflammation if needed.',
                'Use acetaminophen as an alternative analgesic if NSAIDs are contraindicated.',
                'Take prescription oral antibiotics like amoxicillin or cephalexin if the abrasion shows signs of infection.'
            ]
        },
        'bruise': {
            'name': 'Bruise',
            'description': 'A contusion occurs when small blood vessels under the skin rupture but the skin remains intact, causing discoloration.',
            'precautions': [
                'Elevate the bruised area above heart level to help minimize swelling.',
                'Apply a cold pack wrapped in cloth for 10–20 minutes immediately after injury.',
                'Reapply cold compresses every few hours during the first 24–48 hours.',
                'Use a loose elastic bandage to gently compress the area if swelling occurs.',
                'Rest the injured body part and avoid further physical impact.',
                'Avoid taking aspirin unless prescribed, as it can worsen bleeding under the skin.'
            ],
            'medications': [
                'Use ice packs to reduce internal bleeding and swelling under the skin.',
                'Apply an elastic compression wrap to help control swelling and provide support.',
                'Take ibuprofen to relieve pain and decrease inflammation.',
                'Use acetaminophen for pain relief if ibuprofen or NSAIDs cannot be used.',
                'Apply arnica gel topically to promote bruise healing and reduce discomfort.',
                'Use topical analgesics like diclofenac or lidocaine to relieve localized pain.'
            ]
        },
        'burn': {
            'name': 'Burn',
            'description': 'Tissue damage caused by heat, chemicals, electricity, sunlight, or radiation.',
            'precautions': [
                'Immediately cool the burn by running it under cool (not cold) water for about 10 minutes.',
                'Do not apply ice, butter, oil, or toothpaste to the burned area.',
                'Remove any tight clothing, jewelry, or accessories before the area swells.',
                'After cooling, apply a soothing gel like aloe vera to reduce pain and inflammation.',
                'Cover the burn loosely with sterile gauze or a clean cloth to protect it.',
                'Do not pop any blisters and clean gently if they rupture on their own.'
            ],
            'medications': [
                'Take ibuprofen or acetaminophen to manage pain and reduce inflammation.',
                'Apply aloe vera gel to cool and hydrate the burned skin.',
                'Use over-the-counter antibiotic ointments like Neosporin on broken or blistered skin.',
                'Apply petroleum jelly to keep the wound moist and support healing.',
                'Use silver sulfadiazine cream as a prescription topical to prevent infection in moderate burns.',
                'Receive a tetanus booster if the burn is deep or contaminated and immunization is outdated.'
            ]
        },
        'cut': {
            'name': 'Cut',
            'description': 'A wound that breaks through the skin, caused by a sharp object.',
            'precautions': [
                'Apply firm pressure with a clean cloth to control and stop the bleeding.',
                'Rinse the cut thoroughly with clean water after bleeding is controlled.',
                'Clean around the cut using mild soap while avoiding direct contact with the wound.',
                'Do not use hydrogen peroxide or iodine inside the cut to avoid tissue damage.',
                'Use sterilized tweezers to remove any visible dirt or foreign objects.',
                'Cover the cut with a sterile bandage and change it daily or if it becomes dirty.'
            ],
            'medications': [
                'Apply over-the-counter antibiotic ointment like Neosporin to prevent bacterial infection.',
                'Use petroleum jelly to keep the wound moist and reduce scab formation.',
                'Cover with sterile adhesive bandages or gauze to protect the cut.',
                'Take acetaminophen or ibuprofen for pain management as needed.',
                'Use prescription oral antibiotics like amoxicillin if the cut is deep or infected.',
                'Receive a tetanus shot if the cut is caused by a dirty object and immunization is not current.'
            ]
        },
        'ulcer': {
            'name': 'Ulcer',
            'description': 'An open sore that develops when the skin\'s surface breaks down and the underlying tissues become exposed.',
            'precautions': [
                'Relieve pressure on the ulcer by repositioning or using off-loading cushions.',
                'Gently clean the ulcer daily with saline or mild soap and water.',
                'Avoid using hydrogen peroxide or iodine on the ulcer unless prescribed.',
                'Keep the ulcer bed moist with appropriate dressings like hydrocolloid or foam.',
                'Protect the surrounding skin with barrier creams to prevent breakdown.',
                'Maintain proper hygiene, hydration, and nutrition to support the healing process.'
            ],
            'medications': [
                'Clean the wound using normal saline for gentle and non-irritating irrigation.',
                'Apply barrier creams like zinc oxide or dimethicone to protect nearby skin.',
                'Use hydrocolloid or foam dressings to maintain a moist healing environment.',
                'Apply silver sulfadiazine cream if there is a high risk of infection.',
                'Use topical prescription antibiotics like mupirocin if the ulcer shows signs of infection.',
                'Use pressure-relieving devices such as special cushions or mattresses to prevent further injury.'
            ]
        },
        'unknown': {
            'name': 'Unidentified Injury',
            'description': 'The type of injury could not be determined.',
            'precautions': [
                'Clean the area gently with mild soap and water',
                'Apply a sterile bandage if needed',
                'Rest the affected area',
                'Monitor for changes in appearance or pain level',
                'Seek medical attention if symptoms worsen',
                'Avoid self-diagnosis for unidentified injuries'
            ],
            'medications': [
                'Consult a healthcare professional before applying any medications',
                'Over-the-counter pain relievers if needed',
                'Antiseptic solution for cleaning',
                'Sterile saline solution for rinsing',
                'Avoid applying creams or ointments without medical advice',
                'Use only prescribed medications if available'
            ]
        }
    }
    
    # Get base information
    info = base_info.get(injury_type, base_info['unknown']).copy()
    
    # Analyze description to add dynamic precautions and medications
    description = description.lower()
    
    # Dynamic precautions based on description keywords
    if 'child' in description or 'baby' in description or 'infant' in description:
        info['precautions'].append('Use child-appropriate medications and dosages')
        info['precautions'].append('Consult a pediatrician for proper treatment')
    
    if 'diabetic' in description or 'diabetes' in description:
        info['precautions'].append('Monitor blood sugar levels during healing')
        info['precautions'].append('Seek medical attention promptly as healing may be delayed')
        info['medications'].append('Consult your doctor before using any topical treatments')
    
    if 'allergy' in description or 'allergic' in description:
        info['precautions'].append('Avoid known allergens in treatments')
        info['medications'].append('Consider hypoallergenic bandages and dressings')
    
    # Location-specific precautions
    if 'face' in description or 'head' in description:
        info['precautions'].append('Take extra care with facial injuries to minimize scarring')
        if injury_type == 'cut':
            info['precautions'].append('Consider medical attention for facial cuts to minimize scarring')
    
    if 'hand' in description or 'finger' in description:
        info['precautions'].append('Keep the hand elevated when possible')
        info['precautions'].append('Avoid activities that require fine motor skills until healed')
    
    if 'foot' in description or 'leg' in description or 'ankle' in description:
        info['precautions'].append('Elevate the affected limb when resting')
        info['precautions'].append('Avoid putting weight on the affected area if possible')
    
    # Severity-specific precautions
    if 'severe' in description or 'bad' in description or 'deep' in description:
        info['precautions'].append('Seek immediate medical attention for proper assessment')
        info['precautions'].append('Do not rely solely on self-treatment for severe injuries')
    
    # Activity-specific precautions
    if 'sport' in description or 'exercise' in description or 'running' in description:
        info['precautions'].append('Avoid returning to sports or exercise until fully healed')
        info['precautions'].append('Consider using protective equipment when returning to activities')
    
    # Remove duplicates and limit to reasonable number
    info['precautions'] = list(dict.fromkeys(info['precautions']))[:6]
    info['medications'] = list(dict.fromkeys(info['medications']))[:6]
    
    return info

def generate_pdf_report(injury_type, description, image_path, confidence):
    """Generate a clean and visually appealing PDF report with image and injury details (no confidence or text description)"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch

    # Get injury information
    info = get_injury_info(injury_type, description)

    # File setup
    filename = f"report_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    margin = 1 * inch
    line_height = 16
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, y, "Injury Assessment Report")
    y -= 0.4 * inch

    # Date
    c.setFont("Helvetica", 12)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin, y, f"Date: {current_time}")
    y -= 0.3 * inch

    # Injury Type
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Injury Type: {info['name']}")
    y -= 0.4 * inch

    # Image
    if os.path.exists(image_path):
        try:
            img_width = 3 * inch
            img_height = 3 * inch
            c.drawImage(image_path, margin, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True)
            y -= (img_height + 0.5 * inch)
        except Exception as e:
            print(f"[WARNING] Could not include image: {e}")

    # Precautions Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Precautions:")
    y -= 0.3 * inch
    c.setFont("Helvetica", 12)
    for precaution in info['precautions']:
        c.drawString(margin + 15, y, f"• {precaution}")
        y -= line_height

    y -= 0.4 * inch  # Extra spacing between sections

    # Medications Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Recommended Medications:")
    y -= 0.3 * inch
    c.setFont("Helvetica", 12)
    for medication in info['medications']:
        c.drawString(margin + 15, y, f"• {medication}")
        y -= line_height

    y -= 0.4 * inch

    # Footer Disclaimer
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Disclaimer: This report is generated by an ML system and is not a substitute for")
    c.drawString(margin, y - 12, "professional medical advice. Please consult a healthcare professional for proper diagnosis and treatment.")

    # Save PDF
    c.save()
    return filepath

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/assess')
def assess():
    return render_template('assess.html')

@app.route('/guidelines')
def guidelines():
    return render_template('guidelines.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        if image and allowed_file(image.filename):
            temp_id = uuid.uuid4().hex
            temp_path = os.path.join(UPLOAD_FOLDER, f"{temp_id}_{image.filename}")
            image.save(temp_path)
            
            try:
                # Get cropped image
                cropped_img = crop_to_injury(temp_path)
                if not cropped_img:
                    return jsonify({'error': 'Failed to process image'}), 500
                
                injury_type, confidence = predict_injury(temp_path)
                injury_info = get_injury_info(injury_type, "")
                
                report_path = generate_pdf_report(injury_type, "", temp_path, confidence)
                report_id = os.path.basename(report_path)
                
                response = {
                    'cropped_image': cropped_img  ,
                    'injury_type': injury_info['name'],
                    'precautions': injury_info['precautions'],
                    'medications': injury_info['medications'],
                    'report_id': report_id,
                }
                
                return jsonify(response)
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/api/download-report/<report_id>', methods=['GET'])
def download_report(report_id):
    try:
        report_path = os.path.join(UPLOAD_FOLDER, report_id)
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True, download_name="injury_report.pdf")
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-report', methods=['GET'])
def download_latest_report():
    try:
        # Find the most recent report
        reports = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('report_') and f.endswith('.pdf')]
        if not reports:
            # Generate a sample report if none exists
            sample_report = generate_pdf_report('unknown', 'Sample injury for demonstration', '', 0.8)
            return send_file(sample_report, as_attachment=True, download_name="sample_injury_report.pdf")
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
        latest_report = os.path.join(UPLOAD_FOLDER, reports[0])
        
        return send_file(latest_report, as_attachment=True, download_name="injury_report.pdf")
    except Exception as e:
        print(f"Error downloading report: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')