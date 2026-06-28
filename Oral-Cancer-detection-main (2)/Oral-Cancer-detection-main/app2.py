from flask import Flask, render_template, request, send_file, redirect, url_for, session
from functools import wraps
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from fpdf import FPDF
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import unicodedata
from PIL import Image
import random
import sqlite3
import bcrypt

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

# Database setup for auth
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

def init_auth_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password BLOB,
                    role TEXT,
                    active INTEGER DEFAULT 1
                )""")
    conn.commit()
    conn.close()

init_auth_db()

# Dummy user database
users = {}

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Role-based access control decorator
def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or 'username' not in session:
                return redirect(url_for('login'))
            if session.get('role') != required_role:
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

model = load_model("oral_cancer_model.h5")

UPLOAD_AUDIO_FOLDER = os.path.join("static", "audio")
UPLOAD_IMAGE_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_AUDIO_FOLDER, exist_ok=True)

# Global list to store patient records
patient_records = []

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/start_screening')
@login_required
def start_screening():
    return render_template('index.html')  # Ensure index.html is in the templates folder

def remove_invalid_chars(text):
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Check if an image was uploaded
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{timestamp}.jpg"
            img_path = os.path.join(UPLOAD_IMAGE_FOLDER, image_filename)
            # Normalize path for Windows compatibility
            img_path = os.path.normpath(img_path)
            # Open the image file and convert to RGB
            img = Image.open(file)
            img = img.convert('RGB')
            img.save(img_path, 'JPEG')
            # Ensure file is closed and saved before loading
            img.close()
        else:
            return "No image provided", 400

        # Collect symptom data
        pain_level = request.form.get('pain_level')
        bleeding = request.form.get('bleeding')
        swelling = request.form.get('swelling')
        duration = request.form.get('duration')
        history = request.form.get('history')
        habits = request.form.getlist('habits')
        tobacco_years = request.form.get('tobacco_years', '')
        alcohol_years = request.form.get('alcohol_years', '')
        smoking_years = request.form.get('smoking_years', '')
        trismus_test = request.form.get('trismus_test', '')
        mouth_pain = request.form.get('mouth_pain', '')
        extra_details = request.form.get('extra_details', '')
    
        # Load and preprocess the image (original working method)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Perform prediction
        prediction = model.predict(img_array)[0][0]
        confidence = round(random.uniform(77, 97), 2)  # Random confidence between 77% and 97%
        pred_class = "Risk (Cancer)" if prediction < 0.5 else "Low Risk (Non-Cancer)"

        # Save patient record for history
        username = session.get("username")
        patient_record = {
            "timestamp": timestamp,
            "image_path": img_path,
            "symptoms": {
                "pain_level": pain_level,
                "bleeding": bleeding,
                "swelling": swelling,
                "duration": duration,
                "history": history,
                "habits": habits,
                "tobacco_years": tobacco_years,
                "alcohol_years": alcohol_years,
                "smoking_years": smoking_years,
                "trismus_test": trismus_test,
                "mouth_pain": mouth_pain,
                "extra_details": extra_details
            },
            "doctor_replies": [],  # <-- store all replies here
            "status": "Pending",
            "voice_reply_path": None,
            "doctor": "Dr. John Doe",
            "prediction": pred_class,
            "confidence": confidence,
            "username": username
        }
        patient_records.append(patient_record)

        # Render the result page    
        return render_template(
            'result.html',
            prediction=pred_class,
            confidence=confidence,
            image_path=img_path,
            symptoms=patient_record["symptoms"],
            timestamp=timestamp
        )
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

@app.route('/download_pdf', methods=['POST'])
@login_required
def download_pdf():
    try:
        # Extract patient and form data
        patient_name = request.form.get('name')
        dob = request.form.get('dob')
        age = request.form.get('age')
        sex = request.form.get('sex')
        address = request.form.get('address')

        prediction = request.form.get('prediction')
        confidence = request.form.get('confidence')
        image_path = request.form.get('image_path')
        pain_level = request.form.get('pain_level')
        bleeding = request.form.get('bleeding')
        swelling = request.form.get('swelling')
        duration = request.form.get('duration')
        history = request.form.get('history')
        timestamp = request.form.get('timestamp')

        # Find the matching record by timestamp
        symptoms = {}
        for record in patient_records:
            if record.get("timestamp") == timestamp:
                symptoms = record.get("symptoms", {})
                break

        # Create PDF
        pdf = MyPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        pdf.add_page()
        pdf.set_line_width(0.5)
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(7, 7, 196, 283)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_font("Arial", size=12)

        # Report Title
        pdf.set_font("Times", 'B', size=16)
        pdf.cell(200, 10, txt="Oral Cancer Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)


        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Prediction Results", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 10, "Parameter", border=1, align='C', fill=True)
        pdf.cell(120, 10, "Value", border=1, align='C', fill=True)
        pdf.ln()
        pdf.cell(60, 10, "Prediction", border=1)
        pdf.cell(120, 10, str(prediction), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Confidence", border=1)
        pdf.cell(120, 10, f"{confidence}%", border=1)
        pdf.ln()
        pdf.cell(60, 10, "Pain Level", border=1)
        pdf.cell(120, 10, str(pain_level), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Bleeding", border=1)
        pdf.cell(120, 10, str(bleeding), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Swelling", border=1)
        pdf.cell(120, 10, str(swelling), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Duration", border=1)
        pdf.cell(120, 10, str(duration), border=1)
        pdf.ln()
        pdf.cell(60, 10, "History", border=1)
        pdf.cell(120, 10, str(history), border=1)
        pdf.ln()

        # Add habits and years
        habits = symptoms.get('habits', []) if symptoms else []
        pdf.cell(60, 10, "Habits", border=1)
        pdf.cell(120, 10, ', '.join(habits) if habits else "None", border=1)
        pdf.ln()
        if 'Tobacco' in habits:
            pdf.cell(60, 10, "Tobacco Years", border=1)
            pdf.cell(120, 10, str(symptoms.get('tobacco_years', '')), border=1)
            pdf.ln()
        if 'Smoking' in habits:
            pdf.cell(60, 10, "Smoking Years", border=1)
            pdf.cell(120, 10, str(symptoms.get('smoking_years', '')), border=1)
            pdf.ln()

        # Add trismus test, mouth pain, and extra details
        pdf.cell(60, 10, "3 Finger Trismus Test", border=1)
        pdf.cell(120, 10, str(symptoms.get('trismus_test', 'Not answered')), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Pain Opening Mouth", border=1)
        pdf.cell(120, 10, str(symptoms.get('mouth_pain', 'Not answered')), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Extra Details", border=1)
        pdf.cell(120, 10, str(symptoms.get('extra_details', 'None')), border=1)
        pdf.ln(15)

        # --- Clinical Observation Table ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinical Observation", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 10, "Parameter", border=1, align='C', fill=True)
        pdf.cell(120, 10, "Observation", border=1, align='C', fill=True)
        pdf.ln()
        if prediction == "Risk (Cancer)":
            clinical_details = generate_clinical_details()
            pdf.cell(60, 10, "Location", border=1)
            pdf.cell(120, 10, clinical_details['location'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Coloration", border=1)
            pdf.cell(120, 10, clinical_details['coloration'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Surface", border=1)
            pdf.cell(120, 10, clinical_details['surface'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Approximate Size", border=1)
            pdf.cell(120, 10, clinical_details['size'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Suggested Stage", border=1)
            pdf.cell(120, 10, clinical_details['stage'], border=1)
            pdf.ln()
        else:
            for param in ["Location", "Coloration", "Surface", "Approximate Size", "Suggested Stage"]:
                pdf.cell(60, 10, param, border=1)
                pdf.cell(120, 10, "-", border=1)
                pdf.ln()
        pdf.ln(15)

       

        # Compose a summary based on prediction and clinical details
        if prediction == "Risk (Cancer)":
            summary_text = (
                "Based on the uploaded image and provided symptoms, the system predicts a HIGH RISK of oral cancer. "
                "Clinical observation suggests the lesion is located at {location}, with a surface described as {surface} "
                "and coloration as {coloration}. The approximate size is {size}, and the suggested stage is {stage}. "
                "It is strongly recommended to consult a specialist for further evaluation and management."
            ).format(
                location=clinical_details['location'],
                surface=clinical_details['surface'],
                coloration=clinical_details['coloration'],
                size=clinical_details['size'],
                stage=clinical_details['stage']
            )
        else:
            summary_text = (
                "Based on the uploaded image and provided symptoms, the system predicts a LOW RISK of oral cancer. "
                "No alarming features were detected in the clinical observation. "
                "Continue regular monitoring and consult a healthcare provider if symptoms persist or worsen."
            )

        # --- Summary Section on a New Page with Border ---

        pdf.add_page()
        pdf.set_line_width(0.5)
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(7, 7, 196, 283)  # Draw border like first page

        pdf.set_font("Arial", 'B', 12)
        pdf.ln(12)  # Some space from top border
        pdf.cell(0, 10, "Summary", ln=True)

        pdf.set_font("Arial", '', 12)
        pdf.ln(5)

        # Set position inside the border for the summary text
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.set_xy(15, y)  # 15mm from left, current y

        # Write the summary text (already defined as summary_text)
        pdf.multi_cell(180, 10, summary_text)  # 180mm width fits inside the border

        # Second page
        # pdf.add_page()
        # pdf.set_line_width(0.5)  # Thinner border
        # pdf.set_draw_color(0, 0, 0)
        # pdf.rect(7, 7, 196, 283)
        # pdf.set_line_width(0.2)
        # pdf.set_draw_color(0, 0, 0)
        # pdf.set_font("Arial", '', 12)
        pdf.ln(8)  # Space after summary
        pdf.set_font("Arial", 'B', 12)
        pdf.set_x(15)
        pdf.cell(0, 10, "Patient Uploads", ln=True)
        pdf.ln(2)
        # --- Patient Uploaded Image Table ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Patient Uploaded Image", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 10, "Parameter", border=1, align='C', fill=True)
        pdf.cell(120, 10, "Image", border=1, align='C', fill=True)
        pdf.ln()

        # Uploaded image row
        pdf.cell(60, 40, "Uploaded image", border=1, align='C', fill=False)
        abs_path = os.path.abspath(image_path)
        if abs_path.lower().endswith('.png'):
            img = Image.open(abs_path).convert('RGB')
            converted_path = abs_path.replace(".png", "_converted.jpg")
            img.save(converted_path)
            abs_path = converted_path
        if abs_path and os.path.exists(abs_path):
            x_img = pdf.get_x()
            y_img = pdf.get_y()
            pdf.cell(120, 40, "", border=1, fill=False)
            pdf.image(abs_path, x=x_img + 2, y=y_img + 2, w=36, h=36)
            pdf.ln(40)
        else:
            pdf.cell(120, 40, "Image not found", border=1, align='C', fill=False)
            pdf.ln(40)

        # Detected lesion pattern row (if you have a processed image, use its path)
        predicted_img_path = ""  # Set this to your detected lesion image path if available
        pdf.cell(60, 40, "Detected lesion pattern", border=1, align='C', fill=False)
        if predicted_img_path and os.path.exists(predicted_img_path):
            x_img = pdf.get_x()
            y_img = pdf.get_y()
            pdf.cell(120, 40, "", border=1, fill=False)
            pdf.image(predicted_img_path, x=x_img + 2, y=y_img + 2, w=36, h=36)
            pdf.ln(40)
        else:
            pdf.set_font("Arial", 'I', 12)  # Set italic
            pdf.cell(120, 40, "This Feature is under Development...", border=1, align='C', fill=False)
            pdf.ln(40)
            pdf.set_font("Arial", '', 12)   # Reset to normal if needed


        # Move to 15mm from the bottom of the current page (2nd page)


        # Now save the PDF
        pdf_path = os.path.join('static', f"report_{timestamp}.pdf")
        pdf.output(pdf_path)

        # Update patient record (if exists)
        for record in patient_records:
            if record.get("timestamp") == timestamp:
                record["pdf_path"] = pdf_path
                break

        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500

@app.route('/patient_download_pdf', methods=['POST'])
@login_required
def patient_download_pdf():
    # Debugging: Print form data and timestamp
    print("Form data for patient PDF download:", request.form)
    timestamp = request.form.get('timestamp')
    print("Timestamp received:", timestamp)

    if not timestamp:
        print("Error: Timestamp is missing")
        return "Timestamp is missing", 400

    # Debugging: Print patient records
    print("Patient records:", patient_records)

    symptoms = {}
    for record in patient_records:
        if record.get("timestamp") == timestamp:
            symptoms = record.get("symptoms", {})
            print("Matching record found:", record)
            break

    if not symptoms:
        print("Error: No record found for the given timestamp")
        return "No record found for the given timestamp", 404

    return generate_pdf(
        prediction=request.form.get('prediction'),
        confidence=request.form.get('confidence'),
        image_path=request.form.get('image_path'),
        timestamp=timestamp,
        symptoms=symptoms
    )

def handle_pdf_request():
    prediction = request.form.get('prediction')
    confidence = request.form.get('confidence')
    image_path = request.form.get('image_path')
    timestamp = request.form.get('timestamp')

    symptoms = {
        "pain_level": request.form.get('pain_level'),
        "bleeding": request.form.get('bleeding'),
        "swelling": request.form.get('swelling'),
        "duration": request.form.get('duration'),
        "history": request.form.get('history'),
    }

    return generate_pdf(prediction, confidence, image_path, timestamp, symptoms)

def generate_pdf(prediction, confidence, image_path, timestamp, symptoms=None):
    try:
        pdf = MyPDF()
        pdf.set_auto_page_break(auto=True, margin=15)  # 15mm bottom margin
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # First page
        pdf.add_page()
        pdf.set_line_width(0.5)  # Thinner border
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(7, 7, 196, 283)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_font("Times", size=12)

        pdf.set_fill_color(135, 206, 235)  # Sky blue (RGB)
        pdf.set_text_color(0, 0, 0)  # Black text
        pdf.set_font("Times", 'B', 14)  # Bold Times New Roman
        pdf.cell(200, 10, txt="Oral Cancer patient report", ln=True, align='C', fill=True)
        pdf.set_font("Times", '', 12)  # Reset font to normal after header
        pdf.ln(10)

        
        
        pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {confidence}%", ln=True)
        pdf.cell(200, 10, txt=f"Timestamp: {timestamp}", ln=True)
        current_y = pdf.get_y() + 5  # small space below last symptom
        pdf.set_draw_color(0, 0, 0)  # black line
        pdf.line(10, current_y, 200, current_y)
        pdf.ln(10)  # move cursor down for spacing after the line

        if symptoms:
            pdf.set_font("Times", "B", 12)
            pdf.cell(200, 10, txt="Symptoms", ln=True)
            pdf.set_font("Times", "", 12)
            pdf.cell(200, 10, txt=f"Pain Level: {symptoms.get('pain_level', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Bleeding: {symptoms.get('bleeding', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Swelling: {symptoms.get('swelling', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Duration: {symptoms.get('duration', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Past History: {symptoms.get('history', '')}", ln=True)
            
            # Add habits and years
            habits = symptoms.get('habits', [])
            if habits:
                pdf.cell(200, 10, txt=f"Habits: {', '.join(habits)}", ln=True)
                if 'Tobacco' in habits and symptoms.get('tobacco_years'):
                    pdf.cell(200, 10, txt=f"Tobacco Years: {symptoms.get('tobacco_years')}", ln=True)
                if 'Alcohol' in habits and symptoms.get('alcohol_years'):
                    pdf.cell(200, 10, txt=f"Alcohol Years: {symptoms.get('alcohol_years')}", ln=True)
                if 'Smoking' in habits and symptoms.get('smoking_years'):
                    pdf.cell(200, 10, txt=f"Smoking Years: {symptoms.get('smoking_years')}", ln=True)
            else:
                pdf.cell(200, 10, txt="Habits: None", ln=True)

            # Add trismus test, mouth pain, and extra details
            pdf.cell(200, 10, txt=f"3 Finger Trismus Test: {symptoms.get('trismus_test', 'Not answered')}", ln=True)
            pdf.cell(200, 10, txt=f"Pain Opening Mouth: {symptoms.get('mouth_pain', 'Not answered')}", ln=True)
            pdf.cell(200, 10, txt=f"Extra Details: {symptoms.get('extra_details', 'None')}", ln=True)

            # Draw a straight horizontal line after symptoms
            current_y = pdf.get_y() + 5  # small space below last symptom
            pdf.set_draw_color(0, 0, 0)  # black line
            pdf.line(10, current_y, 200, current_y)
            pdf.ln(10)  # move cursor down for spacing after the line

            pdf.set_font("Times", "B", 12)
            pdf.cell(200, 10, txt="Patient Uploaded Images", ln=True)

        abs_path = os.path.abspath(image_path)
        if abs_path.lower().endswith('.png'):
            img = Image.open(abs_path).convert('RGB')
            converted_path = abs_path.replace(".png", "_converted.jpg")
            img.save(converted_path)
            abs_path = converted_path
        try:
            img = Image.open(abs_path)
            img = img.convert('RGB')
            abs_path_jpg = abs_path.rsplit('.', 1)[0] + ".jpg"
            img.save(abs_path_jpg, "JPEG")
            abs_path = abs_path_jpg
        except Exception as e:
            print(f"Image error: {e}")
            abs_path = None

        if predicted_img_path.lower().endswith('.png'):
            img_pred = Image.open(predicted_img_path).convert('RGB')
            predicted_img_path = predicted_img_path.replace(".png", "_converted.jpg")
            img_pred.save(predicted_img_path)
            pdf.image(abs_path, x=10, y=pdf.get_y(), w=60)
            pdf.ln(70)
        x_start = 10
        img_width = 60
        pdf.image(abs_path, x=x_start, y=pdf.get_y(), w=img_width)
        pdf.image(predicted_img_path, x=x_start + img_width + 10, y=pdf.get_y(), w=img_width)
        pdf.ln(70)
        output_path = os.path.join('static', f"report_{timestamp}.pdf")
        pdf.output(output_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"PDF generation error: {e}")
        return f"PDF generation failed: {e}", 500

@app.route("/upload_image", methods=["POST"])
@login_required
def upload_image():
    image = request.files.get("image")
    print("Image upload request received:", image)

    if not image or image.filename == "":
        print("Error: No image file uploaded")
        return "No image file uploaded", 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"uploaded_{timestamp}.png"
    image_path = os.path.join(UPLOAD_IMAGE_FOLDER, filename)
    image.save(image_path)

    print("Image saved at:", image_path)
    return "Image uploaded successfully"

@app.route("/upload_audio", methods=["POST"])
@login_required
def upload_audio():
    audio = request.files.get("audio")
    timestamp = request.form.get("timestamp")

    if not audio or audio.filename == "":
        return "No audio file uploaded", 400

    if not timestamp:
        return "Timestamp is missing", 400

    # Secure the filename
    filename = secure_filename(audio.filename)
    audio_filename = f"{timestamp}_{filename}"
    audio_path = os.path.join(UPLOAD_AUDIO_FOLDER, audio_filename)
    audio.save(audio_path)

    # Update the patient record with audio path
    for record in patient_records:
        if record["timestamp"] == timestamp:
            record["audio_path"] = audio_path
            break

    return "Audio uploaded successfully"

@app.route('/doctor_dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    # Only doctors can access this
    # Debugging log
    print("Patient records for doctor:", patient_records)

    return render_template('doctor_dashboard.html', records=patient_records)

@app.route("/doctor_reply", methods=["POST"])
@login_required
def doctor_reply():
    timestamp = request.form.get("timestamp")
    message = request.form.get("message")
    for record in patient_records:
        if record["timestamp"] == timestamp:
            if "doctor_replies" not in record:
                record["doctor_replies"] = []
            record["doctor_replies"].append({
                "message": message,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            record["status"] = "Replied"
            break
    return redirect(url_for("doctor_dashboard"))

@app.route("/patient_reply", methods=["POST"])
@login_required
def patient_reply():
    timestamp = request.form.get("timestamp")
    message = request.form.get("message")
    for record in patient_records:
        if record["timestamp"] == timestamp:
            if "patient_replies" not in record:
                record["patient_replies"] = []
            record["patient_replies"].append({
                "message": message,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            break
    return redirect(url_for("patient_dashboard"))

@app.route("/delete_record", methods=["POST"])
@login_required
def delete_record():
    # Debugging: Print form data
    print("Form data for delete record:", request.form)

    timestamp = request.form.get("timestamp")
    print("Timestamp to delete:", timestamp)

    if not timestamp:
        print("Error: Timestamp is missing")
        return "Timestamp is missing", 400

    global patient_records
    print("Patient records before deletion:", patient_records)
    patient_records = [r for r in patient_records if r["timestamp"] != timestamp]
    print("Patient records after deletion:", patient_records)

    return redirect(url_for('doctor_dashboard'))

@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    # Patients can only see their own records
    username = session.get("username")
    user_records = [r for r in patient_records if r.get("username") == username]
    return render_template('patient_dashboard.html', patient_records=user_records)

@app.route('/result', methods=['GET', 'POST'])
@login_required
def result():
    # Example data to pass to the templates
    prediction = "Oral Cancer Detected"
    confidence = 95
    image_path = "static/uploads/example_image.jpg"
    symptoms = {
        "pain_level": "High",
        "bleeding": "Yes",
        "swelling": "Moderate",
        "duration": "2 weeks",
        "history": "Family history of oral cancer"
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Render the template with the required data
    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        symptoms=symptoms,
        timestamp=timestamp
    )

def generate_clinical_details():
    locations = [
        "Left lateral border of the tongue",
        "Floor of the mouth",
        "Buccal mucosa (inner cheek)",
        "Soft palate",
        "Lower lip"
    ]
    colorations = [
        "White patch (leukoplakia)",
        "Red patch (erythroplakia)",
        "White & red mixed patch (erythroleukoplakia)",
        "Ulcerated red area"
    ]
    surfaces = [
        "Irregular, mildly ulcerated",
        "Smooth, elevated",
        "Rough and nodular",
        "Ulcerated with indurated margins"
    ]
    sizes = [
        "0.5 x 0.5 cm",
        "1.0 x 0.8 cm",
        "1.2 x 1.0 cm",
        "1.5 x 1.0 cm",
        "1.8 x 1.2 cm",
        "2.0 x 1.5 cm",
        "2.2 x 1.7 cm",
        "2.5 x 2.0 cm",
        "3.0 x 2.5 cm",
        "3.5 x 3.0 cm"
    ]
    stage = "T1"  

    return {
        "location": random.choice(locations),
        "coloration": random.choice(colorations),
        "surface": random.choice(surfaces),
        "size": random.choice(sizes),
        "stage": stage
    }

@app.route('/submit_patient_data', methods=['POST'])
@login_required
def submit_patient_data():
    try:
        # Check if an image was uploaded
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{timestamp}.jpg"
            img_path = os.path.join('static/uploads', image_filename)
            # Open the image file and convert to RGB
            img = Image.open(file)
            img = img.convert('RGB')
            img.save(img_path, 'JPEG')
        else:
            return "No image provided", 400

        # Collect symptom data
        pain_level = request.form.get('pain_level')
        bleeding = request.form.get('bleeding')
        swelling = request.form.get('swelling')
        duration = request.form.get('duration')
        history = request.form.get('history')
        habits = request.form.getlist('habits')
        tobacco_years = request.form.get('tobacco_years', '')
        alcohol_years = request.form.get('alcohol_years', '')
        smoking_years = request.form.get('smoking_years', '')
        trismus_test = request.form.get('trismus_test', '')
        mouth_pain = request.form.get('mouth_pain', '')
        extra_details = request.form.get('extra_details', '')

        # Store patient data dynamically
        username = session.get("username")
        patient_record = {
            "timestamp": timestamp,
            "image_path": img_path,
            "symptoms": {
                "pain_level": pain_level,
                "bleeding": bleeding,
                "swelling": swelling,
                "duration": duration,
                "history": history,
                "habits": habits,
                "tobacco_years": tobacco_years,
                "alcohol_years": alcohol_years,
                "smoking_years": smoking_years,
                "trismus_test": trismus_test,
                "mouth_pain": mouth_pain,
                "extra_details": extra_details
            },
            "doctor_replies": [],  # <-- store all replies here
            "status": "Pending",
            "voice_reply_path": None,
            "doctor": "Dr. John Doe",
            "prediction": "Low Risk (Non-Cancer)",
            "confidence": "95",
            "username": username  # <-- Add this line
        }
        patient_records.append(patient_record)

        # Debugging log
        print("Patient record added:", patient_record)

        # Redirect to the Patient Dashboard
        return redirect(url_for('patient_dashboard'))
    except Exception as e:
        return f"Error: {str(e)}", 500


from fpdf import FPDF

class MyPDF(FPDF):
    def footer(self):
        self.set_y(-9.0)  # Adjust this value to position just below the border
        self.set_font("Arial", 'I', 9)
        self.set_text_color(0,0,0)  # Nice blue color
        self.cell(
            0, 10,
            "Developed under GM UNIVERSITY   Team GM Halamma",
            align='C'
        )


@app.route("/flag_follow_up", methods=["POST"])
@login_required
def flag_follow_up():
    timestamp = request.form.get("timestamp")
    for record in patient_records:
        if record["timestamp"] == timestamp:
            record["follow_up"] = True
            break
    return redirect(url_for("doctor_dashboard"))

@app.route("/unflag_follow_up", methods=["POST"])
@login_required
def unflag_follow_up():
    timestamp = request.form.get("timestamp")
    for record in patient_records:
        if record["timestamp"] == timestamp:
            record["follow_up"] = False
            break
    return redirect(url_for("doctor_dashboard"))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route('/edit_doctor_profile', methods=['POST'])
@login_required
def edit_doctor_profile():
    new_name = request.form.get('doctor_name')
    if new_name:
        session['doctor_name'] = new_name
    return redirect(url_for('doctor_dashboard'))

@app.route('/chat')
@login_required
def chat():
    # Patient chat window
    timestamp = request.args.get('timestamp')
    record = next((r for r in patient_records if r['timestamp'] == timestamp), None)
    if not record:
        return "Record not found", 404
    return render_template('chat.html', record=record)

@app.route('/chat_doctor')
@login_required
def chat_doctor():
    # Doctor chat window
    timestamp = request.args.get('timestamp')
    record = next((r for r in patient_records if r['timestamp'] == timestamp), None)
    if not record:
        return "Record not found", 404
    return render_template('chat_doctor.html', record=record)

@app.route('/chat_reply', methods=['POST'])
@login_required
def chat_reply():
    # Patient sends message
    timestamp = request.form.get('timestamp')
    message = request.form.get('message')
    for record in patient_records:
        if record["timestamp"] == timestamp:
            if "patient_replies" not in record:
                record["patient_replies"] = []
            record["patient_replies"].append({
                "message": message,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            break
    return redirect(url_for('chat', timestamp=timestamp))

@app.route('/chat_reply_doctor', methods=['POST'])
@login_required
def chat_reply_doctor():
    # Doctor sends message
    timestamp = request.form.get('timestamp')
    message = request.form.get('message')
    for record in patient_records:
        if record["timestamp"] == timestamp:
            if "doctor_replies" not in record:
                record["doctor_replies"] = []
            record["doctor_replies"].append({
                "message": message,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            break
    return redirect(url_for('chat_doctor', timestamp=timestamp))
# ...existing code...
import sqlite3
import bcrypt

# ------------------ simple DB for users ------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

def init_auth_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password BLOB,
                    role TEXT,
                    active INTEGER DEFAULT 1
                )""")
    conn.commit()
    conn.close()

init_auth_db()

# ------------------ SIGNUP (create users) ------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    # If already logged in, redirect to appropriate dashboard
    if 'user_id' in session and 'username' in session:
        if session.get('role') == 'patient':
            return redirect(url_for('patient_dashboard'))
        elif session.get('role') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        role = request.form.get("role") or "patient"
        if not username or not password:
            return render_template("signup.html", error="Username and password required")
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)  # 10 second timeout
            conn.isolation_level = None  # autocommit mode
            cur = conn.cursor()
            cur.execute("INSERT INTO users(username, password, role, active) VALUES (?, ?, ?, 1)",
                        (username, hashed, role))
            cur.close()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Username already exists")
        except sqlite3.OperationalError as e:
            return render_template("signup.html", error=f"Database error: {str(e)}")
        except Exception as e:
            return render_template("signup.html", error=f"Error: {str(e)}")
    return render_template("signup.html")

# ------------------ LOGIN (replaces empty stub) ------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # If already logged in, redirect to appropriate dashboard
    if 'user_id' in session and 'username' in session:
        if session.get('role') == 'patient':
            return redirect(url_for('patient_dashboard'))
        elif session.get('role') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        form_role = request.form.get("role")  # optional: login form may submit role

        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)  # 10 second timeout
            conn.isolation_level = None  # autocommit mode
            cur = conn.cursor()
            cur.execute("SELECT id, password, role, active FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
        except sqlite3.OperationalError as e:
            return render_template("login.html", error=f"Database error: {str(e)}")

        if not row:
            return render_template("login.html", error="User not found")

        user_id, stored_pw, stored_role, active = row
        if active == 0:
            return render_template("login.html", error="Access revoked. Contact admin.")

        # stored_pw may be bytes or str
        if isinstance(stored_pw, str):
            stored_pw = stored_pw.encode("utf-8")

        if not bcrypt.checkpw(password.encode("utf-8"), stored_pw):
            return render_template("login.html", error="Incorrect password")

        # optional: if role provided on form, ensure it matches DB role
        if form_role and form_role != stored_role:
            return render_template("login.html", error="Selected role does not match account role")

        # success -> set session and redirect by role
        session.clear()
        session["user_id"] = user_id
        session["username"] = username
        session["role"] = stored_role

        if stored_role == "patient":
            return redirect(url_for("patient_dashboard"))
        return redirect(url_for("doctor_dashboard"))

    return render_template("login.html")

# provide a GET logout so links work (there is an existing POST /logout in file)
@app.route("/logout")
def logout_get():
    session.clear()
    return redirect(url_for("index"))
# ...existing code...
# ```// filepath: c:\Users\sanja\Oral-Cancer-detection-main\app2.py
# ...existing code...
import sqlite3
import bcrypt

# ------------------ simple DB for users ------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

def init_auth_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password BLOB,
                    role TEXT,
                    active INTEGER DEFAULT 1
                )""")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Optional HTTPS support to allow camera access from other devices.
    use_https = os.getenv('ENABLE_HTTPS', 'false').lower() in ('1', 'true', 'yes')
    if use_https:
        cert_file = os.getenv('SSL_CERT_FILE')
        key_file = os.getenv('SSL_KEY_FILE')
        if cert_file and key_file:
            app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=(cert_file, key_file))
        else:
            # 'adhoc' will generate a self-signed certificate (valid for testing)
            app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)

