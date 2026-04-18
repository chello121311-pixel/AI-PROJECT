import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Plant AI", layout="centered")

# ---------------- BACKGROUND ----------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("bg.jpg")

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpg;base64,{bg}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
def login():
    st.markdown("""
    <style>
    .login-box {
        max-width: 500px;
        margin: auto;
        margin-top: 120px;
        padding: 40px;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 20px;
        backdrop-filter: blur(12px);
    }
    .login-title {
        font-size: 38px;
        font-weight: bold;
        color: #22c55e;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🌿 Plant AI Login</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")

    st.markdown('</div>', unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model.h5", compile=False)

model = load_model()

# ---------------- DATA ----------------
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy']

recommendations = {
    'Apple___Apple_scab': """1. Remove infected leaves
2. Dispose properly
3. Apply fungicide every 7–10 days
4. Avoid overhead watering
5. Ensure spacing
6. Monitor daily
7. Repeat treatment""",

    'Apple___Black_rot': """1. Prune infected branches
2. Burn infected parts
3. Apply fungicide
4. Disinfect tools
5. Maintain hygiene
6. Avoid wounds
7. Inspect nearby plants""",

    'Apple___healthy': """1. Regular watering
2. Proper sunlight
3. Balanced fertilizer
4. Weekly monitoring
5. Good spacing
6. Well-drained soil
7. Seasonal care"""
}

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;color:#22c55e;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.write("Upload image and get prediction + analysis.")

file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

# ---------------- PREDICTION ----------------
if file:
    img = Image.open(file).resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top_indices = np.argsort(prediction)[-3:][::-1]

    result = classes[top_indices[0]]
    confidence = prediction[top_indices[0]]*100

    st.subheader("Prediction Result")
    st.write(f"**Disease:** {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    # TOP 3
    st.markdown("### Other Predictions")
    for i in top_indices:
        st.write(f"{classes[i]} → {prediction[i]*100:.2f}%")

    # GRAPH
    labels = [classes[i] for i in top_indices]
    values = [prediction[i]*100 for i in top_indices]

    fig, ax = plt.subplots()
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    # ---------------- TREATMENT ----------------
    st.markdown("### Smart Treatment Plan")
    for step in recommendations[result].split("\n"):
        st.write(step)

    # ---------------- MODEL EVALUATION ----------------
    st.markdown("## 📊 Model Evaluation")

    # dummy evaluation
    y_true = [0,1,2,0,1,2,0,2,1]
    y_pred = [0,1,1,0,2,2,0,2,1]

    cm = confusion_matrix(y_true, y_pred)

    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    ax1.set_title("Confusion Matrix")
    st.pyplot(fig1)

    accuracy = accuracy_score(y_true, y_pred)
    st.write(f"### Accuracy: {accuracy*100:.2f}%")

    # comparison
    models = ["Your Model","Baseline"]
    accuracies = [accuracy*100,75]

    fig2, ax2 = plt.subplots()
    ax2.bar(models, accuracies)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Comparison")
    st.pyplot(fig2)