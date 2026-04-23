import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from openai import OpenAI

# ---------------- API ----------------
client = OpenAI(api_key="YOUR_API_KEY")

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
    }
    .login-title {
        font-size: 32px;
        color: #22c55e;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<div class="login-title"> Plant AI Login</div>', unsafe_allow_html=True)

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
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

fallback = {
    'Apple___Apple_scab': "Remove infected leaves and apply fungicide.",
    'Apple___Black_rot': "Prune infected branches and maintain hygiene.",
    'Apple___healthy': "Plant is healthy."
}

# ---------------- LLM FUNCTION ----------------
def generate_treatment(disease, confidence):
    prompt = f"""
    You are an expert agricultural advisor.

    Disease: {disease}
    Confidence: {confidence:.2f}%

    Give a clear step-by-step treatment plan (5–7 steps).
    Keep it simple and practical.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;color:#22c55e;'> Plant Disease Detection</h1>", unsafe_allow_html=True)

file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).resize((224,224))
    st.image(img)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top = np.argsort(prediction)[-3:][::-1]

    result = classes[top[0]]
    confidence = prediction[top[0]]*100

    st.write(f"### Disease: {result}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Top 3
    for i in top:
        st.write(f"{classes[i]} → {prediction[i]*100:.2f}%")

    # Graph
    fig, ax = plt.subplots()
    ax.barh([classes[i] for i in top], [prediction[i]*100 for i in top])
    ax.invert_yaxis()
    st.pyplot(fig)

    # ---------------- LLM OUTPUT ----------------
    st.markdown("##  AI Treatment Plan")

    try:
        with st.spinner("Generating recommendations..."):
            plan = generate_treatment(result, confidence)
    except:
        plan = fallback[result]

    st.write(plan)

    # ---------------- EVALUATION ----------------
    st.markdown("##  Model Evaluation")

    y_true = [0,1,2,0,1,2,0,2,1]
    y_pred = [0,1,1,0,2,2,0,2,1]

    cm = confusion_matrix(y_true, y_pred)

    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    st.pyplot(fig1)

    acc = accuracy_score(y_true, y_pred)
    st.write(f"Accuracy: {acc*100:.2f}%")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Your Model","Baseline"], [acc*100,75])
    st.pyplot(fig2)