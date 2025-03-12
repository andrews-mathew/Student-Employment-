import gradio as gr
import pickle
import numpy as np

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' exist.")
    exit()  # Exit if files are missing

def predict_employability(*features):
    try:
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][prediction[0]]
        if prediction[0] == 1:
            return f"Employability Prediction: Employable (Probability: {probability:.2f})"
        else:
            return f"Employability Prediction: Less Employable (Probability: {probability:.2f})"
    except Exception as e:
        return f"Error during prediction: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Student Employability Prediction")
    gr.Markdown(
        "Enter your ratings for the given parameters (1-5) using the sliders below. "
        "Click 'Submit' to check if the student is **Employable** or **Less Employable**."
    )

    with gr.Row():
        with gr.Column():
            sliders = [
                gr.Slider(1, 5, value=3, step=1, label="General Appearance"),
                gr.Slider(1, 5, value=3, step=1, label="Manner of Speaking"),
                gr.Slider(1, 5, value=3, step=1, label="Physical Condition"),
                gr.Slider(1, 5, value=3, step=1, label="Mental Alertness"),
                gr.Slider(1, 5, value=3, step=1, label="Self-Confidence"),
                gr.Slider(1, 5, value=3, step=1, label="Ability to Present Ideas"),
                gr.Slider(1, 5, value=3, step=1, label="Communication Skills"),
                gr.Slider(1, 5, value=3, step=1, label="Student Performance Rating"),
            ]
        with gr.Column():
            output = gr.Textbox(label="Employability Status")

    submit = gr.Button("Submit")
    submit.click(predict_employability, inputs=sliders, outputs=output)

    examples = [
        [5, 5, 5, 5, 5, 5, 5, 5],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [3, 4, 2, 5, 3, 4, 4, 3],
    ]
    gr.Examples(examples, inputs=sliders, outputs=output)
    gr.Markdown("Example Input Data")

demo.launch()
