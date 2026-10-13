import gradio as gr
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load Models
model_student = joblib.load('student_model.pkl')
model_movie = joblib.load('movie_model.pkl')

def predict_score(study_hours, last_score):
    pred = model_student.predict([[study_hours, last_score]])
    return f"Expected Score: {pred[0]:.1f} / 100"

def predict_rating(year, popularity):
    pred = model_movie.predict([[year, popularity]])
    return f"Expected Rating: {pred[0]:.2f} / 10"

# Using a clean Monochrome theme with a side-by-side layout
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("<center><h1>📊 Analytics Hub</h1><p>Evaluate student performance and movie ratings side-by-side.</p></center>")
    
    with gr.Row():
        # Left Column: Student
        with gr.Column(variant="panel"):
            gr.Markdown("### 🎓 Student Exam Predictor")
            study_hours = gr.Number(label="Study Hours")
            last_score = gr.Number(label="Previous Exam Score (0-100)")
            btn_student = gr.Button("Calculate Final Score", variant="primary")
            # Removed show_copy_button for compatibility
            out_student = gr.Textbox(label="Result") 
            btn_student.click(predict_score, inputs=[study_hours, last_score], outputs=out_student)

        # Right Column: Movies
        with gr.Column(variant="panel"):
            gr.Markdown("### 🎬 Movie Rating Predictor")
            year = gr.Number(label="Release Year (e.g., 2014)")
            popularity = gr.Number(label="Popularity Score (e.g., 150.5)")
            btn_movie = gr.Button("Calculate IMDB Rating", variant="primary")
            # Removed show_copy_button for compatibility
            out_movie = gr.Textbox(label="Result") 
            btn_movie.click(predict_rating, inputs=[year, popularity], outputs=out_movie)

if __name__ == "__main__":
    demo.launch()