from flask import Flask, request, render_template
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the trained models
model_student = joblib.load('student_model.pkl')
model_movie = joblib.load('movie_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_score', methods=['POST'])
def predict_score():
    study_hours = float(request.form['study_hours'])
    last_score = float(request.form['last_score'])
    
    pred = model_student.predict([[study_hours, last_score]])
    result = f"{pred[0]:.1f} / 100 Points"
    
    return render_template('index.html', result_student=result)

@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    year = float(request.form['year'])
    popularity = float(request.form['popularity'])
    
    pred = model_movie.predict([[year, popularity]])
    result = f"{pred[0]:.2f} / 10 Stars"
    
    return render_template('index.html', result_movie=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)