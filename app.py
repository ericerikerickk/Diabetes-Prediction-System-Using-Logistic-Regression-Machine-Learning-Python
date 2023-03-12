# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		output = model.predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
		result_str += "Percentage of being not diabetic: " + "{:.2f}".format(output[0][0]) + "<br />"
		result_str += "Percentage of being diabetic: " + "{:.2f}".format(output[0][1]) + "<br />"
	return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	Pregnancies = request.form.get('Pregnancies')
	Glucose = request.form.get('Glucose')
	BloodPressure = request.form.get('BloodPressure')
	SkinThickness = request.form.get('SkinThickness')
	Insulin = request.form.get('Insulin')
	BMI = request.form.get('BMI')
	DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
	Age = request.form.get('Age')

	Pregnancies = int(Pregnancies)
	Glucose = int(Glucose)
	BloodPressure = int(BloodPressure)
	SkinThickness = int(SkinThickness)
	Insulin = int(Insulin)
	BMI = float(BMI)
	DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
	Age	 =int(Age)
	the_output = the_model(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)