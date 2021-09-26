from flask import Flask, request, jsonify, render_template, session, url_for, redirect
from tensorflow.keras.models import load_model
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
import joblib
import numpy as np

def return_prediction(model, scalar, sample_json):
    
    s_len = sample_json["sepal_length"]
    s_width = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_width = sample_json["petal_width"]
    
    flower = [[s_len, s_width, p_len, p_width]]
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    flower = scalar.transform(flower)
    
    class_ind = model.predict(flower)[0]
    
    class_ind= np.argmax(class_ind)
    
    return classes[class_ind]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlaskForm):

    sep_len = TextField("Sepal Length")
    sep_width = TextField("Sepal Width")
    pet_len = TextField("Petal Length")
    pet_width = TextField("Petal Width")

    submit = SubmitField("Predict")


@app.route("/", methods=['GET','POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['sep_len'] = form.sep_len.data
        session['sep_width'] = form.sep_width.data
        session['pet_len'] = form.pet_len.data
        session['pet_width'] = form.pet_width.data

        return redirect(url_for("prediction"))

    return render_template("home.html", form = form)



flower_model = load_model('final_iris_model.h5', compile=False)
flower_scalar = joblib.load('iris_scalar.pkl')

@app.route("/prediction")
def prediction():
	# content = request.json
	# results = return_prediction(flower_model, flower_scalar, content)
	# return jsonify(results)
    content = {}
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_width'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['sep_width'])

    results = return_prediction(flower_model, flower_scalar, content)

    return render_template("prediction.html", results= results)


if __name__ == '__main__':
	app.run(debug=True)
   