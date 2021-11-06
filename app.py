import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 0:
        return render_template('prediction.html', prediction_text='It is safe to go out on date. It will not rain.')
    else:
        return render_template('predictionRain.html', prediction_text='It is better to have a date at home.')


if __name__ == '__main__':
    app.run(debug=True, port=8000)