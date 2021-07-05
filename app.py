import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('models\\titanic_rf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        is_mr = request.form.get('is_Mr')
        is_mrs = request.form.get('is_Mrs')
        is_miss = request.form.get('is_Miss')
        is_master = request.form.get('is_Master')
        is_pclass_1 = request.form.get('is_Pclass_1')
        is_pclass_2 = request.form.get('is_Pclass_2')
        is_embark_S = request.form.get('is_embark_S')
        is_embark_C = request.form.get('is_embark_C')
        is_parent_or_child = request.form.get('is_parent_or_child')
        test_data = np.array([is_mr, is_mrs, is_miss, is_master, is_pclass_1, is_pclass_2,\
                              is_embark_C, is_embark_S, is_parent_or_child]).reshape(1, -1)
        prediction = model.predict(test_data)

    return render_template('predict.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
