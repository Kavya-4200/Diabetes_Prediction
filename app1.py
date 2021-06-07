import flask
import pandas
import pandas as pd
import pd
from flask import request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

diabetes = pandas.read_csv('diabetes.csv')
del diabetes [ 'DiabetesPedigreeFunction' ]
print ( diabetes.info ( ) )
app = flask.Flask ( __name__ )


@app.route('/')
def index():
    return flask.render_template("index.html")


@app.route('/About')
def About():
    return flask.render_template("About.html")


@app.route('/register')
def register():
    return flask.render_template("register.html")


@app.route('/login')
def login():
    return flask.render_template("login.html")


@app.route('/admin')
def admin():
    return flask.render_template("admin.html")


@app.route('/prediction')
def prediction():
    return flask.render_template("predict.html")


@app.route('/DPP', methods=['POST'])
def DPP():
    pass


knn = None
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],
                                                    stratify=diabetes['Outcome'], random_state=66)

neighbors_settings: range = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

app = flask.Flask(__name__)


class Diabetes(object):
    age = int(request.form['Age'])
    glucose = int(request.form['Glucose'])
    insulin = int(request.form['Insulin'])
    bp = int(request.form['BloodPressure'])
    st = float(request.form['SkinThickness'])
    bmi = float(request.form['BodyMassIndex'])
    preg = int(request.form['Pregnancies'])
    hd = request.form['Heredity']


@app.route('/', methods=['GET', 'POST'])
def main():
    global knn
    form = Diabetes(flask.request.form)
    if flask.request.method == 'POST' and form.validate():
        res = knn.predict_proba(
            {(
                float(form.pregnancies.data),
                float(form.glucose.data),
                float(form.bp.data),
                float(form.skin.data),
                float(form.insulin.data),
                float(form.bmi.data),
                float(form.age.data)
            )}
        )
        for i in res.tolist():
            print(i)
        return flask.render_template("result.html", res=int(res[0][1] * 100))

    return flask.render_template("index.html", form=form)
