from flask import Flask, render_template, request
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import flask

diabetes = pandas.read_csv ( "diabetes.csv" )

app = Flask ( __name__ )


@app.route ( '/' )
def index():
    return render_template ( "index.html" )


@app.route ( '/About' )
def About():
    return render_template ( "About.html" )


@app.route ( '/register' )
def register():
    return render_template ( "register.html" )


@app.route ( '/login' )
def login():
    return render_template ( "login.html" )


@app.route ( '/admin' )
def admin():
    return render_template ( "admin.html" )


@app.route ( '/prediction' )
def prediction():
    return render_template ( "predict.html" )


@app.route ( '/test' )
def test():
    return render_template ( "test.py" )


@app.route ( '/DPP', methods=[ 'POST' ] )
def DPP():
    # pdb.set_trace()
    age = int ( request.form [ 'Age' ] )
    glucose = int ( request.form [ 'Glucose' ] )
    insulin = int ( request.form [ 'Insulin' ] )
    bp = int ( request.form [ 'BloodPressure' ] )
    bmi = float ( request.form [ 'BodyMassIndex' ] )
    preg = int ( request.form [ 'Pregnancies' ] )
    hd = request.form [ 'Heredity' ]

    if (hd == "Yes"):
        dpf = 1
    else:
        dpf = 0

    knn = None
    X_train, X_test, y_train, y_test = train_test_split ( diabetes.loc [ :, diabetes.columns != 'Outcome' ],
                                                          diabetes [ 'Outcome' ],
                                                          stratify=diabetes [ 'Outcome' ], random_state=66 )

    neighbors_settings = range ( 1, 11 )

    for n_neighbors in neighbors_settings:
        # build the model
        knn = KNeighborsClassifier ( n_neighbors=n_neighbors )
        knn.fit ( X_train, y_train )

    result = knn.predict ( [ [ preg, glucose, bp, insulin, bmi, dpf, age ] ] )
    result1 = knn.predict_proba ( [ [ preg, glucose, bp, insulin, bmi, dpf, age ] ] )

    if (result == 0):
        answer = "You do NOT have Diabetes."
    else:
        answer = "You do have Diabetes."

    return render_template ( 'Result.html', value=int(result1[0][1]*100))


if __name__ == '__main__':
    app.run ( debug=True )
