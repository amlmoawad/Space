from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

# load the model from disk
filename = 'Logistic Regression2.pkl'
clf = pickle.load(open(filename, 'rb'))
mms=pickle.load(open('MinMaxS2.pkl','rb'))
app = Flask(__name__,template_folder="templates")

@app.route('/')
def home():
	return render_template('form1.html')

@app.route('/result',methods=['POST'])
def result():

    if request.method == 'POST':
    
        message1 = float(request.form['est_diameter_min'])
        message2 = float(request.form['est_diameter_max'])
        message3 = float(request.form['relative_velocity'])
        message4 = float(request.form['miss_distance'])
        message5 = float(request.form['absolute_magnitude'])
        #message = request.args.get('message')

        data = [message1,message2,message3,message4,message5]
        transformedData = mms.transform([data])
        my_prediction = clf.predict(transformedData)
        # print("my_prediction: " ,my_prediction)
        status=None
        if my_prediction[0]==False:
              status="Not Harmful"
        else:
              status="Harmful"
    return render_template('result.html',prediction = status)



if __name__ == '__main__':
	app.run(debug=True)