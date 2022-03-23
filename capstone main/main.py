from re import template
from flask import Flask,render_template,request


app=Flask(__name__)
import pickle

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()
 
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/probabilitydetector',methods=["GET","POST"])
def covid_prob_dect():
    if request.method=="POST":
        mydict=request.form
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        bodypain=int(mydict['bodypain'])
        loss=int(mydict['loss-taste-smell'])
        soreThroat=int(mydict['soreThroat'])
        headache=int(mydict['head-ache'])
        diffbreathing=int(mydict['diff-breathing'])
# Fever	Body Pain	Age	Lost of smell or taste	Sore throat	headache	Difficulty in breathing
        inputFeatures=[fever,bodypain,age,loss,soreThroat,headache,diffbreathing]
        infprob=clf.predict_proba([inputFeatures])[0][1]
        print(infprob)
        return render_template("show.html",infprob=round(infprob*100))

    return render_template("probdetector.html")



if __name__=="__main__":
    app.run(debug=True)
