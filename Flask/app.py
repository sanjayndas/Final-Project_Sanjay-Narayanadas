import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# making list of survey columns
surveys =['Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Food and drink', 'Online boarding',
       'Seat comfort', 'Inflight entertainment', 'On-board service',
       'Leg room service', 'Baggage handling', 'Checkin service',
       'Inflight service', 'Cleanliness']
survey_ids = ['Inflight_wifi_service', 'Departure/Arrival_time_convenient',
       'Ease_of_Online booking', 'Food_and_drink', 'Online_boarding',
       'Seat_comfort', 'Inflight_entertainment', 'On-board_service',
       'Leg_room_service', 'Baggage_handling', 'Checkin_service',
       'Inflight_service', 'Cleanliness']
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/prediction')
def prediction():
    return render_template('index.html', len = len(surveys), surveys = surveys, survey_ids = survey_ids)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #features entered by user are collected and passed to model created for prediction
    int_features = [float(x) for x in request.form.values()]
    print('request.form.values()')
    print(np.array(int_features))
    final_features = [np.array(int_features)]

    output =[]

    final_features = {'Customer Type': final_features[0][0], 'Age' : final_features[0][1], 'Type of Travel':final_features[0][2], 'Class':final_features[0][3],'Flight Distance':final_features[0][4], 'Inflight wifi service':final_features[0][5], 'Departure/Arrival time convenient':final_features[0][6], 'Ease of Online booking':final_features[0][7], 'Food and drink':final_features[0][8], 'Online boarding':final_features[0][9], 'Seat comfort':final_features[0][10], 'Inflight entertainment':final_features[0][11], 'On-board service':final_features[0][12], 'Leg room service':final_features[0][13], 'Baggage handling':final_features[0][14], 'Checkin service':final_features[0][15], 'Inflight service':final_features[0][16], 'Cleanliness':final_features[0][17], 'Arrival Delay in Minutes':final_features[0][18]}
    final_features = pd.DataFrame(data=final_features, index=[0])
    prediction = model.predict(final_features)

    if (prediction[0]==0):
        output = 'The passenger is dissatisfied or gave a neutral response'
    elif(prediction[0]==1):
        output = 'The passenger is Satisfied'

    # the predicted value is returned to the html
    return render_template('satisfaction.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run()
