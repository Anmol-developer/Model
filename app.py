import os
from flask import Flask, request, jsonify
# from celery import Celery
# from geopy.geocoders import Nominatim
import pandas as pd
from flask_cors import CORS
from datetime import date
from pymongo import MongoClient
from bson.objectid import ObjectId
app = Flask(__name__, static_url_path='/static')

# Set up MongoDB
client = MongoClient("mongodb+srv://AMalfez:AMalfez%402003@cluster0.h28yihv.mongodb.net")
db = client['PVcast']
collection = db['users']

# app.config["CELERY_BROKER_URL"] = "tcp://localhost" #implimenting celery for background calculation and zeromq as message broker
# app.config['CELERY_RESULT_BACKEND'] = 'mongodb+srv://AMalfez:AMalfez%402003@cluster0.h28yihv.mongodb.net/task'
# celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
# celery.conf.update(app.config)

CORS(app,origins='http://localhost:3000') #to allow requests from client 
app.config['UPLOAD_FOLDER'] = 'uploads'  # Set the upload folder directory
app.config['ALLOWED_EXTENSIONS'] = {'xlsx','csv'}  # Set the allowed file extensions
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

time_array = [
    "07:00",
    "07:15",
    "07:30",
    "07:45",
    "08:00",
    "08:15",
    "08:30",
    "08:45",
    "09:00",
    "09:15",
    "09:30",
    "09:45",
    "10:00",
    "10:15",
    "10:30",
    "10:45",
    "11:00",
    "11:15",
    "11:30",
    "11:45",
    "12:00",
    "12:15",
    "12:30",
    "12:45",
    "13:00",
    "13:15",
    "13:30",
    "13:45",
    "14:00",
    "14:15",
    "14:30",
    "14:45",
    "15:00",
    "15:15",
    "15:30",
    "15:45",
    "16:00",
    "16:15",
    "16:30",
    "16:45",
    "17:00",
  ]

       
@app.route('/api/flask/processing', methods=['POST'])
def process_file():
    nextdate = request.args.get('nextdate')
    userId = request.args.get('_id')
    if 'file' not in request.files:
        return jsonify(error='No file uploaded.')
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error='No file selected.')
    
    try:
        # forecast.delay(file)
        df = pd.read_excel(file)
        X=df[['Ambient Temperature','Relative Humidity','Wind Speed','Cloud Cover']]
        Y=df[['Solar Irradiation']]
        Y=Y.values.ravel()
        # Perform data processing on the df dataframe here
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from lightgbm import LGBMRegressor
        from sklearn.svm import SVR

        train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=42, shuffle=False, test_size=0.2)

        # Random Forest
        rf_model = RandomForestRegressor(random_state=300)
        rf_model.fit(train_X, train_y)
        rf_preds = pd.DataFrame(rf_model.predict(val_X))

        # KNN
        knn_model = KNeighborsRegressor(n_neighbors=25)
        knn_model.fit(train_X, train_y)
        knn_preds = pd.DataFrame(knn_model.predict(val_X))

        # LGBM
        lgbm_model = LGBMRegressor(n_estimators=25)
        lgbm_model.fit(train_X, train_y)
        lgbm_preds = pd.DataFrame(lgbm_model.predict(val_X))

        # SVR
        svr_model = SVR(kernel='linear', C=1, gamma="scale", epsilon=0.0001)
        svr_model.fit(train_X, train_y)
        svr_preds = pd.DataFrame(svr_model.predict(val_X))

        # sliced_svr = svr_preds[0][0]
        sliced_svr = []
        for pred in range(41):
            sliced_svr.append(svr_preds[0][pred])

        sliced_lgbm = []
        for pred in range(41):
            sliced_lgbm.append(lgbm_preds[0][pred])

        sliced_rf = []
        for pred in range(41):
            sliced_rf.append(rf_preds[0][pred])

        sliced_knn = []
        for pred in range(41):
            sliced_knn.append(knn_preds[0][pred])

        temp={
            'date': date.today(),
            '_id': ObjectId(),
            'randomforest':[],
            'svr':[],
            'knn':[],
            'lgbm':[]
        }

        for i in range(len(time_array)):
            temp['randomforest'].append({
                'date':nextdate, 'time':time_array[i], 'PredictedPV':sliced_rf[i]
            })
            temp['svr'].append({
                'date':nextdate, 'time':time_array[i], 'PredictedPV':sliced_svr[i]
            })
            temp['knn'].append({
                'date':nextdate, 'time':time_array[i], 'PredictedPV':sliced_knn[i]
            })
            temp['lgbm'].append({
                'date':nextdate, 'time':time_array[i], 'PredictedPV':sliced_lgbm[i]
            })

        collection.update_one({'_id': ObjectId(userId)}, {'$push': {'PredictedData': temp}})
        return jsonify({'message': 'Success'}), 200

    except Exception as e:
        return jsonify(error=str(e))
    

@app.route('/result')
def result():
    # Read the CSV file
    filename = 'processed_data_LGBM.csv'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(file_path).values.tolist()

    # Pass the data to the template context
    return data

@app.route('/testing', methods=['GET'])
def testing():
    #do some operations
    # new_document = request.get_json()
    collection.insert_one({'testing':'testing'})
    return jsonify({'message': 'Document added successfully'})


if __name__ == '__main__':
    app.run(debug=True)
