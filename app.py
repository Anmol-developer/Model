import os
from flask import Flask, request, jsonify,redirect,url_for
# from celery import Celery
# from geopy.geocoders import Nominatim
import pandas as pd
from flask_cors import CORS
app = Flask(__name__, static_url_path='/static')

# Set up MongoDB
# client = pymongo.MongoClient("mongodb+srv://AMalfez:AMalfez%402003@cluster0.h28yihv.mongodb.net")
# db = client['task']
# collection = db['tasks']

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

     
# @celery.task()
# def forecast(file):
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
    rf_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_RF.csv')
    rf_preds.to_csv(rf_filename, index=False)
    # KNN
    knn_model = KNeighborsRegressor(n_neighbors=25)
    knn_model.fit(train_X, train_y)
    knn_preds = pd.DataFrame(knn_model.predict(val_X))
    knn_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_knn.csv')
    knn_preds.to_csv(knn_filename, index=False)
    # LGBM
    lgbm_model = LGBMRegressor(n_estimators=25)
    lgbm_model.fit(train_X, train_y)
    lgbm_preds = pd.DataFrame(lgbm_model.predict(val_X))
    lgbm_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_lgbm.csv')
    lgbm_preds.to_csv(lgbm_filename, index=False)
    # SVR
    svr_model = SVR(kernel='linear', C=1, gamma="scale", epsilon=0.0001)
    svr_model.fit(train_X, train_y)
    svr_preds = pd.DataFrame(svr_model.predict(val_X))
    svr_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_svr.csv')
    svr_preds.to_csv(svr_filename, index=False)
    return 'Done forecasting!'


       
@app.route('/api/flask/processing', methods=['POST'])
def process_file():
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
        rf_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_RF.csv')
        rf_preds.to_csv(rf_filename, index=False)

        # KNN
        knn_model = KNeighborsRegressor(n_neighbors=25)
        knn_model.fit(train_X, train_y)
        knn_preds = pd.DataFrame(knn_model.predict(val_X))
        knn_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_knn.csv')
        knn_preds.to_csv(knn_filename, index=False)

        # LGBM
        lgbm_model = LGBMRegressor(n_estimators=25)
        lgbm_model.fit(train_X, train_y)
        lgbm_preds = pd.DataFrame(lgbm_model.predict(val_X))
        lgbm_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_lgbm.csv')
        lgbm_preds.to_csv(lgbm_filename, index=False)

        # SVR
        svr_model = SVR(kernel='linear', C=1, gamma="scale", epsilon=0.0001)
        svr_model.fit(train_X, train_y)
        svr_preds = pd.DataFrame(svr_model.predict(val_X))
        svr_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_svr.csv')
        svr_preds.to_csv(svr_filename, index=False)

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


if __name__ == '__main__':
    app.run(debug=True)
