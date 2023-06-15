import os
from flask import Flask, render_template, request, jsonify, send_from_directory,redirect,url_for
from geopy.geocoders import Nominatim
import pandas as pd
from flask_cors import CORS
app = Flask(__name__, static_url_path='/static')
CORS(app,origins='http://localhost:3000')
app.config['UPLOAD_FOLDER'] = 'uploads'  # Set the upload folder directory
app.config['ALLOWED_EXTENSIONS'] = {'xlsx','csv'}  # Set the allowed file extensions
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# @app.route('/')
# def index():
#     return render_template('index.html')
#     location = request.args.get('location')
#     geolocator = Nominatim(user_agent='my-app')

#     try:
#         location_data = geolocator.geocode(location, timeout=10)
#         if location_data:
#             latitude = location_data.latitude
#             longitude = location_data.longitude
#             return jsonify(latitude=latitude, longitude=longitude)
#         else:
#             return jsonify(error='Coordinates not found for provided location.')
#     except Exception as e:
#         return jsonify(error='An error occurred while fetching coordinates.')
        
       
@app.route('/api/flask/processing', methods=['POST'])
def process_file():
    model = request.args.get('model')
    if 'file' not in request.files:
        return jsonify(error='No file uploaded.')
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error='No file selected.')
    
    try:
        df = pd.read_excel(file)
        # X=df[['Ambient Temperature','Relative Humidity','Wind Speed','Cloud Cover','Ambient Temperature']]
        X=df[['Ambient Temperature','Relative Humidity','Wind Speed','Cloud Cover']]
        Y=df[['Solar Irradiation']]
        Y=Y.values.ravel()
        # Perform data processing on the df dataframe here
        from sklearn.model_selection import train_test_split
        train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=42,shuffle=False,test_size=0.2)
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        rf_model = RandomForestRegressor(random_state=300)
        
        # fit your model_RF
        rf_model.fit(train_X,train_y)
        # Predict
        preds = pd.DataFrame(rf_model.predict(val_X))
        #KNN
        knn=KNeighborsRegressor(n_neighbors=25)
        knn_model=knn.fit(train_X,train_y)
        # fit your model
        knn_model=knn.fit(train_X,train_y)
        # Calculate the mean absolute error of your Random Forest model on the validation data
        y_knn_pred=knn_model.predict(val_X)
        
        #LGBM
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(n_estimators=25)
        lgbm.fit(train_X,train_y)
        lgbm_pred = lgbm.predict(val_X)
        
        #SVR
        from sklearn.svm import SVR
        svr_rbf = SVR(kernel='linear', C=1, gamma="scale",epsilon=0.0001)
        svr_rbf.fit(train_X,train_y)
        svr_pred = svr_rbf.predict(val_X)
        
        # Save the processed data as a file
        processed_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_RF.csv')
        processed_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_KNN.csv')
        processed_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_LGBM.csv')
        processed_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data_SVR.csv')
        preds.to_csv(processed_filename1, index=False)
        y_knn_pred.to_csv(processed_filename2, index=False)
        lgbm_pred.to_csv(processed_filename3, index=False)
        svr_pred.to_csv(processed_filename4, index=False)
        
        # Redirect to the result page with the filename as a query parameter
        # redirect(url_for('result/randomforest', filename=processed_filename1))

        def switch_case(model):
            if model == 'randomforest':
                return redirect(url_for('result/randomforest', filename=processed_filename1))

            elif model == 'knn':
                return redirect(url_for('result/knn', filename=processed_filename2))

            elif model == 'lgbm':
                return redirect(url_for('result/lgbm', filename=processed_filename3))

            elif model == 'svr':
                return redirect(url_for('result/svr', filename=processed_filename4))

        
    except Exception as e:
        return jsonify(error=str(e))
    
@app.route('/result/randomforest')
def randomForest():
    # Read the CSV file
    filename = 'uploads/processed_data_RF.csv'
    data = pd.read_csv(filename)
    
    # Convert the data to a list of rows
    data_rows = data.values.tolist()
    
    # Pass the data to the template context
    return data_rows

@app.route('/result/knn')
def knn():
    # Read the CSV file
    filename = 'uploads/processed_data_KNN.csv'
    data = pd.read_csv(filename)
    
    # Convert the data to a list of rows
    data_rows = data.values.tolist()
    
    # Pass the data to the template context
    return data_rows

@app.route('/result/lgbm')
def lgbm():
    # Read the CSV file
    filename = 'uploads/processed_data_LGBM.csv'
    data = pd.read_csv(filename)
    
    # Convert the data to a list of rows
    data_rows = data.values.tolist()
    
    # Pass the data to the template context
    return data_rows

@app.route('/result/svr')
def svr():
    # Read the CSV file
    filename = 'uploads/processed_data_SVR.csv'
    data = pd.read_csv(filename)
    
    # Convert the data to a list of rows
    data_rows = data.values.tolist()
    
    # Pass the data to the template context
    return data_rows


if __name__ == '__main__':
    app.run(debug=True)

