import os
from flask import Flask, request, jsonify,redirect,url_for
# from geopy.geocoders import Nominatim
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

        # Redirect to the result page with the filename as a query parameter
        # redirect(url_for('result/randomforest', filename=processed_filename1))

        # def switch_case(model):
        #     if model == 'randomforest':
        #         return redirect(url_for('result/randomforest', filename=processed_filename1))

        #     elif model == 'knn':
        #         return redirect(url_for('result/knn', filename=processed_filename2))

        #     elif model == 'lgbm':
        #         return redirect(url_for('result/lgbm', filename=processed_filename3))

        #     elif model == 'svr':
        #         return redirect(url_for('result/svr', filename=processed_filename4))
        return redirect(url_for('result', rf_filename=rf_filename, knn_filename=knn_filename, lgbm_filename=lgbm_filename, svr_filename=svr_filename))


        
    except Exception as e:
        return jsonify(error=str(e))
    
# @app.route('/result/randomforest')
# def randomForest():
#     # Read the CSV file
#     filename = 'uploads/processed_data_RF.csv'
#     data = pd.read_csv(filename)
    
#     # Convert the data to a list of rows
#     data_rows = data.values.tolist()
    
#     # Pass the data to the template context
#     return data_rows

@app.route('/result')
def result():
    # Read the CSV file
    filename = 'processed_data_LGBM.csv'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(file_path).values.tolist()

    # Pass the data to the template context
    return data

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

