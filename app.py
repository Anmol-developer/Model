import os
from flask import Flask, render_template, request, jsonify, send_from_directory,redirect,url_for
# from geopy.geocoders import Nominatim
# from flask import send_file
from flask_cors import CORS
import pandas as pd
# import matplotlib.pyplot as plt
app = Flask(__name__, static_url_path='/static')
CORS(app,origins='http://localhost:3000')
app.config['UPLOAD_FOLDER'] = 'uploads'  # Set the upload folder directory
app.config['ALLOWED_EXTENSIONS'] = {'xlsx','csv'}  # Set the allowed file extensions
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')
        
       
@app.route('/api/flask/processing', methods=['POST'])
def process_file():
    # userId = request.args.get('userId')
    if 'file' not in request.files:
        return jsonify(error='No file uploaded.')
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error='No file selected.')
    
    try:
        df = pd.read_excel(file)
        X=df[['Ambient Temperature','Relative Humidity','Wind Speed','Cloud Cover','Ambient Temperature']]
        Y=df[['Solar Irradiation']]
        Y=Y.values.ravel()
        # Perform data processing on the df dataframe here
        from sklearn.model_selection import train_test_split
        train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=42,shuffle=False,test_size=0.2)
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        rf_model = RandomForestRegressor(random_state=300)
        
        # fit your model
        rf_model.fit(train_X,train_y)
        # Predict
        preds = pd.DataFrame(rf_model.predict(val_X))
        

        # Save the processed data as a file
        processed_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        preds.to_csv(processed_filename, index=False)
        
        # Redirect to the result page with the filename as a query parameter
        return redirect(url_for('result', filename=processed_filename))
        # return redirect(f"http://localhost:8080/forecast/{userId}?model=Randomforest")

        
    except Exception as e:
        return jsonify(error=str(e))

@app.route('/result')
def result():
    # Read the CSV file
    filename = 'uploads/processed_data.csv'
    data = pd.read_csv(filename)
    
    # Convert the data to a list of rows
    data_rows = data.values.tolist()
    
    # Pass the data to the template context
    return jsonify(data_rows)

@app.route('/download_result/<filename>')
def download_result(filename):
    uploads_folder = app.config['UPLOAD_FOLDER']
    return send_from_directory(uploads_folder, filename, as_attachment=True)

@app.route('/upload_file/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)





if __name__ == '__main__':
    app.run(debug=True)

