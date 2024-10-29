import os
from flask import render_template, request, redirect, flash, send_file, url_for
from datetime import datetime
import pandas as pd
from preprosessing.preprosessor import main  # Import the main function from preprosessor
import matplotlib.pyplot as plt
from app import app
from modelling.new_revenue_forecast_multicaptive_2 import RevenueForecastMulticaptive2
from modelling.revenue_forecast_runrate import RevenueForecastRunrate 

import csv
from flask import make_response
import io


# Ensure the directory for uploads exists
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/download_forecast', methods=['GET'])
def download_forecast():
    try:
        csv_filename = os.path.join(UPLOAD_FOLDER, 'forecast_results.csv')
        return send_file(csv_filename, as_attachment=True, attachment_filename='forecast_results.csv')
    except Exception as e:
        return f"Error generating CSV: {str(e)}", 500


UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preprocess')
def preprocess():
    return render_template('preprocess.html')

def save_file(file):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return filepath

def process_files(bil_path, tlx_path, calendar_path):
    current_date = datetime.now().strftime('%Y%m%d')
    output_datafile_path = os.path.join(UPLOAD_FOLDER, f'output_datafile_{current_date}.csv')
    processed_bil_path = os.path.join(UPLOAD_FOLDER, f'processed_bil_{current_date}.csv')
    processed_tlx_path = os.path.join(UPLOAD_FOLDER, f'processed_tlx_{current_date}.csv')
    main(bil_path, tlx_path, calendar_path, output_datafile_path, processed_bil_path, processed_tlx_path)
    return output_datafile_path, processed_bil_path, processed_tlx_path

@app.route('/process', methods=['POST'])
def process():
    if 'bil_file' not in request.files or 'tlx_file' not in request.files or 'calendar_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    try:
        bil_file = request.files['bil_file']
        tlx_file = request.files['tlx_file']
        calendar_file = request.files['calendar_file']
        
        bil_path = save_file(bil_file)
        tlx_path = save_file(tlx_file)
        calendar_path = save_file(calendar_file)
        
        output_datafile_path, processed_bil_path, processed_tlx_path = process_files(bil_path, tlx_path, calendar_path)
        
        return render_template('download.html', 
                               output_datafile=output_datafile_path, 
                               processed_bil=processed_bil_path, 
                               processed_tlx=processed_tlx_path)
    except Exception as e:
        flash(f"Error processing files: {str(e)}")
        return redirect(request.url)


@app.route('/forecast', methods=['GET', 'POST'])
def forecast_page():
    if request.method == 'POST':
        try:
            # Train the model
            service = RevenueForecastMulticaptive2() 
            service.train_model('2021-10-01', '2024-09-30') 
            

            # Forecasting
            forecast_fin, forecast_ind, forecast_ns, forecast_total = service.forecast('2024-11-01', '2025-11-30')
            plix = service.validate(forecast_fin, forecast_ind, forecast_ns, '2024-11-01', '2025-11-30', save_errors=True)

            plix.reset_index(inplace=True)
            validation_results_dict = plix.to_dict(orient='records')

            forecast_fin_aug, forecast_ind_aug, forecast_ns_aug, forecast_total_aug = service.forecast('2024-08-01', '2024-08-31')
            plix_aug = service.validate(forecast_fin_aug, forecast_ind_aug, forecast_ns_aug, '2024-08-01', '2024-08-31', save_errors=True)
            plix_aug.reset_index(inplace=True)
            validation_results_dict_aug = plix_aug.to_dict(orient='records')

            # Forecasting and validating for September 2024
            forecast_fin_sep, forecast_ind_sep, forecast_ns_sep, forecast_total_sep = service.forecast('2024-09-01', '2024-09-30')
            plix_sep = service.validate(forecast_fin_sep, forecast_ind_sep, forecast_ns_sep, '2024-09-01', '2024-09-30', save_errors=True)
            plix_sep.reset_index(inplace=True)
            validation_results_dict_sep = plix_sep.to_dict(orient='records')

            # Forecasting and validating for October 2024
            forecast_fin_oct, forecast_ind_oct, forecast_ns_oct, forecast_total_oct = service.forecast('2024-10-01', '2024-10-31')
            plix_oct = service.validate(forecast_fin_oct, forecast_ind_oct, forecast_ns_oct, '2024-10-01', '2024-10-31', save_errors=True)
            plix_oct.reset_index(inplace=True)
            validation_results_dict_oct = plix_oct.to_dict(orient='records')
            # print()
            # print("AUG")
            # print(validation_results_dict_aug)
            # print()
            # print(validation_results_dict_oct)
            # print(validation_results_dict_sep)

            service2 = RevenueForecastRunrate(use_trend=False) 
            service2.train_model('2021-10-01', '2024-09-30')

            forecast_fin, forecast_ind, forecast_ns, forecast_total = service2.forecast('2024-11-01', '2025-11-30')
            plix2 = service2.validate(forecast_fin, forecast_ind, forecast_ns, '2024-11-01', '2025-11-30', save_errors=True)

            validation_results_dict2 = plix2.to_dict(orient='records')

            forecast_fin_aug, forecast_ind_aug, forecast_ns_aug, forecast_total_aug = service2.forecast('2024-08-01', '2024-08-31')
            plix_aug2 = service2.validate(forecast_fin_aug, forecast_ind_aug, forecast_ns_aug, '2024-08-01', '2024-08-31', save_errors=True)
            plix_aug2.reset_index(inplace=True)
            validation_results_dict_aug2 = plix_aug2.to_dict(orient='records')

            # Forecasting and validating for September 2024
            forecast_fin_sep, forecast_ind_sep, forecast_ns_sep, forecast_total_sep = service2.forecast('2024-09-01', '2024-09-30')
            plix_sep2 = service2.validate(forecast_fin_sep, forecast_ind_sep, forecast_ns_sep, '2024-09-01', '2024-09-30', save_errors=True)
            plix_sep2.reset_index(inplace=True)
            validation_results_dict_sep2 = plix_sep2.to_dict(orient='records')

            # Forecasting and validating for October 2024
            forecast_fin_oct, forecast_ind_oct, forecast_ns_oct, forecast_total_oct = service2.forecast('2024-10-01', '2024-10-31')
            plix_oct2 = service2.validate(forecast_fin_oct, forecast_ind_oct, forecast_ns_oct, '2024-10-01', '2024-10-31', save_errors=True)
            plix_oct2.reset_index(inplace=True)
            validation_results_dict_oct2 = plix_oct2.to_dict(orient='records')

            print(validation_results_dict_aug)

        
            # Save to CSV
            csv_filename = os.path.join(UPLOAD_FOLDER, 'forecast_results.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Date', 'Actual FIN', 'Forecast FIN', 'Error% FIN',
                                 'Actual IND', 'Forecast IND', 'Error% IND',
                                 'Actual NS', 'Forecast NS', 'Error% NS',
                                 'Actual Total', 'Forecast Total', 'Error% Total'])

                for entry in validation_results_dict:
                    writer.writerow([
                        entry['Date'].strftime('%Y-%m-%d'),
                        entry['Actual FIN'],
                        entry['Forecast FIN'],
                        entry['Error% FIN'],
                        entry['Actual IND'],
                        entry['Forecast IND'],
                        entry['Error% IND'],
                        entry['Actual NS'],
                        entry['Forecast NS'],
                        entry['Error% NS'],
                        entry['Actual Total'],
                        entry['Forecast Total'],
                        entry['Error% Total']
                    ])

            return render_template('forecast_results.html', 
                                   forecast_results_multicaptive=validation_results_dict,
                                   forecast_results_runrate=validation_results_dict2, aug=validation_results_dict_aug, 
                                   oct=validation_results_dict_oct, 
                                   sep=validation_results_dict_sep,
                                   aug2=validation_results_dict_aug2, 
                                   oct2=validation_results_dict_oct2, 
                                   sep2=validation_results_dict_sep2  )

        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'danger')
            return redirect(url_for('home'))  # Redirect back to home on error

    return redirect(url_for('home'))  # Redirect to home if GET request




@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)
