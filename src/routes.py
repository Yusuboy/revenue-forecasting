import os
from os import getenv
from app import app
from flask import render_template, request, redirect, flash, send_file, url_for
from datetime import datetime
import pandas as pd
from preprosessing.preprosessor import main  # Import the main function from preprosessor
from modelling.forecast_utils import ForecastUtils
from modelling.revenue_forecast_sarimax import RevenueForecastSarimax
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'uploads'  # Folder for saving uploaded files

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')


def save_file(file):
    """Saves the uploaded file to the designated upload folder."""
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return filepath

def process_files(bil_path, tlx_path, calendar_path):
    """Processes the uploaded files and generates output."""
    current_date = datetime.now().strftime('%Y%m%d')
    output_datafile_path = os.path.join(UPLOAD_FOLDER, f'output_datafile_path.csv')
    processed_bil_path = os.path.join(UPLOAD_FOLDER, f'processed_bil.csv')
    processed_tlx_path = os.path.join(UPLOAD_FOLDER, f'processed_tlx.csv')

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

        # Save the uploaded files
        bil_path = save_file(bil_file)
        tlx_path = save_file(tlx_file)
        calendar_path = save_file(calendar_file)

        # Process files and get the output paths
        output_datafile_path, processed_bil_path, processed_tlx_path = process_files(bil_path, tlx_path, calendar_path)

        # Render a new template with download links
        return render_template('download.html', 
                               output_datafile=output_datafile_path, 
                               processed_bil=processed_bil_path,
                               processed_tlx=processed_tlx_path)

    except Exception as e:
        flash(f"Error processing files: {str(e)}")
        return redirect(request.url)
    
@app.route('/forecast', methods=['GET'])
def forecast_form():
    """Serve the forecasting input page."""
    return render_template('forecast.html')


@app.route('/forecast', methods=['POST'])
def run_forecast():
    """Handle the forecasting process."""
    if 'data_file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    try:
        data_file = request.files['data_file']
        train_start = request.form['train_start']
        train_end = request.form['train_end']
        forecast_start = request.form['forecast_start']
        forecast_end = request.form['forecast_end']
        print(1)

        # Save the uploaded processed data file
        data_file_path = save_file(data_file)

        # Load the data using ForecastUtils
        df = ForecastUtils.load_data(data_file_path)
        print(2)

        # Initialize the forecasting model
        forecast_model = RevenueForecastSarimax()

        print(3)
        # Train the model using the specified period
        forecast_model.train_model(train_start, train_end)

        print(4)
        # Run the forecast for the specified period
        forecast_fin, forecast_ind, forecast_ns, forecast_total = forecast_model.forecast(forecast_start, forecast_end)

        # Save the plot to a file in the UPLOAD_FOLDER
        plot_path = save_forecast_plot(df, forecast_fin, forecast_start, forecast_end)

        # Store the results in the session to pass to the results page
        return redirect(url_for('forecast_results', 
                                forecast_fin=forecast_fin,
                                forecast_ind=forecast_ind,
                                forecast_ns=forecast_ns,
                                forecast_total=forecast_total,
                                plot_path=plot_path))

    except Exception as e:
        flash(f"Error running forecast: {str(e)}")
        return redirect(request.url)


@app.route('/forecast/results')
def forecast_results():
    """Display the forecasting results."""
    forecast_fin = request.args.get('forecast_fin', type=float)
    forecast_ind = request.args.get('forecast_ind', type=float)
    forecast_ns = request.args.get('forecast_ns', type=float)
    forecast_total = request.args.get('forecast_total', type=float)
    plot_path = request.args.get('plot_path', type=str)

    return render_template(
        'forecast_results.html', 
        forecast_fin=forecast_fin,
        forecast_ind=forecast_ind,
        forecast_ns=forecast_ns,
        forecast_total=forecast_total,
        plot_path=plot_path
    )


def save_forecast_plot(df, forecast_fin, forecast_start, forecast_end):
    """Save the forecast plot to a file and return the path."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Revenue FIN'], label='Historical FIN', color='blue')
    plt.plot(pd.date_range(forecast_start, forecast_end, freq='MS'), forecast_fin, label='Forecast FIN', color='red')
    plt.title('Forecast vs Historical Revenue FIN')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'forecast_plot.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    return plot_path



@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

