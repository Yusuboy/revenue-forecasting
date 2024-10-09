import os
from os import getenv
from app import app
from flask import Blueprint, render_template, request, redirect, flash, send_file
from datetime import datetime
import pandas as pd
from preprosessing.preprosessor import main  # Import the main function from preprosessor

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

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)
