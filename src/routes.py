import os
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    flash,
    send_file,
    url_for,
)
from datetime import datetime, timedelta
import pandas as pd
from preprosessor import Preprocessor
import matplotlib.pyplot as plt

# from app import app
from revenue_forecast_multicaptive_normalized import (
    RevenueForecastMulticaptiveNormalized,
)
from revenue_forecast_runrate import RevenueForecastRunrate
import logging
import calendar
import csv
from flask import make_response
import io
from dateutil.relativedelta import relativedelta
from forecast_utils import ForecastUtils
import os
import glob

logging.basicConfig(level=logging.DEBUG)
routes_bp = Blueprint("routes_bp", __name__)

# Ensure the directory for uploads exists
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# @app.route('/download_forecast', methods=['GET'])


@routes_bp.route("/download_forecast")
def download_forecast():
    try:
        csv_filename = os.path.join(UPLOAD_FOLDER, "forecast_results.csv")
        return send_file(
            csv_filename, as_attachment=True, attachment_filename="forecast_results.csv"
        )
    except Exception as e:
        return f"Error generating CSV: {str(e)}", 500


UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# @app.route('/')


@routes_bp.route("/")
def home():
    logging.debug("*** Showing home ***")
    return render_template("home.html")


# @app.route('/preprocess')


@routes_bp.route("/save_changes", methods=["POST"])
def save_changes():
    try:
        # Parse the JSON data from the request
        data = request.get_json()

        # Generate a unique filename with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        filename = f"critical_values_{timestamp}.csv"
        save_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the data to a CSV file
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header row
            writer.writerow(["Cell ID", "Value"])
            # Write each modified cell
            for cell_id, value in data.items():
                writer.writerow([cell_id, value])

        return f"Changes saved successfully! File: {filename}", 200
    except Exception as e:
        logging.error(f"Error saving critical values: {e}")
        return str(e), 500


@routes_bp.route("/load_saved_values", methods=["GET"])
def load_saved_values():
    try:
        # Find the most recent CSV file with the saved values
        csv_files = glob.glob(os.path.join(UPLOAD_FOLDER, "critical_values_*.csv"))
        if not csv_files:
            return {"message": "No saved values found"}, 404

        # Sort files by modification time (newest first)
        csv_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = csv_files[0]

        # Read the CSV and map the data into a dictionary
        saved_values = {}
        with open(latest_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                cell_id, value = row
                saved_values[cell_id] = value

        return saved_values, 200
    except Exception as e:
        logging.error(f"Error loading saved values: {e}")
        return {"message": str(e)}, 500


@routes_bp.route("/preprocess")
def preprocess():
    return render_template("preprocess.html")


def save_file(file):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return filepath


def process_files(bil_path, calendar_path):
    current_date = datetime.now().strftime("%Y%m%d")
    output_datafile_path = os.path.join(f"data.csv")
    # processed_bil_path = os.path.join(UPLOAD_FOLDER, f'processed_bil_{current_date}.csv')
    # processed_tlx_path = os.path.join(UPLOAD_FOLDER, f'processed_tlx_{current_date}.csv')
    preprocessor = Preprocessor()
    preprocessor.prerpocess(bil_path, calendar_path, output_datafile_path)
    return output_datafile_path  # , processed_bil_path, processed_tlx_path#


# @app.route('/process', methods=['POST'])


@routes_bp.route("/process", methods=["POST"])
def process():
    if "bil_file" not in request.files or "calendar_file" not in request.files:
        flash("Both files are required!", "danger")
        return redirect(url_for("routes_bp.preprocess"))

    try:
        bil_file = request.files["bil_file"]
        calendar_file = request.files["calendar_file"]

        bil_path = save_file(bil_file)
        calendar_path = save_file(calendar_file)
        process_files(bil_path, calendar_path)

        flash(
            "Preprocessing completed successfully. See file data.csv in the exection directory.",
            "success",
        )
    except Exception as e:
        flash(f"Error during preprocessing: {str(e)}", "danger")

    return redirect(url_for("routes_bp.preprocess"))


# @app.route('/forecast', methods=['GET', 'POST'])


@routes_bp.route("/forecast", methods=["GET", "POST"])
def forecast():

    print("forecast")

    if request.method == "POST" or request.method == "GET":

        try:
            # Initializing models
            print("*** Initializing models ***")

            multiplicative_model = RevenueForecastMulticaptiveNormalized()
            run_rate_model = RevenueForecastRunrate(use_trend=False)

            # Training models
            train_start_date = "2021-10-01"
            first_day_of_current_month = datetime.now().replace(day=1)
            last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
            training_end_date = last_day_of_previous_month.strftime("%Y-%m-%d")
            print("*** Training models " + train_start_date + "-" + training_end_date)
            multiplicative_model.train_model(train_start_date, training_end_date)
            run_rate_model.train_model(train_start_date, training_end_date)

            # Forecasting
            forecast_start_date = first_day_of_current_month.strftime("%Y-%m-%d")
            first_day_of_14th_month = first_day_of_current_month + relativedelta(
                months=+13
            )
            last_day_of_13th_month = first_day_of_14th_month - timedelta(days=1)
            forecast_end_date = last_day_of_13th_month.strftime("%Y-%m-%d")
            print(
                "*** Forecasting models" + forecast_start_date + "-" + forecast_end_date
            )
            forecast_fin_mp, forecast_ind_mp, forecast_ns_mp, forecast_total_mp = (
                multiplicative_model.forecast(forecast_start_date, forecast_end_date)
            )
            forecast_fin_rr, forecast_ind_rr, forecast_ns_rr, forecast_total_rr = (
                run_rate_model.forecast(forecast_start_date, forecast_end_date)
            )

            # Creating monthly column headers
            now = datetime.now()
            month_headers = [
                calendar.month_abbr[(now.month - 3 + i) % 12 or 12] for i in range(16)
            ]

            # print(month_headers)

            # Convert forecasts to list and round to int to allow easy handling in the html template
            forecast_fin_mp = list(map(int, forecast_fin_mp))
            forecast_ind_mp = list(map(int, forecast_ind_mp))
            forecast_ns_mp = list(map(int, forecast_ns_mp))
            forecast_fin_rr = list(map(int, forecast_fin_rr))
            forecast_ind_rr = list(map(int, forecast_ind_rr))
            forecast_ns_rr = list(map(int, forecast_ns_rr))

            print("forecast fin mp")
            print(forecast_fin_mp)

            # Get the actual revenues for the last 3 months in the training period
            current_year = datetime.now().year
            current_month = datetime.now().month
            df = ForecastUtils.load_data()
            df_filtered = df[
                (df["Year"] < current_year)
                | ((df["Year"] == current_year) & (df["Month"] < current_month))
            ]
            last_three_months = df_filtered.tail(3)
            actual_fin = last_three_months["Revenue FIN"].tolist()
            actual_ind = last_three_months["Revenue IND"].tolist()
            actual_ns = last_three_months["Revenue NS"].tolist()
            actual_fin = list(map(int, actual_fin))
            actual_ind = list(map(int, actual_ind))
            actual_ns = list(map(int, actual_ns))

            # Rendering html template
            print("*** Rendering forecast results (forecast_results.html) ***")
            saved_values = load_saved_values()
            return render_template(
                "forecast_results.html",
                saved_values=saved_values,
                forecast_fin_mp=forecast_fin_mp,
                forecast_ind_mp=forecast_ind_mp,
                forecast_ns_mp=forecast_ns_mp,
                forecast_fin_rr=forecast_fin_rr,
                forecast_ind_rr=forecast_ind_rr,
                forecast_ns_rr=forecast_ns_rr,
                actual_fin=actual_fin,
                actual_ind=actual_ind,
                actual_ns=actual_ns,
                month_headers=month_headers,
            )

        except Exception as e:
            print(f"Exception occurred: {e}")
            flash(f"Error processing request: {str(e)}", "danger")
            return redirect(url_for("routes_bp.home"))  # Redirect back to home on error

    return redirect(url_for("routes_bp.home"))  # Redirect to home if GET request


# @app.route('/download/<filename>', methods=['GET'])


@routes_bp.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)
