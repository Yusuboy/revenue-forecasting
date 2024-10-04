from datetime import date
from preprosessing.prepros3 import main
def run_data_processing(input_file, output_file, report_file):
    """Run data processing on the specified files."""
    main(input_file, output_file, report_file)

if __name__ == "__main__":
    input_file_path = 'files/2024-09-20 Kopio Liikevaihto_data (2).csv'  # Path to your new input file

    # Get today's date and format it
    today = date.today().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD

    # Create output and report file names with the current date
    output_file_path = f'preprosessed_{today}.csv'  # E.g., preprosessed_2024-09-25.csv
    report_file_path = f'report_file_{today}.csv'  # E.g., report_file_2024-09-25.csv

    # Run the data processing
    run_data_processing(input_file_path, output_file_path, report_file_path)
