# 0. Prerequisites

-   Revenue forecasting application installed. See [instructions for windows installation in GitHub](https://github.com/Yusuboy/revenue-forecasting/blob/master/Installation_windows.md).
-   Revenue data exported from finance system. For support, please contact Joonas Rosi.
-   Revenue data needs to be in CSV format. In case it is in Excel format, save a version in CSV format.
-   The timestamp format is strict: dd/mm/yyyy HH:MM.
    -   When using Excel for conversion, extra cautiousness with timestamps is needed.
-   Up-to date calendar file available for the application. Calendar file includes number of monthly working days per location and historical rate raise percentages.

# 1. Upload Data

-   Copy data to a directory where you want to store it. The directory needs to be available for the application through web UI.

# 2. Start Application

-   Change to the src sub-directory: *cd \<application root\>/src. Application root is the directory where you have installed the application.*
-   Run command *flask --app app run.*
-   If successful, you should see a message *Running on* [*http://127.0.0.1:5000*](http://127.0.0.1:5000)*.*
-   Open a web browser in address [*http://127.0.0.1:5000*](http://127.0.0.1:5000). You should see the home page of the app with three tabs: *Home, Preprocessing and Forecast*.

Note: If you are using virtual environment remember to activate it first (see installation_windows.md)

# 3. Preprocess Data

-   Browse to the *Preprocess* tab.
-   Select the revenue data file with the file picker.
-   Select the calendar file with the file picker.
-   Click the *Process Data* button. This may take a while so be prepared to wait for few minutes.
-   The preprocessed datafile *data.csv* is available in the directory where the application was run.
-   In case there are some incomplete data for the ongoing month in the revenue file, the result data for that month needs to be removed from *data.csv* manually.
    -   For example, if the export is done in early January 2025 and there is already some January 2025 revenue data in export, January 2025 cells in columns Revenue\* need to be removed.
-    *data.csv* in src folder is used to train models when you select *Forecast* tab. It is adviced to copy this file to a separate folder (and rename it for example *data_0125.csv*) in case you want (or need) to inspect predictions from some specific month in more detail.


# 4. Forecast

-   Browse to the *Forecast* tab.
-   Choose the used model per month from precalculated values.
-   Add correction terms.
-   Add additional revenue components.
-   Always click the *Calculate Totals* button to calculate effect of adjustment. The totals are not updated automatically.
-   Remember to save the inputs (*Save Changes* button) if you want to make the adjustments available in the next run.
-   The forecast can be exported in CSV format to be used further in Excel.
