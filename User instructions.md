# 0. Prerequisites

-   Revenue forecasting application installed. See [instructions for windows installation in GitHub](https://github.com/Yusuboy/revenue-forecasting/blob/master/Installation_windows.md).
-   Revenue data exported from finance system. For support, please contact Joonas Rosi.
-   Revenue data needs to be in CSV format. In case it is in Excel format, save a version in CSV format.
-   The timestamp format is strict: dd/mm/yyyy HH:MM.
    -   When using Excel for conversion, extra cautiousness with timestamps is needed.
-   Up-to-date *calendar.csv* file available for the application. Calendar file includes number of monthly working days per location and historical rate raise percentages. You may find this file in src folder of this application (remember to update it as time goes on).

# 1. Upload Data

-   Copy data to a directory where you want to store it. The directory needs to be available for the application through web UI.

# 2. Start Application

-   Navigate to src sub-directory: cd <application_root>\src. *Application root is the directory where you have installed the application.*
-   Run command *flask --app app run.*
-   If successful, you should see a message *Running on* [*http://127.0.0.1:5000*](http://127.0.0.1:5000)*.*
-   Open a web browser in address [*http://127.0.0.1:5000*](http://127.0.0.1:5000). You should see the home page of the app with three tabs: *Home, Preprocessing and Forecast*.

Note: If you are using virtual environment remember to activate it first (see installation_windows.md)

# 3. Preprocess Data

-   Browse to the *Preprocess* tab.
-   Select the revenue data file with the file picker.
-   Select the calendar file with the file picker.
-   Click the *Process Data* button. Processing usually takes 2 to 10 minutes, depending on data size.
-   The preprocessed datafile *data.csv* is available in the directory where the application was run.
-   **Important:** In case there are some incomplete data for the ongoing month in the revenue file, the result data for that month needs to be removed from *data.csv* manually.
    -   For example, if the export is done in early January 2025 and there is already some January 2025 revenue data in export, January 2025 cells in columns Revenue\* need to be removed.
-    *data.csv* in src folder is used to train models when you select *Forecast* tab. It is adviced to copy this file to a separate folder (and rename it for example *data_0125.csv*) in case you want (or need) to inspect predictions from some specific month in more detail.

Note: When you click *Process data* it will overwrite any existing *data.csv* in your src folder.

# 4. Forecast

## Time and Material forecast
-   Browse to the *Forecast* tab. This will use *data.csv* to train models (Multiplicative and Run rate) and produces 12 month forecast per each model.
-   On top you can see 12 month forecast per each model (Multiplicative and Run rate). Green cell indicates that this value will be used when calculating total forecast per month per location (at the bottom).

Note: If forecasts provided by both models differ significantly from each others it may be wise to consider possible explanations for that.

## T&M Forecast adjustments
-   You can add adjustments to forecast per month per location based on your better knowledge. Red indicates that a cell has been activated in order to manipulate its value.
-   You can add additional revenue components from bottom by clicking *Add Revenue Row*. This will create a new row that will be used in calculating total revenue forecast.
-   If you want to use same adjustments in the next run you can save modified adjustments and added revenue components by clicking *Save Changes* button.

## Total Forecast

-   *Calculate totals* gives you a final forecast that takes in consideration all adjustments and added revenue components.
-   If you make adjustments to your forceast remenber to always click the *Calculate Totals* button. Totals are not updated automatically.
-   The forecast can be exported in CSV format to be used further in Excel (see your usual download folder).

Note: Models provide only a baseline for total forecast based on revenue data exported from finance system. Person executing this forecast may add domain knolwdge outside that data through adjustments mentioned above.
