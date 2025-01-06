# Revenue Forcasting

## Project Overview

This project is focused on enhancing the revenue forecasting capabilities for a real-world ICT company that is part of a major global ICT corporation

### Company Details

-   **Parent Company:** Global ICT Leader
-   **Local Workforce:** 300+ employees
-   **Global Workforce:** 300-400 employees

### Project Scope

-   **Revenue Forecasting:** Rolling forecast for a 12-month period
-   **Focus Area:** Accuracy of the forecast for the upcoming month
-   **Integration:** Part of the global financial reporting process

The goal of this project is to refine forecasting methods to better predict both short and long term revenues as well as reduce amount of manual work needed for forecasting.

# Instructions (Python)

1.  Pre-requisites: python installed. If needed, check the [download page](https://www.python.org/downloads/).
2.  Installation, two options: use Git or Export GitHub repository as a ZIP file:
   
   **Git**
-   Clone the repository to your machine and navigate to the root directory.

```
git clone git@github.com:Yusuboy/revenue-forecasting.git
```
   **Export zip**
-   Create a local directory for the code in your computer
-   In GitHub, go to the Repository and open the repository's main page.
-   Click on "Code": You’ll see a green button labelled "Code" near the top.
-   Download ZIP: Click "Download ZIP" to download the entire repository as a ZIP file, which you can then extract to your desktop.
-   Extract the ZIP file to your computer to the directory you created
3.  Set up a virtual environment and install the requirements in the root directory of the project

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

4.  Run the following command to start the app

```
$ flask --app src.app run
```
# Instructions (Windows)

Open installation_windows.md file from root folder and follow instructions


## If you want to use virtual environment follow instructions below:

1. Create a new directory for your project in a location with a short path (e.g., C:\Projects\forecastProject)
2. Download zip file for the code and extract it to your project file on your computer

### Instructions to create virtual environment:

1. Open a terminal or command prompt and navigate to your new project directory: cd C:\Projects\forecastProject
2. Create a virtual environment using venv (recommended): This creates a folder named .venv inside your project directory
```   
python -m venv .venv
```

### Instructions to install requirements:

1. Activate the virtual environment (from your project folder where .venv is located):

```
venv\Scripts\activate
```

2. Install dependencies (navigate to folder where requirements.txt is located):

```
pip install -r requirements.txt
```

Benefits:
Keeps your global Python installation clean and prevents dependency conflicts.

### Running app with venv and installed requirements:

1. Navigate to your project folder where .venv is 
2. Activate your virtual environment
```
   .venv\Scripts\activate
```
3. Navigate to your project folder where app.py is located (src folder) and run following command

```
flask --app app run
```
4. Open http-link that you see in command prompt
