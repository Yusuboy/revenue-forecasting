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

# Instructions

1.  Pre-requisites: python installed. If needed, check the [download page](https://www.python.org/downloads/).
2.  Installation, two options: use Git or Export GitHub repository as a ZIP file:
    1.  Git
-   Clone the repository to your machine and navigate to the root directory.

```
git clone git@github.com:Yusuboy/revenue-forecasting.git
```

    2. Export ZIP
    -   Create a local directory for the code in your computer
        -   In GitHub, go to the Repository and open the repository's main page.
        -   Click on "Code": You’ll see a green button labelled "Code" near the top.
        -   Download ZIP: Click "Download ZIP" to download the entire repository as a ZIP file, which you can then extract to your desktop.
        -   Extract the ZIP file to your computer to the directory you created
2.  Set up a virtual environment and install the requirements in the root directory of the project

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

1.  Run the following command to start the app

```
$ flask --app src.app run
```
