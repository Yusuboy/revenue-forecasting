# Prerequisites

-   Python installed. If needed, [download](https://www.python.org/downloads/) and install. In installation, ensure that you select adding python to the windows path environment variable.
-   Verify that python and pip are available:
    -   Open windows command prompt.
    -   Run command *python -V.* You should get a message about your python version.
    -   Run command *pip -V*. You should get a message about your pip version and location.
    -   In case of python and/or pip not available, check your python installation and path in Windows environment variables.
-   Visua Studio Build Tools installed. If needed, download and install from https://visualstudio.microsoft.com/visual-cpp-build-tools/

![img1](https://github.com/Yusuboy/revenue-forecasting/blob/master/inst_win_img1.jpg)

# Get code

-   Open [GitHub repository](https://github.com/Yusuboy/revenue-forecasting)
-   Click *Code* → *Download ZIP*
-   Use the browser download page or Windows Explorer to find the downloaded file

![img2](https://github.com/Yusuboy/revenue-forecasting/blob/master/inst_win_img2.jpg)

# Install code

-   Extract the downloaded zip file to a directory where you want to locate it, e.g. *Program Files*.
-   The code will be available in a sub-directory *revenue-forecasting-master.* You are free to define and modify the directory names as you wish.

![img4](https://github.com/Yusuboy/revenue-forecasting/blob/master/inst_win_img5.jpg)

Note: If you want to use virtual environment, jump to section "Using virtual environment" below. Otherwise continue from next section.

# Install dependencies

-   Browse to the installation directory (*revenue-forecasting-master*) with windows command prompt
-   Run command *pip install -r requirements.txt*
-   Validate that there are no error messages in the output

![img5](https://github.com/Yusuboy/revenue-forecasting/blob/master/inst_win_img6.jpg)


# Run code

-   Change to the *src* sub-directory: *cd src*
-   Run command *flask --app app run*
-   If successful, you should see a message *Running on http://127.0.0.1:5000*
-   Open a web browser in address [*http://127.0.0.1:5000*](http://127.0.0.1:5000)

![img6](https://github.com/Yusuboy/revenue-forecasting/blob/master/inst_win_img7.jpg)

# Use the application

-   See the instructions in the homepage and the separate user instructions document.

# Using virtual environment

Benefits for using virtual environment:
Keeps your global Python installation clean and prevents dependency conflicts.

## Download code and define project folder:

(In case you did this already you can skip this phase)

1. Create a new directory for your project in a location with a short path (e.g., C:\Projects\forecastProject)
2. Download zip file for the code and extract it to your project folder on your computer

## Create virtual environment:

1. Open a terminal or command prompt and navigate to your project directory: cd C:\Projects\forecastProject
2. Navigate to master folder: cd C:\Projects\forecastProject\revenue-forecasting-master
3. Create a virtual environment using venv (recommended): This creates a folder named <your_virtual_env> inside your project directory
```   
python -m venv <your_virtual_env>
```

## Install requirements:

1. Activate the virtual environment (from your project folder where folder named <your_virtual_env> is located):

```
<your_virtual_env>\Scripts\activate
```

2. Install dependencies (navigate to folder where requirements.txt is located). This will take few minutes most likely:

```
pip install -r requirements.txt
```

## Running app with virtual environment and installed requirements:

1. Navigate to your project folder where folder named <your_virtual_env> is located
2. Activate your virtual environment
```
   <your_virtual_env>\Scripts\activate
```
3. Navigate to your project folder where app.py is located (src folder) and run following command

```
flask --app app run
```
4. Open http-link that you see in command prompt (Edge or Chrome)
