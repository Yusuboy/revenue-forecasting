from flask import Flask
from os import getenv
from dotenv import load_dotenv  # Import the load_dotenv function
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Ensure the SECRET_KEY is set in your environment
app.secret_key = getenv("SECRET_KEY")  # This will get the SECRET_KEY from .env

# Registering the main routes
import routes
