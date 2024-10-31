from flask import Flask
from os import getenv
from dotenv import load_dotenv  # Import the load_dotenv function
import os
from routes import routes_bp

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Ensure the SECRET_KEY is set in your environment
app.secret_key = getenv("SECRET_KEY")  # This will get the SECRET_KEY from .env

# Registering the main routes
app.register_blueprint(routes_bp)

if __name__ == "__main__":
    app.run(debug=True)
