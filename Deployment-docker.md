create a Dockerfile to define the environment and dependencies for our model deployment.

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV PORT=8501

# Expose the port for the Streamlit app
EXPOSE $PORT

# Command to run the Streamlit app when the container starts
CMD ["streamlit", "run", "--server.port=$PORT", "app.py"]

<!-- build the Docker image using the following command: -->

docker build -t house_prices_app .

<!-- Once the Docker image is built successfully, you can run the Docker container: -->

docker run -p 8501:8501 house_prices_app

<!-- Streamlit App Deployment:
To deploy the Streamlit app on a cloud service, let's use Heroku as an example:

Create a Heroku Account: If you don't have one, sign up for a free account on Heroku.
Install Heroku CLI: Install the Heroku Command Line Interface (CLI) by following the instructions on the Heroku Dev Center.
Login to Heroku: Open your terminal and log in to your Heroku account using the command heroku login.
Initialize Git Repository: If you haven't already, initialize a Git repository in your project folder.
Create a Heroku App: Run the following command to create a new Heroku app: -->

<!-- heroku create <app-name>
Replace <app-name> with your desired app name. This will also add a new remote named heroku.
Deploy App: Deploy your app to Heroku using Git: -->
git push heroku master

<!-- Open App: Once the deployment is successful, you can open your app in the browser using: -->
heroku open
