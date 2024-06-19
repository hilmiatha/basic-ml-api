FROM continuumio/anaconda3
LABEL UNP https://unp.education

# Copy the application code
COPY ./flask_demo /usr/local/python/

# Expose the port the app runs on
EXPOSE 5000

# Set the working directory
WORKDIR /usr/local/python/

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies with trusted hosts to avoid SSL issues
RUN pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Set the command to run the application
CMD ["python", "random_forest_API.py"]
