# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /challenge

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Specify the command to run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]