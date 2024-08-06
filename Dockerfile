# syntax=docker/dockerfile:1.2
FROM python:3.11.9
# put you docker configuration here

# Set the working directory in the container
WORKDIR /challenge

# Copiar el archivo app.py y otros archivos necesarios al contenedor
COPY challenge/api.py /challenge/
COPY requirements.txt /challenge/

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run the application
CMD ["python", "api.py"]