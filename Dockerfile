# Use an official Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install the libraries
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your entire project into the container
COPY . .

# Tell the container what command to run when it starts
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]