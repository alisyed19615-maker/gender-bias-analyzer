# Use an official Python 3.9 image
FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Copy and install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the app's code
COPY . .

# NEW: Create a cache directory inside our app and make sure it's writable
RUN mkdir -p /code/.cache && chmod -R 777 /code/.cache

# Tell the container what command to run
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]