# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s ~/.local/bin/poetry /usr/local/bin/poetry

# Copy the project files into the container
COPY . /app

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Expose the port the app runs on
EXPOSE 7777

# Set the default command to run the application
CMD ["python", "/app/robomind/app.py"]