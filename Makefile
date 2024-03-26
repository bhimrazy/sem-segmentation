# Default target: Help Message
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "dataset    : Download data"

# Download data
dataset:
	python scripts/download_datasets.py

# Run the main script
run:
	python main.py
