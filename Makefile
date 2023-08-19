# Variables
PYTHON = python
MAIN_SCRIPT = main.py
CONFIG_FILE = configs/config.yaml

# Default target: Help Message
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  dataset    : Download data"

# Download data
dataset:
	$(PYTHON) scripts/download_datasets.py

# Run the main script
run:
	$(PYTHON) $(MAIN_SCRIPT) -c $(CONFIG_FILE)
