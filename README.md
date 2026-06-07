# Smart City Adaptive Energy Dispatch

A smart city energy dispatch and extreme-event forecasting system built with Flask, machine learning, and optimization scripts.

## Project Description

This repository contains the core application and supporting scripts for adaptive energy dispatch in a smart city environment. It includes a Flask-based dashboard, pre-trained model artifacts, and analysis scripts used for forecasting, optimization, and extreme-event analytics.

## Repository Structure

- `backend/`
  - `app.py` - Flask application entry point
  - `requirements.txt` - Python dependencies
  - `static/` - front-end JS and CSS
  - `templates/` - HTML templates
  - `models/` - trained model files and preprocessing artifacts
- `scripts/` - data preparation, forecasting, training, optimization, and analysis scripts

## Notes

- The repository is intentionally clean and excludes raw data and large datasets.
- Data folders from the original project are not included here to keep the repository lightweight.
- If you need to reproduce results, restore the data from the original dataset sources or use the scripts to regenerate processed data.

## Quick Start

1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r backend\requirements.txt
   ```
3. Run the Flask app:
   ```powershell
   cd backend
   python app.py
   ```
4. Open the dashboard in your browser at `http://127.0.0.1:5000`.

## Recommended GitHub Repository Name

`smart-city-adaptive-energy-dispatch`

## Recommended Description

`Adaptive smart city energy dispatch and extreme event forecasting system using Flask, machine learning models, and optimization scripts.`
