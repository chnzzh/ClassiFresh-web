# ClassiFresh-web

A deep learning-based food freshness classification web application that can assess the freshness of uploaded food images.

## Project Overview

ClassiFresh-web uses a pre-trained ResNet50 deep learning model to classify food images into three freshness levels: fresh, semi-fresh, and spoiled. The application provides a simple web interface where users can upload food images and receive instant freshness assessment results.

## Features

- Clean and easy-to-use web interface
- Support for image upload and real-time analysis
- ResNet50-based image classification
- Display of classification results with confidence percentages

## Installation and Setup

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/chnzzh/ClassiFresh-web.git
   cd ClassiFresh-web
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the model file is placed in the correct location:
   - Place the trained model file `best_model_resnet50.pth` in the models directory

## Usage Instructions

1. Start the application:

   ```bash
   python app.py
   ```

2. Access in your browser: [http://localhost:5000](http://localhost:5000)


3. Upload image and get the freshness classification results

## Project Structure

```bash
ClassiFresh-web/
├── app.py             # Flask application main file
├── config.py          # Application configuration
├── models/            # Directory for pre-trained models
├── static/            # Static files directory
│   └── uploads/       # User upload images
└── templates/         # HTML templates
    └── index.html     # Main page template
```