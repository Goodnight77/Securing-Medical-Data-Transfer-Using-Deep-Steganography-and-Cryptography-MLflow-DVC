
# Securing Medical Data Transfer Using Deep Steganography and Cryptography - MLflow & DVC

## Project Overview

This project implements a secure system for medical data transfer using deep steganography and cryptography, integrated with MLflow for experiment tracking and DVC for data version control. The system aims to safeguard sensitive medical data during transmission by embedding it within cover images, encrypting the embedded data, and utilizing cryptographic keys for secure access. 

## Features

* **Deep Steganography**:  Leverages a deep learning model to embed medical data within cover images in a way that is difficult to detect.
* **Cryptography**:  Employs robust encryption algorithms to protect the embedded data from unauthorized access.
* **MLflow Integration**: Tracks experiments, including model parameters, metrics, and artifacts, allowing for efficient model comparison and analysis.
* **DVC Integration**: Manages data versions, ensuring reproducibility and traceability of data used in the project.
* **User Authentication**: Provides secure user authentication for accessing and managing medical data.

## Installation

This project utilizes a virtual environment and requires the installation of specified dependencies. To set up the project:

1. **Create a virtual environment:**

   ```bash
   python -m venv venv 
   ```

2. **Activate the virtual environment:**

   ```bash
   source venv/bin/activate 
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Ingestion:** Prepare your medical data and cover images. The project provides a `data_ingestion` script for managing data.

2. **Model Training:** Train the deep learning steganography model. You can utilize the `model_training` script for this purpose. MLflow will log training progress and metrics.

3. **Model Evaluation:** Evaluate the trained model's performance using the `model_evaluation_with_mlflow` script. 

4. **Model Split:** Split the trained model into hiding and revealing components using the `model_split` script. 

5. **Data Hiding:** Use the hiding component of the split model to embed your medical data into a cover image.

6. **Data Revealing:** Utilize the revealing component of the split model to recover the hidden medical data from the cover image.

## License

This project is licensed under the Apache License 2.0.

## Contact Information

For any questions or inquiries, please contact:

* **Email:** [mohammedarbi.nsibi@supcom.tn]

