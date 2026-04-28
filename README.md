# Garbage Classifier - Streamlit App

A web application for classifying garbage items using a pre-trained Keras neural network model.

## Features

- Upload images (JPG, PNG, GIF, BMP) to classify garbage items
- Real-time predictions with confidence scores
- Displays probabilities for all garbage classes
- User-friendly web interface built with Streamlit

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Setup

1. **Activate the virtual environment** (if you created one):
   ```bash
   .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. Click "Browse files" or drag and drop an image
2. Select an image file (JPG, PNG, GIF, or BMP)
3. The app will display:
   - The uploaded image
   - Predicted garbage class
   - Confidence score
   - Probabilities for all classes

## Model Information

- **Model**: Garbage_Classifier_Final_95.h5
- **Accuracy**: 95%
- **Location**: `model/` directory

## Customization

To modify the class labels, edit the `class_labels` list in `app.py`:

```python
class_labels = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash"
]
```

## Notes

- The app uses image preprocessing to resize images to 224x224 pixels
- Images are normalized to the [0, 1] range
- Model predictions are cached for better performance

## Troubleshooting

- **Model not found**: Ensure `model/Garbage_Classifier_Final_95.h5` exists in the project directory
- **Slow predictions**: The model is cached after the first load for faster subsequent predictions
- **Memory issues**: Close other applications if you encounter memory errors
