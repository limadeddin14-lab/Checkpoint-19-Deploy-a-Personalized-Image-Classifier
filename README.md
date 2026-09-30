
# Checkpoint 19 - Personalized Image Classifier

## Project
A Streamlit web application that serves a custom-trained PyTorch CNN for handwritten digit classification.

## Files
- `train_model.py` - trains the CNN and creates `model.pt`
- `streamlit_app.py` - Streamlit interface
- `requirements.txt` - dependencies
- `model.pt` - trained model (created after running the training script)

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## Test image
Use a clear image containing one handwritten digit (0-9), preferably centered on a plain background.

## Submission explanation
This project deploys a custom-trained PyTorch convolutional neural network (CNN) for handwritten digit image classification. Users can upload a JPG, JPEG, or PNG image through a Streamlit interface and receive a predicted digit with a confidence score and probability chart. The application preprocesses the uploaded image before sending it to the trained model. Future improvements could include multi-image batch prediction, a download option for prediction results, and support for more complex image classes.
