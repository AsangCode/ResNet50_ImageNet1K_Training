---
title: ResNet50 ImageNet Classifier
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
license: mit
---

# ResNet50 ImageNet Classifier

This is a Hugging Face Spaces app that uses a ResNet50 model trained on ImageNet-1K to classify images.

## Model

The model is a ResNet50 architecture trained on the ImageNet-1K dataset with 1000 classes. The model achieves state-of-the-art performance for image classification tasks.

## Usage

1. Upload an image or select one of the example images
2. The model will predict the top 5 classes for the image with confidence scores

## Examples

The app includes example images that you can use to test the model.

## Local Development

To run this app locally:

```bash
pip install -r requirements.txt
python app.py
```

## Files

- `app.py`: The Gradio web application
- `model.py`: ResNet50 model definition
- `model_49.pth`: Trained model weights
- `imagenet_classes.json`: Mapping of class indices to human-readable labels (auto-generated)
- `requirements.txt`: Dependencies needed to run the app
- Example images for testing

## Deployment to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select Gradio as the SDK
3. Clone the Space repository to your local machine
4. Copy all files from this directory to the cloned repository
5. Push the changes to Hugging Face
6. The Space will automatically build and deploy your app

## License

This project is open-source and available for educational and research purposes.
