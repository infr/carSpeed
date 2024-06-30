import kagglehub

# Download latest version
path = kagglehub.model_download("tensorflow/ssd-mobilenet-v2/tensorFlow2/fpnlite-320x320")

print("Path to model files:", path)