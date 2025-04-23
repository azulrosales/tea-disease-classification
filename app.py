import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

st.title("🍃 Tea Disease Classification")

# Upload class labels
labels_file = st.file_uploader("Class labels (.txt)", type=["txt"])
if labels_file:
    labels = [line.strip() for line in labels_file.readlines()]
    # st.success("Labels loaded!")
else:
    labels = None

# Upload model weights
model_file = st.file_uploader("EfficientNet weights (.pth)", type=["pth"])
if model_file and labels:
    try:
        model = EfficientNet.from_name('efficientnet-b0')
        model._fc = torch.nn.Linear(model._fc.in_features, len(labels))
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        # st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None
else:
    model = None

# Upload image
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if image_file and model and labels:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_classes = probs.topk(3)

    st.subheader("Predictions")
    for i in range(top_probs.size(0)):
        class_idx = top_classes[i].item()
        class_name = labels[class_idx] if class_idx < len(labels) else f"Unknown ({class_idx})"
        st.write(f"{class_name}: {round(top_probs[i].item() * 100, 2)}%")
