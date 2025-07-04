import torch
import torchvision
from PIL import Image
import numpy as np
import os
from io import BytesIO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import torchvision.transforms as T


def plot_detections(image_path, boxes, labels, scores, threshold=0.5, num_classes=3):
    image = np.array(Image.open(image_path).convert("RGB"))
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    ax = plt.gca()

    # Define a color for each label (assuming labels are from 1 to num_classes)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            color = colors[label]
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'{label}: {score:.2f}', bbox=dict(facecolor=color, alpha=0.5), fontsize=10,
                    color='white')

    plt.axis('off')
    # Save the plot to a BytesIO object and return it
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)


# Define a new box predictor with a hidden layer
class CustomFastRCNNPredictor(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_neurons):
        super(CustomFastRCNNPredictor, self).__init__()
        self.hidden_neurons = hidden_neurons  # Store hidden neurons count
        self.cls_score = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons, num_classes)
        )
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def create_model(num_classes, hidden_neurons):
    # Load the pretrained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = CustomFastRCNNPredictor(in_features, num_classes, hidden_neurons)
    return model


# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Load the model
def load_model(path, num_classes=3):
    # Load the pretrained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    # Note: hidden_neurons will be inferred from the loaded state_dict
    model.roi_heads.box_predictor = CustomFastRCNNPredictor(in_features, num_classes,
                                                            100)  # Placeholder for hidden_neurons
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Get hidden neurons count from the loaded model
def get_hidden_neurons(model):
    return model.roi_heads.box_predictor.hidden_neurons


# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)  # Add batch dimension


def Model_WAM_RCNN_train(Images_path, img):
    num_classes = 3
    hidden_neurons = 100
    out_dir = os.listdir(Images_path)
    Result = []

    model = create_model(num_classes, hidden_neurons)
    model.eval()  # Set model to evaluation mode

    for i in range(len(img)):
        print(i, len(img))
        in_dir = Images_path + '/' + out_dir[i]
        imgs = img[i]
        image_path = in_dir

        # Load and preprocess the image
        image = load_image(image_path)

        # Perform object detection
        with torch.no_grad():
            prediction = model(image)

        # Extracting bounding boxes, labels, and scores
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        # Visualize the detections and get the plotted image
        plotted_image = plot_detections(image_path, boxes, labels, scores)
        results = imgs
        Result.append(results)

    return model, Result


def Model_WAM_RCNN_Test(Model, Images, img, sol=None):
    if sol is None:
        sol = [5]
    path = Images
    out_dir = os.listdir(path)
    Result = []

    model = Model

    for i in range(len(img)):
        print(i, len(img))
        in_dir = path + '/' + out_dir[i]
        imgs = img[i]
        image_path = in_dir

        # Load and preprocess the image
        image = load_image(image_path)

        # Perform object detection
        with torch.no_grad():
            prediction = model(image)

        # Extracting bounding boxes, labels, and scores
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        # Visualize the detections and get the plotted image
        plotted_image = plot_detections(image_path, boxes, labels, scores)
        results = imgs
        Result.append(results)
    return Result

