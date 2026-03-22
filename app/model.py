import torch
from PIL import Image

from app.utils import preprocess_image
from models.resnet import ResNet18


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet18().to(device)
model.load_state_dict(
    torch.load("checkpoints/cifar10_resnet18.pth", map_location=device)
)
model.eval()


def predict_image(image: Image.Image):
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probs, 3)

    top3 = {}
    for prob, idx in zip(top3_probs, top3_indices):
        top3[CLASS_NAMES[idx.item()]] = float(prob.item())

    top1 = CLASS_NAMES[top3_indices[0].item()]
    confidence = float(top3_probs[0].item())

    return {
        "top1": top1,
        "confidence": confidence,
        "top3": top3
    }
