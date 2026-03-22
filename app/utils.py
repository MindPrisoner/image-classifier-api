from PIL import Image
import torchvision.transforms as transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])


def preprocess_image(image: Image.Image):
    transform = get_transform()
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    return image
