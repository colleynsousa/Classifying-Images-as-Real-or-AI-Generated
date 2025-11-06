from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError

def safe_loader(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (OSError, UnidentifiedImageError):
        print(f"⚠️ Skipping bad image: {path}")
        return None

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = safe_loader(path)
        if sample is None:
            # Skip this image by moving to next
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target