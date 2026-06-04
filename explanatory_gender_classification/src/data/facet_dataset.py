"""FACET benchmark dataset loader for cross-dataset evaluation."""
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FACETDataset(Dataset):
    """
    Dataset for loading FACET dataset.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        label = row['encoded_label']
        return image, label, row.to_dict()
