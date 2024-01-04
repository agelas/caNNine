import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

class CannineModel(nn.Module):
    def __init__(self, num_classes):
        super(CannineModel, self).__init__()
        self.base_model = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class InferencePipeline:
    def __init__(self, fine_tuned_path):
        weights_default = ResNet18_Weights.DEFAULT
        resnet_base = resnet18(weights = weights_default)
        resnet_base.eval()

        resnet_fine_tuned = CannineModel(4)
        resnet_fine_tuned.load_state_dict(torch.load(fine_tuned_path, map_location = torch.device('cpu')))
        resnet_fine_tuned.eval()

        self.base_model = resnet_base
        self.fine_tune = resnet_fine_tuned
        self.target_categories = [207, 208, 250, 258] # Classes from default ResNet that we want
        self.inference_mapping = {0: 207, 1: 208, 2: 250, 3:258} # Mapping for fine-tuned ResNet-18
        self.categories = weights_default.meta["categories"]
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # For testing
    def load_image(self, image_path):
        image = Image.open(image_path)
        return self.image_transform(image).unsqueeze(0)
    
    def classify_image(self, image):
        image = self.image_transform(image).unsqueeze(0)

        # Base Model Inference
        with torch.no_grad():
            base_output = self.base_model(image)
            base_probabilities = torch.nn.functional.softmax(base_output, dim=1)
            base_top_prob, base_top_class = torch.max(base_probabilities, 1)

        if base_top_class.item() in self.target_categories and base_top_prob > 0.5:
            with torch.no_grad():
                finetuned_output = self.fine_tune(image)
                finetuned_probabilities = torch.nn.functional.softmax(finetuned_output, dim=1)
                finetuned_top_prob, finetuned_top_class = torch.max(finetuned_probabilities, 1)
            
            final_prob = finetuned_top_prob.item()
            final_class_idx = finetuned_top_class.item()
            mapped_index = self.inference_mapping[final_class_idx] # Need to use the interop because fine tuned model only has 4 outputs that are indexed to 0-3
            final_class = self.categories[mapped_index]
        else:
            final_prob = base_top_prob.item()
            final_class_id = base_top_class.item()
            final_class = self.categories[final_class_id]
        
        return final_class, final_prob
