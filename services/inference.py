import torch
import numpy as np
from models.model_definitions import IDSModelHybrid

MODEL_PATH = "models/trained_ids_model.pth"
NUM_FEATURES = 78  # change if your model uses different feature count
NUM_CLASSES = 2   # binary: normal vs attack


@torch.no_grad()
def load_model():
    model = IDSModelHybrid(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )
    model.eval()
    return model


def predict(model, features):
    """
    features: list or numpy array of shape (NUM_FEATURES,)
    """
    x = np.array(features, dtype=np.float32)
    x = torch.tensor(x).unsqueeze(0)  # (1, features)

    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    return {
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item())
    }

