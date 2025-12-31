import shap
import numpy as np
import torch

def explain_prediction(model, background_data, sample):
    """
    model: trained PyTorch model
    background_data: numpy array (e.g., 50 rows)
    sample: numpy array (1 row)
    """

    def model_forward(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()

    explainer = shap.KernelExplainer(
        model_forward,
        background_data
    )

    shap_values = explainer.shap_values(sample)

    return shap_values
