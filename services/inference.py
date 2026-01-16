import torch
import numpy as np
from model_definitions import IDSModelHybrid

MODEL_PATH = "trained_ids_model.pth"

SEQ_LEN = 1
FEAT_DIM = 78
NUM_CLASSES = 15

@torch.no_grad()
def load_model():
    model = IDSModelHybrid(
        seq_len=SEQ_LEN,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        use_attention=True
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, features):
    x = np.array(features, dtype=np.float32)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0)  # (1, seq, feat)

    logits, _, _ = model(x)
    probs = torch.softmax(logits, dim=1)

    return {
        "prediction": int(torch.argmax(probs, dim=1)),
        "confidence": float(torch.max(probs))
    }

