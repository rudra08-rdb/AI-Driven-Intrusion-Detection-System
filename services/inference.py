import torch
import numpy as np
from model_definitions import IDSModelHybrid

SEQ_LEN = 1
FEAT_DIM = 78
NUM_CLASSES = 15
MODEL_PATH = "trained_ids_model.pth"

def load_model():
    model = IDSModelHybrid(
        seq_len=SEQ_LEN,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        use_attention=True
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, feature_vector):
    x = np.array(feature_vector, dtype=np.float32)
    x = x.reshape(1, SEQ_LEN, FEAT_DIM)
    x_tensor = torch.tensor(x)

    with torch.no_grad():
        logits, _, attn = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

    return {
        "prediction": pred,
        "confidence": conf,
        "attention": attn
    }

