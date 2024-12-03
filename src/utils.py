import torch
from safetensors.torch import save_file, load_file

def save_model(model, save_path):
    try:
        save_file(model.state_dict(), save_path)
        print(f"Model berhasil disimpan di {save_path}")
    except Exception as e:
        print(f"Gagal menyimpan model: {str(e)}")

def load_model(model_class, save_path, device='cuda'):
    try:
        model = model_class()
        model.load_state_dict(load_file(save_path))
        model.to(device)
        print(f"Model berhasil dimuat dari {save_path}")
        return model
    except Exception as e:
        print(f"Gagal memuat model: {str(e)}")
        return None
