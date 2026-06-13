import torch
print(f"CUDA dostępna: {torch.cuda.is_available()}")
print(f"Liczba wykrytych GPU: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}")