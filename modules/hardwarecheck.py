from ultralytics import checks
import platform
import torch

# Cek informasi Python & YOLO version
def pythyoloversion():
    print("============== DETAIL SOFTWARE ==============")
    print(f"Versi Python Terinstal : ", platform.python_version()),
    print(f"Versi YOLOv8 (Ultralytics) : ")
    checks()
    print("=============================================")

# Cek informasi GPU dan CPU Hardware
def cudagpu():
    print("============== DETAIL HARDWARE ==============")
    print(f"Versi Pytorch Terinstal :",torch.__version__,
        "\nApakah GPU Tersedia? ", torch.cuda.is_available(),
        "\nVersi CUDA : ", torch.version.cuda,
        "\nNama Device :", torch.cuda.get_device_name(0))
    print("=============================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device