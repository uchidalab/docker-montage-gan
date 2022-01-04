from torch import cuda


def check_env():
    print("[PyTorch] CUDA check:", cuda.is_available())


if __name__ == "__main__":
    check_env()