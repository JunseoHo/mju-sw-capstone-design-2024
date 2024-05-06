import torch

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # 윈도우 시스템의 재귀 Fork 시스템콜을 방지.
  