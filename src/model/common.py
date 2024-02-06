import os

import torch
import platform
from torch import nn
from tqdm import tqdm


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f"BioLearning  Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    return torch.device(arg)


def precision_cal(predictor, dataloader, device):
    predictor.eval()  # 将模型设置为评估模式
    criterion = nn.BCELoss()
    sample_total = 0

    total_output = []
    threshold = 1.0e-01
    with torch.no_grad():  # 在评估过程中不计算梯度
        test_loss = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for testImgs, testLabel in dataloader:  # 假设test_loader是您的测试数据加载器
            testImgs, testLabel = testImgs.to(device), testLabel.to(device)
            outputs = predictor(testImgs)
            test_loss += criterion(outputs, testLabel.float()).item()

            # 使用张量操作计算TP, FP, TN, FN
            predicted = (outputs >= threshold).float()  # 预测结果大于等于阈值的为1，否则为0
            TP += (predicted * testLabel).sum().item()
            TN += ((1 - predicted) * (1 - testLabel)).sum().item()
            FP += (predicted * (1 - testLabel)).sum().item()
            FN += ((1 - predicted) * testLabel).sum().item()
            sample_total += testLabel.numel()  # 更新样本总数

    avg_accuracy = ((TP + TN) / sample_total)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    avg_test_loss = test_loss / len(dataloader)  # 计算平均损失

    return (avg_accuracy, precision, recall), avg_test_loss, total_output


def model_train(model: nn.Module, device, dataloader, criterion, EPOCHS: int, optimizer, autoencode=False):
    model.train()
    loss_value = []
    for epoch in range(1, EPOCHS + 1):
        loss = 0
        processBar = tqdm(dataloader, unit='step')
        for step, (frames, labels) in enumerate(processBar):
            frames, labels = frames.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            # 前向传播
            outputs = model(frames)
            comp = frames if autoencode else labels.float()
            # 计算损失
            loss = criterion(outputs, comp)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            processBar.set_description(f"[{epoch}/{EPOCHS}] Loss: {loss.item():.10f}")
        loss_value.append(loss.item())
    return model, loss_value