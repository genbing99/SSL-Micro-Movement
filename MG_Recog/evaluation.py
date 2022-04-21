import torch
    
def topKCorrect(gt, yhat, k):
    values, indices = torch.topk(yhat, k)
    correct = 0
    for index in range(len(gt)):
        if gt[index] in indices[index]:
            correct += 1
    return correct