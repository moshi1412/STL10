import torch
import numpy as np
import logging
class Metric():
    def __init__(self, num_classes,label_name):
        self.num_classes = num_classes
        self.label_name=label_name
        self.TP = torch.zeros(num_classes)
        self.FP = torch.zeros(num_classes)
        self.FN = torch.zeros(num_classes)
        
        self.precision = []
        self.recall = []
        self.f1 = []
        self.acc = []
        
        self.avg_precision = 0.0
        self.avg_recall = 0.0
        self.avg_f1 = 0.0
        self.avg_acc = 0.0

    def __after_step__(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        
        for c in range(self.num_classes):
            TP_c = ((pred == c) & (label == c)).sum().item()
            FP_c = ((pred == c) & (label != c)).sum().item()
            FN_c = ((pred != c) & (label == c)).sum().item()
            
            self.TP[c] += TP_c
            self.FP[c] += FP_c
            self.FN[c] += FN_c

    def __after_epoch__(self):
        
        eps = 1e-5  # 防止除0
        precision = []
        recall = []
        f1 = []
        acc = []

        
        for c in range(self.num_classes):
            p = self.TP[c] / (self.TP[c] + self.FP[c] + eps)
            r = self.TP[c] / (self.TP[c] + self.FN[c] + eps)
            f = 2 * p * r / (p + r + eps)
            a = self.TP[c] / (self.TP[c] + self.FP[c] + self.FN[c] + eps)

            precision.append(p.item())
            recall.append(r.item())
            f1.append(f.item())
            acc.append(a.item())

        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.acc = acc
        self.avg_precision = np.mean(precision)
        self.avg_recall = np.mean(recall)
        self.avg_f1 = np.mean(f1)
        self.avg_acc = np.mean(acc)

    def print_metric(self):
        
        #各分类指标
        for c in range(self.num_classes):
            logging.info(f"class {self.label_name[c]}  Precision: {self.precision[c]:.4f}")
            logging.info(f"class {self.label_name[c]}  Recall:    {self.recall[c]:.4f}")
            logging.info(f"class {self.label_name[c]}  F1:        {self.f1[c]:.4f}")
            logging.info(f"class {self.label_name[c]}  Acc:       {self.acc[c]:.4f}")
            logging.info("=" * 50)

        logging.info(f"Avg Precision: {self.avg_precision:.4f}") 
        logging.info(f"Avg Recall:    {self.avg_recall:.4f}")
        logging.info(f"Avg F1:        {self.avg_f1:.4f}")
        logging.info(f"Avg Acc:       {self.avg_acc:.4f}")
        logging.info("=" * 50)

    def reset(self):
        """每个 epoch 开始前清空统计"""
        self.TP = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)