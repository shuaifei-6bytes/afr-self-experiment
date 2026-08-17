# ============================================================
# AFR vs SELF 联邦遗忘对比实验（Kaggle 版本）
# ============================================================
# 
# 使用方法：
# 1. 在 Kaggle 上创建新 Notebook
# 2. 设置加速器：GPU T4
# 3. 将下方代码全部复制粘贴到第一个单元格
# 4. 上传 waterbird_complete95_forest2water2 数据集到 Kaggle Datasets
# 5. 运行
#
# 预计运行时间：30-60 分钟（T4 GPU）
# ============================================================

import os
import copy
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 1. 数据集模块
# ============================================================

@dataclass
class Sample:
    """单个样本信息。"""
    img_path: str
    y: int  # 标签：0=陆鸟, 1=水鸟
    place: int  # 背景：0=陆背景, 1=水背景
    split: int  # 划分：0=train, 1=val, 2=test


class WaterbirdsSubset(Dataset):
    """Waterbirds 数据集子集。"""
    
    def __init__(self, samples: List[Sample], data_root: str, train: bool = True):
        self.samples = samples
        self.data_root = data_root
        self.train = train
        
        # 数据增强
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample.img_path)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, sample.y, sample.place, idx


def _read_metadata(csv_path: str) -> List[Sample]:
    """读取 metadata.csv。"""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(
                img_path=row['img_filename'],
                y=int(row['y']),
                place=int(row['place']),
                split=int(row['split'])
            ))
    return samples


def _assign_clients(
    samples: List[Sample],
    num_clients: int = 2,
    z0_prob: float = 0.8,
    rho_list: List[float] = None
) -> Dict[int, List[Sample]]:
    """按伪关联强度分配样本到客户端。"""
    if rho_list is None:
        rho_list = [0.9, 0.1]
    
    client_samples = {i: [] for i in range(num_clients)}
    
    for s in samples:
        if s.split != 0:  # 只要训练集
            continue
        
        # z=0: 多数群体（水背景→水鸟，陆背景→陆鸟）
        is_majority = (s.place == 1 and s.y == 1) or (s.place == 0 and s.y == 0)
        z = 0 if is_majority else 1
        
        # 分配到客户端
        if random.random() < z0_prob:
            # 按 z 分配
            client_id = 0 if z == 0 else 1
        else:
            # 随机分配
            client_id = random.randint(0, num_clients - 1)
        
        # 伪关联强度调整
        if random.random() < rho_list[client_id]:
            client_samples[client_id].append(s)
    
    return client_samples


def build_waterbirds_dataset(
    data_root: str,
    num_clients: int = 2,
    seed: int = 42
) -> Dict:
    """构建 Waterbirds 联邦数据集。"""
    random.seed(seed)
    
    # 读取 metadata
    csv_path = os.path.join(data_root, 'metadata.csv')
    all_samples = _read_metadata(csv_path)
    
    # 分配到客户端
    client_samples = _assign_clients(all_samples, num_clients)
    
    # 验证集和测试集
    val_samples = [s for s in all_samples if s.split == 1]
    test_samples = [s for s in all_samples if s.split == 2]
    
    # 统计
    print(f"Waterbirds 联邦数据集:")
    print(f"  客户端数: {num_clients}")
    for c in range(num_clients):
        samples = client_samples[c]
        minority = sum(1 for s in samples if (s.place == 0 and s.y == 1) or (s.place == 1 and s.y == 0))
        waterbirds = sum(1 for s in samples if s.y == 1)
        water_bg = sum(1 for s in samples if s.place == 1)
        print(f"  客户端 {c}: {len(samples)} 样本, 少数群体 {minority} ({minority/len(samples)*100:.1f}%), 水鸟 {waterbirds} ({waterbirds/len(samples)*100:.1f}%), 水背景 {water_bg} ({water_bg/len(samples)*100:.1f}%)")
    
    total_train = sum(len(client_samples[c]) for c in range(num_clients))
    print(f"  总训练样本: {total_train}")
    print(f"  验证集: {len(val_samples)}")
    print(f"  测试集: {len(test_samples)}")
    
    return {
        'train_data': [client_samples[c] for c in range(num_clients)],
        'val_data': val_samples,
        'test_data': test_samples,
        'data_root': data_root
    }


# ============================================================
# 2. 模型定义
# ============================================================

class BackboneHeadNet(nn.Module):
    """ResNet-18 backbone + 可替换 head。"""
    
    def __init__(self):
        super().__init__()
        
        # ResNet-18 backbone
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后 FC
        self.feature_dim = 512
        
        # 分类头
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # 冻结标志
        self._backbone_frozen = False
    
    def forward(self, x):
        features = self.backbone(x).squeeze()
        logits = self.head(features)
        return logits
    
    def freeze_backbone(self):
        """冻结 backbone 参数。"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
    
    def unfreeze_backbone(self):
        """解冻 backbone 参数。"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
    
    def get_head_params(self):
        """获取 head 参数。"""
        return self.head.parameters()


# ============================================================
# 3. 联邦训练模块
# ============================================================

def train_federated(model, dataset, cfg) -> BackboneHeadNet:
    """FedAvg 联邦训练。"""
    model = copy.deepcopy(model).to(device)
    train_data = dataset['train_data']
    
    # 客户端 DataLoader
    client_loaders = []
    for c in range(cfg['num_clients']):
        subset = WaterbirdsSubset(train_data[c], dataset['data_root'], train=True)
        loader = DataLoader(subset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
        client_loaders.append(loader)
    
    # FedAvg 循环
    for round_idx in range(cfg['fed_rounds']):
        # 客户端本地训练
        client_models = []
        for c in range(cfg['num_clients']):
            client_model = copy.deepcopy(model).to(device)
            client_model.train()
            optimizer = torch.optim.SGD(client_model.parameters(), lr=cfg['fed_lr'], momentum=0.9)
            
            for epoch in range(cfg['local_epochs']):
                for images, labels, _, _ in client_loaders[c]:
                    images, labels = images.to(device), labels.to(device)
                    logits = client_model(images)
                    loss = F.cross_entropy(logits, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            client_models.append(client_model)
        
        # 聚合
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            for client_model in client_models:
                for param_agg, param_client in zip(model.parameters(), client_model.parameters()):
                    param_agg.add_(param_client.data)
            for param in model.parameters():
                param.div_(cfg['num_clients'])
        
        if (round_idx + 1) % 10 == 0:
            # 计算平均损失
            model.eval()
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for c in range(cfg['num_clients']):
                    for images, labels, _, _ in client_loaders[c]:
                        images, labels = images.to(device), labels.to(device)
                        logits = model(images)
                        total_loss += F.cross_entropy(logits, labels).item() * len(labels)
                        count += len(labels)
            print(f"  FedAvg 轮次 {round_idx + 1}/{cfg['fed_rounds']}, 平均损失: {total_loss/count:.4f}", flush=True)
    
    model.freeze_backbone()
    return model


def train_erm_centralized(model, dataset, cfg, num_epochs=None) -> BackboneHeadNet:
    """ERM 集中式训练。"""
    if num_epochs is None:
        num_epochs = cfg['erm_epochs']
    
    model = copy.deepcopy(model).to(device)
    
    # 合并所有数据
    all_samples = []
    for c in range(cfg['num_clients']):
        all_samples.extend(dataset['train_data'][c])
    
    loader = DataLoader(
        WaterbirdsSubset(all_samples, dataset['data_root'], train=True),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['erm_lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(labels)
            count += len(labels)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ERM epoch {epoch + 1}/{num_epochs}, 损失: {epoch_loss/count:.4f}", flush=True)
    
    return model


# ============================================================
# 4. AFR 方法
# ============================================================

def compute_afr_weights(model, dataset, cfg):
    """计算 AFR 样本权重。"""
    model.eval()
    
    # 合并训练数据
    all_samples = []
    for c in range(cfg['num_clients']):
        all_samples.extend(dataset['train_data'][c])
    
    loader = DataLoader(
        WaterbirdsSubset(all_samples, dataset['data_root'], train=False),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    weights = []
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
            
            # 置信度倒数作为权重
            batch_weights = 1.0 / (max_probs.cpu().numpy() + 1e-6)
            weights.extend(batch_weights.tolist())
    
    weights = np.array(weights)
    weights = np.clip(weights, cfg['clip_low'], cfg['clip_high'])
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32)


def afr_retrain_head(model, dataset, weights, cfg):
    """AFR 重训 head。"""
    model = copy.deepcopy(model).to(device)
    model.freeze_backbone()
    
    # 合并数据
    all_samples = []
    for c in range(cfg['num_clients']):
        all_samples.extend(dataset['train_data'][c])
    
    loader = DataLoader(
        WaterbirdsSubset(all_samples, dataset['data_root'], train=True),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    model.train()
    optimizer = torch.optim.SGD(model.get_head_params(), lr=cfg['head_retrain_lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['head_retrain_epochs'], eta_min=1e-5)
    
    # 权重字典
    weight_dict = {i: weights[i].item() for i in range(len(weights))}
    
    for epoch in range(cfg['head_retrain_epochs']):
        for images, labels, _, indices in loader:
            images, labels = images.to(device), labels.to(device)
            
            # 获取批次权重
            batch_weights = torch.tensor([weight_dict[i.item()] for i in indices]).to(device)
            
            logits = model(images)
            loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
            loss = (loss_per_sample * batch_weights).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    return model


# ============================================================
# 5. SELF 方法
# ============================================================

def compute_self_weights(erm_model, early_model, dataset, cfg):
    """计算 SELF 样本权重（基于模型分歧）。"""
    erm_model.eval()
    early_model.eval()
    
    # 合并数据
    all_samples = []
    for c in range(cfg['num_clients']):
        all_samples.extend(dataset['train_data'][c])
    
    loader = DataLoader(
        WaterbirdsSubset(all_samples, dataset['data_root'], train=False),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    divergences = []
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            
            # 两个模型的预测
            erm_logits = erm_model(images)
            early_logits = early_model(images)
            
            # KL 散度作为分歧度量
            erm_probs = F.softmax(erm_logits, dim=1)
            early_probs = F.softmax(early_logits, dim=1)
            kl_div = (erm_probs * (erm_probs / (early_probs + 1e-6) + 1e-6).log()).sum(dim=1)
            
            divergences.extend(kl_div.cpu().numpy())
    
    divergences = np.array(divergences)
    
    # 选择 top-k 高分歧样本
    k = max(int(len(divergences) * cfg['top_k_percent']), cfg['min_samples'])
    top_k_indices = np.argsort(-divergences)[:k]
    
    weights = np.zeros(len(divergences))
    weights[top_k_indices] = 1.0
    
    return torch.tensor(weights, dtype=torch.float32)


def self_retrain_head(model, dataset, weights, cfg):
    """SELF 重训 head。"""
    return afr_retrain_head(model, dataset, weights, cfg)


# ============================================================
# 6. 评估模块
# ============================================================

def evaluate_on_subset(model, samples, cfg):
    """评估模型在样本集上的性能。"""
    model.eval()
    
    loader = DataLoader(
        WaterbirdsSubset(samples, cfg['data_root'], train=False),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    correct = 0
    total = 0
    minority_correct = 0
    minority_total = 0
    
    with torch.no_grad():
        for images, labels, places, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            places = places.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            # 少数群体：水鸟+陆背景 或 陆鸟+水背景
            minority_mask = ((places == 0) & (labels == 1)) | ((places == 1) & (labels == 0))
            minority_correct += ((preds == labels) & minority_mask).sum().item()
            minority_total += minority_mask.sum().item()
    
    accuracy = correct / total
    minority_accuracy = minority_correct / max(minority_total, 1)
    
    return {
        'accuracy': accuracy,
        'minority_accuracy': minority_accuracy
    }


def compute_d_prob_waterbirds(model, dataset, cfg):
    """计算伪关联强度 d_prob。"""
    model.eval()
    
    # 合并训练数据
    all_samples = []
    for c in range(cfg['num_clients']):
        all_samples.extend(dataset['train_data'][c])
    
    loader = DataLoader(
        WaterbirdsSubset(all_samples, cfg['data_root'], train=False),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # 多数群体：水背景→水鸟预测概率，陆背景→陆鸟预测概率
    majority_probs = []
    # 少数群体：水背景→陆鸟预测概率，陆背景→水鸟预测概率
    minority_probs = []
    
    with torch.no_grad():
        for images, labels, places, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            places = places.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            # 水背景 (place=1): 水鸟概率
            water_bg_mask = places == 1
            if water_bg_mask.sum() > 0:
                water_bg_probs = probs[water_bg_mask, 1].cpu()
                water_bg_labels = labels[water_bg_mask]
                
                # 多数群体：水背景 + 水鸟
                majority_mask = water_bg_labels == 1
                if majority_mask.sum() > 0:
                    majority_probs.extend(water_bg_probs[majority_mask].numpy())
                
                # 少数群体：水背景 + 陆鸟
                minority_mask = water_bg_labels == 0
                if minority_mask.sum() > 0:
                    minority_probs.extend(water_bg_probs[minority_mask].numpy())
            
            # 陆背景 (place=0): 陆鸟概率
            land_bg_mask = places == 0
            if land_bg_mask.sum() > 0:
                land_bg_probs = probs[land_bg_mask, 0].cpu()
                land_bg_labels = labels[land_bg_mask]
                
                # 多数群体：陆背景 + 陆鸟
                majority_mask = land_bg_labels == 0
                if majority_mask.sum() > 0:
                    majority_probs.extend(land_bg_probs[majority_mask].numpy())
                
                # 少数群体：陆背景 + 水鸟
                minority_mask = land_bg_labels == 1
                if minority_mask.sum() > 0:
                    minority_probs.extend(land_bg_probs[minority_mask].numpy())
    
    majority_probs = np.array(majority_probs)
    minority_probs = np.array(minority_probs)
    
    d_prob = majority_probs.mean() - minority_probs.mean()
    
    return d_prob


def evaluate_all(model, dataset, cfg, method_name):
    """完整评估模型性能。"""
    print(f"\n[评估] {method_name}...")
    
    # 训练集评估
    all_train_samples = []
    for c in range(cfg['num_clients']):
        all_train_samples.extend(dataset['train_data'][c])
    
    train_metrics = evaluate_on_subset(model, all_train_samples, {'data_root': dataset['data_root'], 'batch_size': cfg['batch_size']})
    train_d_prob = compute_d_prob_waterbirds(model, dataset, cfg)
    
    # 验证集评估
    val_metrics = evaluate_on_subset(model, dataset['val_data'], {'data_root': dataset['data_root'], 'batch_size': cfg['batch_size']})
    
    print(f"  训练集准确率: {train_metrics['accuracy']:.4f}")
    print(f"  少数群体准确率: {train_metrics['minority_accuracy']:.4f}")
    print(f"  d_prob: {train_d_prob:.4f}")
    print(f"  验证集准确率: {val_metrics['accuracy']:.4f}")
    
    return {
        'method': method_name,
        'train_accuracy': train_metrics['accuracy'],
        'train_minority_accuracy': train_metrics['minority_accuracy'],
        'train_d_prob': train_d_prob,
        'val_accuracy': val_metrics['accuracy']
    }


# ============================================================
# 7. Placebo 对照
# ============================================================

def run_placebo(erm_model, dataset, cfg):
    """Placebo 对照：随机权重重训 head。"""
    placebo_model = copy.deepcopy(erm_model).to(device)
    placebo_model.freeze_backbone()
    
    # 合并数据
    all_samples = []
    for c in range(cfg['num_clients']):
        all_samples.extend(dataset['train_data'][c])
    
    # 随机权重
    random_weights = torch.tensor([random.random() * 2.0 for _ in all_samples])
    random_weights = random_weights / random_weights.mean()
    
    return afr_retrain_head(placebo_model, dataset, random_weights, cfg)


# ============================================================
# 8. 主实验流程
# ============================================================

def main():
    """主实验流程。"""
    print("=" * 60)
    print("AFR vs SELF 完整对比实验（Waterbirds 数据集）")
    print("=" * 60)
    
    # 配置
    cfg = {
        'num_clients': 2,
        'fed_rounds': 30,
        'local_epochs': 1,
        'batch_size': 32,
        'fed_lr': 0.001,
        'erm_lr': 0.001,
        'erm_epochs': 15,
        'early_stop_epochs': 5,
        'head_retrain_lr': 0.001,
        'head_retrain_epochs': 30,
        'clip_low': 0.1,
        'clip_high': 10.0,
        'top_k_percent': 0.2,
        'min_samples': 100,
    }
    
    # 数据目录（自动探测：先扫 /kaggle/input，再扫 /kaggle/working）
    data_root = None
    for base in ['/kaggle/input', '/kaggle/working', '.']:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            if 'metadata.csv' in files:
                data_root = root
                break
        if data_root:
            break

    if data_root is None:
        print("错误：未找到 metadata.csv")
        print("请在 Notebook 里先执行以下命令下载数据集：")
        print("  !wget -q https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz")
        print("  !tar -xzf waterbird_complete95_forest2water2.tar.gz")
        print("\n当前 /kaggle/input 目录结构：")
        for root, dirs, files in os.walk('/kaggle/input'):
            print(f"  {root}: {dirs[:8]} {files[:8]}")
        return
    print(f"找到数据集: {data_root}")
    
    # Step 1: 构建数据集
    print("\n[Step 1] 构建数据集...")
    dataset = build_waterbirds_dataset(data_root, num_clients=cfg['num_clients'], seed=42)
    
    # Step 2: FedAvg 联邦训练
    print("\n[Step 2] FedAvg 联邦训练...")
    model = BackboneHeadNet()
    fed_model = train_federated(model, dataset, cfg)
    
    # Step 3: ERM
    print("\n[Step 3] ERM 训练...")
    erm_model = BackboneHeadNet()
    erm_model = train_erm_centralized(erm_model, dataset, cfg)
    
    # Step 4: Early-stopped
    print("\n[Step 4] Early-stopped 模型...")
    early_model = BackboneHeadNet()
    early_model = train_erm_centralized(early_model, dataset, cfg, num_epochs=cfg['early_stop_epochs'])
    
    # 评估
    print("\n[评估] ERM 基线...")
    erm_model.freeze_backbone()
    erm_results = evaluate_all(erm_model, dataset, cfg, "ERM")
    
    print("\n[评估] AFR...")
    afr_weights = compute_afr_weights(erm_model, dataset, cfg)
    afr_model = afr_retrain_head(erm_model, dataset, afr_weights, cfg)
    afr_results = evaluate_all(afr_model, dataset, cfg, "AFR")
    
    print("\n[评估] SELF...")
    self_weights = compute_self_weights(erm_model, early_model, dataset, cfg)
    self_model = self_retrain_head(erm_model, dataset, self_weights, cfg)
    self_results = evaluate_all(self_model, dataset, cfg, "SELF")
    
    print("\n[评估] Placebo...")
    placebo_model = run_placebo(erm_model, dataset, cfg)
    placebo_results = evaluate_all(placebo_model, dataset, cfg, "Placebo")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("完整实验结果汇总")
    print("=" * 60)
    
    results = [erm_results, placebo_results, afr_results, self_results]
    
    header = f"{'方法':<12} {'训练acc':>10} {'少数acc':>10} {'d_prob':>10} {'验证acc':>10}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        print(f"{r['method']:<12} {r['train_accuracy']:>10.4f} {r['train_minority_accuracy']:>10.4f} {r['train_d_prob']:>10.4f} {r['val_accuracy']:>10.4f}")
    
    # 保存结果
    import json
    output_path = '/kaggle/working/comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    return results


if __name__ == "__main__":
    results = main()
