from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchmetrics
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch.optim.lr_scheduler import StepLR

batch_size = 64

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, teacher, student, T=20, alpha=0.5):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.T = T
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        teacher_outputs = self.teacher(x) / self.T
        student_outputs = self.student(x) / self.T
        
        ce_loss = self.ce_loss(student_outputs, y)
        kd_loss = nn.KLDivLoss()(torch.log_softmax(student_outputs, dim=1), torch.softmax(teacher_outputs, dim=1))
        loss = (1 - self.alpha) * ce_loss + self.alpha * self.T * self.T * kd_loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.student(x)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log("val_acc", acc)
        return acc
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        #scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)
        return optimizer
    
    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=4)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=False, 
                                           transform=transforms.ToTensor(),  
                                           download=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers=4)
        return val_loader



#check gpu
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#setup
torch.set_float32_matmul_precision('high')

# initialize models
teacher = TeacherNet()
student = StudentNet()

# initialize trainer
trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=[1])

# initialize knowledge distillation module
kd_module = KnowledgeDistillation(teacher, student)

# train the student network using knowledge distillation
trainer.fit(kd_module)


