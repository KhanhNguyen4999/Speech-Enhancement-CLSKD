params = {
    "dataset": "cifar10",
    "student": "resnet20",
    "teacher": "resnet56",
    "teacher_weight_path": f"./pretrained/resnet56.pt",

    "batch_size": 64,
    "num_epochs": 20,
    "lr": 0.1,
    "lr_decay_steps": [12, 17],
    "lr_decay_rate": 0.1,
    "weight_decay": 5e-4,
    "args": 0,

    "kd_loss_weight": 0.6,

    "seed": 0,
}
