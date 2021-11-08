"""
SMSR 학습, 추론 코드

Writer : KHS0616
Last Update : 2021-11-03
"""
import torch
from torch.optim.lr_scheduler import StepLR
from torch.serialization import save
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

from model import SMSR
from datasets import TrainDatasets

class Trainer():
    """ 학습 클래스 """
    def __init__(self):
        self.setDevice()
        self.setDataLoader()
        self.setModel()
        self.setOptimizer()
        self.setLoss()

    def setDevice(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def setDataLoader(self):
        self.datasets = TrainDatasets()
        self.dataloader = DataLoader(
            dataset=self.datasets,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )

    def setModel(self):
        self.model = SMSR().to(self.device)

    def setOptimizer(self):
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=2e-4, betas=(0.9, 0.999))
        self.scheduler = StepLR(self.optimizer, step_size=200, gamma=0.5)

    def setLoss(self):
        self.loss_func = torch.nn.L1Loss()

    def process(self):
        # lr 설정
        lr = 2e-4 * (2 ** -(self.scheduler.last_epoch+1 // 200))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(1, 600, 1):
            # 학습
            for batch, (lr, hr, filename) in enumerate(self.dataloader):
                lr, hr = lr.to(self.device), hr.to(self.device)

                # tau 업데이트
                tau = max(1 - (epoch-1 - 1) / 500, 0.4)
                for m in self.model.modules():
                    if hasattr(m, '_update_tau'):
                        m._update_tau(tau)

                # inference
                self.optimizer.zero_grad()
                sr, sparsity = self.model(lr)

                # Loss 측정
                loss_SR = self.loss_func(sr, hr)
                loss_sparsity = sparsity.mean()
                lambda0 = 0.1
                lambda_sparsity = min((epoch-1 -1) / 50, 1) * lambda0
                loss = loss_SR + lambda_sparsity * loss_sparsity

                # backpropagation
                loss.backward()
                self.optimizer.step()

                # Check Train status
                print(batch)
                if batch % 10 == 0:
                    grid_tensor = make_grid([F.interpolate(lr, size=192)[0], sr[0], hr[0]], normalize=True)
                    save_image(grid_tensor, f"results/SMSR-{epoch*(batch+1)}.png")

            # stpe scheduler
            self.scheduler.step()

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"results/SMSR-{epoch}.pth")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.process()