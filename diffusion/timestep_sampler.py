# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    从预定义的采样器库中创建一个 ScheduleSampler。
    :param name: 采样器的名称。
    :param diffusion: 要为其采样的扩散对象。
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    扩散过程中时间步的分布，旨在减少目标函数的方差。
    默认情况下，采样器执行无偏重要性采样，目标函数的均值保持不变。
    但是，子类可以重写 sample() 来改变重采样项的重新加权方式，
    从而允许实际改变目标函数。
    """

    @abstractmethod
    def weights(self):
        """
        获取一个 numpy 数组的权重，每个扩散步一个。
        权重不需要归一化，但必须为正。
        """

    def sample(self, batch_size, device):
        """
        对一个批次进行重要性采样的时间步。
        :param batch_size: 时间步的数量。
        :param device: 要保存到的 torch 设备。
        :return: 一个元组 (timesteps, weights)：
                 - timesteps: 时间步索引的 tensor。
                 - weights: 用于缩放结果损失的权重 tensor。
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        使用模型损失来更新重新加权。
        在每个 rank 上调用此方法，传入一批时间步及这些时间步对应的损失。
        此方法将执行同步，以确保所有 rank 保持完全相同的重新加权。
        :param local_ts: 时间步的整数 Tensor。
        :param local_losses: 损失的 1 维 Tensor。
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # 将 all_gather 批次填充到最大批次大小。
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        使用模型损失来更新重新加权。
        子类应重写此方法，以使用模型损失更新重新加权。
        此方法直接更新重新加权，不在 worker 之间同步。
        它由 update_with_local_losses 从所有 rank 以相同的参数调用。
        因此，它应具有确定性行为，以在各 worker 之间保持一致状态。
        :param ts: int 类型时间步的列表。
        :param losses: float 类型损失的列表，每个时间步一个。
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int_)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # 移出最旧的损失项。
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
