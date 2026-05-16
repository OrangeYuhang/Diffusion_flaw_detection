# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    对所有非批次维度取均值。
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    模型预测的输出类型。
    """

    PREVIOUS_X = enum.auto()  # 模型预测 x_{t-1}
    START_X = enum.auto()  # 模型预测 x_0
    EPSILON = enum.auto()  # 模型预测 epsilon


class ModelVarType(enum.Enum):
    """
    模型输出方差使用何种类型。
    LEARNED_RANGE 选项被添加以使模型能够预测介于
    FIXED_SMALL 和 FIXED_LARGE 之间的值，从而简化其任务。
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # 使用原始 MSE 损失（学习方差时加上 KL）
    RESCALED_MSE = (
        enum.auto()
    )  # 使用原始 MSE 损失（学习方差时使用 RESCALED_KL）
    KL = enum.auto()  # 使用变分下界
    RESCALED_KL = enum.auto()  # 类似 KL，但重新缩放以估计完整的 VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    这是已弃用的用于创建 beta 调度表的 API。
    请参阅 get_named_beta_schedule() 以获取新的调度表库。
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1（JSD 调度）
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    根据给定名称获取预定义的 beta 调度表。
    beta 调度表库包含在 num_diffusion_timesteps 极限下保持相似的 beta 调度表。
    beta 调度表可以添加，但一旦提交后不应删除或更改，
    以保持向后兼容性。
    """
    if schedule_name == "linear":
        # 来自 Ho 等人的线性调度表，扩展以适用于任意数量的扩散步。
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建一个 beta 调度表，将给定的 alpha_t_bar 函数离散化，
    该函数定义了 (1-beta) 在 t = [0,1] 上的累积乘积。
    :param num_diffusion_timesteps: 要生成的 beta 数量。
    :param alpha_bar: 一个 lambda 函数，接受 0 到 1 的参数 t，
                      并产生扩散过程中到该部分为止的 (1-beta) 累积乘积。
    :param max_beta: 使用的最大 beta 值；使用小于 1 的值以防止奇点。
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    用于训练和采样扩散模型的工具集。
    原始代码移植自:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: 一个 1 维 numpy 数组，包含每个扩散时间步的 beta 值，
                  从 T 开始到 1。
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # 使用 float64 以保证精度。
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 扩散过程 q(x_t | x_{t-1}) 等的计算
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # 后验分布 q(x_{t-1} | x_t, x_0) 的计算
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 以下：对数计算进行了截断，因为后验方差在扩散链的起始处为 0
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        # self.defect_w = th.nn.Parameter(torch.FloatTensor(1), requires_grad=True)


    def q_mean_variance(self, x_start, t):
        """
        获取分布 q(x_t | x_0)。
        :param x_start: 无噪声输入的 [N x C x ...] tensor。
        :param t: 扩散步数（减 1）。这里 0 表示一步。
        :return: 一个元组 (mean, variance, log_variance)，形状均与 x_start 相同。
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        对数据执行给定步数的扩散。
        换句话说，从 q(x_t | x_0) 中采样。
        :param x_start: 初始数据批次。
        :param t: 扩散步数（减 1）。这里 0 表示一步。
        :param noise: 如果指定，则为预先采样的正态噪声。
        :return: x_start 的加噪版本。
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算扩散后验分布的均值和方差：
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        应用模型获取 p(x_{t-1} | x_t)，以及初始 x (x_0) 的预测。
        :param model: 模型，接受信号和一批时间步作为输入。
        :param x: 在时间 t 的 [N x C x ...] tensor。
        :param t: 时间步的 1 维 Tensor。
        :param clip_denoised: 如果为 True，将去噪信号裁剪到 [-1, 1]。
        :param denoised_fn: 如果不为 None，则在采样前应用于 x_start 预测的函数。
                            在 clip_denoised 之前应用。
        :param model_kwargs: 如果不为 None，则为传递给模型的额外关键字参数字典。
                             可用于条件控制。
        :return: 一个包含以下键的字典：
                 - 'mean': 模型均值输出。
                 - 'variance': 模型方差输出。
                 - 'log_variance': 'variance' 的对数。
                 - 'pred_xstart': 对 x_0 的预测。
        """
        if model_kwargs is None:
            model_kwargs = {}
        new_mask=None
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if model_kwargs == {}:
            model_output = model(x, t, **model_kwargs)
        else:
            model_output, new_mask, _ = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # model_var_values 在 [-1, 1] 范围内，对应 [min_var, max_var]。
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # 对于 FIXED_LARGE，我们如此设置初始（对数）方差，
                # 以获得更好的解码器对数似然。
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        if new_mask is None:
            new_mask = pred_xstart
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
            "mask":new_mask,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        给定函数 cond_fn（计算条件对数概率关于 x 的梯度），计算上一步的均值。
        具体来说，cond_fn 计算 grad(log(p(y|x)))，我们希望以 y 为条件。
        这使用了 Sohl-Dickstein 等人 (2015) 的条件策略。
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean


    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        计算如果模型的 score 函数以 cond_fn 为条件时，
        p_mean_variance 原本应有的输出。
        有关 cond_fn 的详细信息，请参阅 condition_mean()。
        与 condition_mean() 不同，此处使用 Song 等人 (2020) 的条件策略。
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        在给定时间步从模型中采样 x_{t-1}。
        :param model: 要从中采样的模型。
        :param x: 当前在 x_{t-1} 的 tensor。
        :param t: t 的值，第一个扩散步从 0 开始。
        :param clip_denoised: 如果为 True，将 x_start 预测裁剪到 [-1, 1]。
        :param denoised_fn: 如果不为 None，则在采样前应用于 x_start 预测的函数。
        :param cond_fn: 如果不为 None，则是一个行为类似模型的梯度函数。
        :param model_kwargs: 如果不为 None，则为传递给模型的额外关键字参数字典。
                             可用于条件控制。
        :return: 一个包含以下键的字典：
                 - 'sample': 来自模型的随机样本。
                 - 'pred_xstart': 对 x_0 的预测。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0 时无噪声
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mask":out["mask"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        从模型生成样本。
        :param model: 模型模块。
        :param shape: 样本的形状 (N, C, H, W)。
        :param noise: 如果指定，则为从编码器采样的噪声。
                      应与 `shape` 具有相同的形状。
        :param clip_denoised: 如果为 True，将 x_start 预测裁剪到 [-1, 1]。
        :param denoised_fn: 如果不为 None，则在采样前应用于 x_start 预测的函数。
        :param cond_fn: 如果不为 None，则是一个行为类似模型的梯度函数。
        :param model_kwargs: 如果不为 None，则为传递给模型的额外关键字参数字典。
                             可用于条件控制。
        :param device: 如果指定，则为在其上创建样本的设备。
                       如果未指定，则使用模型参数的设备。
        :param progress: 如果为 True，显示 tqdm 进度条。
        :return: 不可微分的样本批次。
        """
        final = None
        mask = 0
        num = 0
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):

            num += 1
            final = sample
            if num > 45:
                mask += sample["mask"]
        return final["sample"], mask / 5

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        从模型生成样本，并产出扩散每个时间步的中间样本。
        参数与 p_sample_loop() 相同。
        返回一个字典生成器，其中每个字典是 p_sample() 的返回值。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 延迟导入，以免依赖 tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型中采样 x_{t-1}。
        用法同 p_sample()。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 通常模型输出 epsilon，但以防我们使用了 x_start 或 x_prev 预测，
        # 这里重新推导它。
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # 公式 12。
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0 时无噪声
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用 DDIM 逆 ODE 从模型中采样 x_{t+1}。
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # 通常模型输出 epsilon，但以防我们使用了 x_start 或 x_prev 预测，
        # 这里重新推导它。
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # 公式 12 的逆。
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型生成样本。
        用法同 p_sample_loop()。
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型采样，并产出 DDIM 每个时间步的中间样本。
        用法同 p_sample_loop_progressive()。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 延迟导入，以免依赖 tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        获取变分下界的项。
        结果单位为比特（bit），而非人们可能预期的 nat。
        这便于与其他论文进行比较。
        :return: 一个包含以下键的字典：
                 - 'output': 形状为 [N] 的 NLL 或 KL tensor。
                 - 'pred_xstart': x_0 的预测值。
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # 在第一个时间步返回解码器 NLL，
        # 否则返回 KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, label_mask=None, mask_resize=None, mask_att=None):
        """
        计算单个时间步的训练损失。
        :param model: 要评估损失的模型。
        :param x_start: 输入的 [N x C x ...] tensor。
        :param t: 一批时间步索引。
        :param model_kwargs: 如果不为 None，则为传递给模型的额外关键字参数字典。
                             可用于条件控制。
        :param noise: 如果指定，则为要尝试去除的特定高斯噪声。
        :param label_mask: 标签掩码（调节损失）
        :return: 一个包含键 "loss" 的字典，其中包含形状为 [N] 的 tensor。
                 某些均值或方差设置可能还包含其他键。
        """

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output, att_mask, att_loss = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # 使用变分界学习方差，但不让它影响我们的均值预测。
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # 除以 1000 以与初始实现保持一致。
                    # 没有 1/1000 因子的话，VB 项会损害 MSE 项。
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            # ---- 前景/背景分离损失 ----
            # mask_resize: (N, C, H, W) — 扩展到潜在通道的缺陷区域掩码
            # target / model_output: (N, C, H, W)
            diff_sq = (target - model_output) ** 2

            if mask_resize is not None and mask_resize.shape == diff_sq.shape:
                # 使用 mask_resize 作为空间权重：降低背景权重，提高缺陷权重
                # mask_resize 值：缺陷区域为 1.0，背景区域为 0.0（变换后）
                # 软混合以避免硬边界伪影
                mask_fg = mask_resize
                mask_bg = 1.0 - mask_fg

                # 每个样本的前景和背景 MSE
                loss_fg = mean_flat(diff_sq * mask_fg)  # 缺陷区域
                loss_bg = mean_flat(diff_sq * mask_bg)  # 背景区域

                # 自适应前景权重：当掩码面积较小时权重更高
                fg_ratio = mask_fg.mean(dim=[1, 2, 3], keepdim=False).clamp(min=0.01)
                fg_weight = 3.0 / (fg_ratio + 0.1)  # 较小缺陷获得更高权重

                loss_diffusion = loss_bg + fg_weight * loss_fg
            else:
                loss_diffusion = mean_flat(diff_sq)

            # ---- 注意力对齐损失 ----
            # att_loss: 来自模型的交叉注意力热力图 (N, H//2, W//2)
            # mask_att: 相同分辨率的地面真值掩码 (N, H//2, W//2)
            if att_loss is not None and mask_att is not None:
                # BCE 风格注意力对齐：鼓励在掩码上的注意力，抑制其他地方的注意力
                att_pos = att_loss * mask_att       # 掩码内的注意力
                att_neg = att_loss * (1 - mask_att) # 掩码外的注意力
                # 希望掩码内注意力高，掩码外注意力低
                loss_att_align = mean_flat(att_neg) - mean_flat(att_pos) * 0.1
                # 仅截断极端异常值，不截断信号本身
                loss_att_align = th.clamp(loss_att_align, -10.0, 10.0)
            else:
                loss_att_align = th.tensor(0.0, device=target.device)

            # ---- 掩码质量损失（Dice + MSE） ----
            mse_mask = mean_flat((att_mask - label_mask) ** 2)

            # Dice 损失以获得更好的掩码边界质量
            if label_mask is not None:
                smooth = 1.0
                # min-max 归一化到 [0, 1]，确保 Dice 在数学上良定义
                att_min = att_mask.amin(dim=[1, 2, 3], keepdim=True)
                att_max = att_mask.amax(dim=[1, 2, 3], keepdim=True)
                att_norm = (att_mask - att_min) / (att_max - att_min + 1e-8)
                lab_min = label_mask.amin(dim=[1, 2, 3], keepdim=True)
                lab_max = label_mask.amax(dim=[1, 2, 3], keepdim=True)
                lab_norm = (label_mask - lab_min) / (lab_max - lab_min + 1e-8)
                intersection = (att_norm * lab_norm).sum(dim=[1, 2, 3])
                union = (att_norm + lab_norm).sum(dim=[1, 2, 3])
                dice = (2.0 * intersection + smooth) / (union + smooth)
                loss_dice = (1.0 - dice).mean()
            else:
                loss_dice = th.tensor(0.0, device=target.device)

            terms["mse"] = mean_flat(diff_sq)
            terms["mask_mse"] = mse_mask
            terms["mask_dice"] = loss_dice
            terms["att_align"] = loss_att_align

            if "vb" in terms:
                terms["loss"] = (loss_diffusion + terms["vb"]
                                 + 1.0 * mse_mask + 5.0 * loss_dice
                                 + 1.0 * loss_att_align)
            else:
                terms["loss"] = (loss_diffusion
                                 + 1.0 * mse_mask + 5.0 * loss_dice
                                 + 1.0 * loss_att_align)
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        获取变分下界的先验 KL 项，以 bits-per-dim 为单位。
        此项无法优化，因为它只依赖于编码器。
        :param x_start: 输入的 [N x C x ...] tensor。
        :return: 一批 [N] KL 值（单位为比特），每个批次元素一个。
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算完整的变分下界（以 bits-per-dim 为单位）以及其他相关量。
        :param model: 要评估损失的模型。
        :param x_start: 输入的 [N x C x ...] tensor。
        :param clip_denoised: 如果为 True，裁剪去噪样本。
        :param model_kwargs: 如果不为 None，则为传递给模型的额外关键字参数字典。
                             可用于条件控制。
        :return: 一个包含以下键的字典：
                 - total_bpd: 每个批次元素的完整变分下界。
                 - prior_bpd: 下界中的先验项。
                 - vb: 下界中各项的 [N x T] tensor。
                 - xstart_mse: 每个时间步 x_0 MSE 的 [N x T] tensor。
                 - mse: 每个时间步 epsilon MSE 的 [N x T] tensor。
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # 计算当前时间步的 VLB 项
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从 1 维 numpy 数组中为一批索引提取值。
    :param arr: 1 维 numpy 数组。
    :param timesteps: 要提取的数组索引的 tensor。
    :param broadcast_shape: K 维的较大形状，其中批次维度等于时间步的长度。
    :return: 形状为 [batch_size, 1, ...] 的 tensor，其中形状有 K 维。
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
