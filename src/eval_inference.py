import torch
import matplotlib.pyplot as plt
import math
from tasks import LinearRegression
import numpy as np


def get_inference_noise_std(noise_levels, task_sampler, xs, model):
    ys_result = []
    pred_result = []
    loss_result = []
    for noise_level in noise_levels:
        task = task_sampler(noise_std=noise_level)
        ys = task.evaluate(xs)
        with torch.no_grad():
            pred = model(xs, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).numpy()
        ys_result.append(ys)
        pred_result.append(pred)
        loss_result.append(loss)
    return ys_result, pred_result, loss_result



def get_inference_scale(x_scales, task_sampler, xs, model, noise_std=1):
    ys_result = []
    pred_result = []
    loss_result = []
    for x_scale in x_scales:
        xs_scaled = x_scale * xs 
        task = task_sampler(noise_std=noise_std)
        ys = task.evaluate(xs_scaled)
        with torch.no_grad():
            pred = model(xs_scaled, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).numpy()
        ys_result.append(ys)
        pred_result.append(pred)
        loss_result.append(loss)
    return ys_result, pred_result, loss_result


def get_inference_baselines(baselines, task_sampler, xs, ICLmodel, noise_std=1):
    pred_result = []
    loss_result = []
    task = task_sampler(noise_std=noise_std)
    ys = task.evaluate(xs)
    ### ICL model
    with torch.no_grad():
        pred = ICLmodel(xs, ys)
    metric = task.get_metric()
    loss = metric(pred, ys).numpy()
    pred_result.append(pred)
    loss_result.append(loss) 
    ### baselines
    for baseline in baselines:
        with torch.no_grad():
            pred = baseline(xs, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).numpy()
        pred_result.append(pred)
        loss_result.append(loss)   
    return pred_result, loss_result  


def visualize_inference_noise_std(baseline, noise_levels, loss_result, noise_name):
    for i in range(len(noise_levels)):
        noise_level = noise_levels[i]
        loss = loss_result[i]
        plt.plot(loss.mean(axis=0), lw=1, label="Transformer(noise_std = {})".format(noise_level))
    plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples({} noise)".format(noise_name))
    plt.ylabel("squared error")
    plt.legend()
    plt.show()



def visualize_inference_scale(baseline, x_scales, loss_result, noise_name):
    for i in range(len(x_scales)):
        x_scale = x_scales[i]
        loss = loss_result[i]
        plt.plot(loss.mean(axis=0), lw=1, label="Transformer(x_scale = {})".format(x_scale))
    plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples({} noise)".format(noise_name))
    plt.ylabel("squared error")
    plt.legend()
    plt.show()



def visualize_inference_baselines(baselines, baseline, loss_result, noise_name):
    for i in range(len(baselines)):
        name = baselines[i]
        loss = loss_result[i]
        plt.plot(loss.mean(axis=0), lw=1, label="{}".format(name))
    plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples({} noise)".format(noise_name))
    plt.yscale('log')
    plt.ylabel("log squared error")
    plt.legend()
    plt.show()



def get_noise_task_sampler(noise_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs):
    noise_names_to_classes = {
        "gaussian": GaussianNoisyLinearRegression,
        "uniform": UniformNoisyLinearRegression,
        "expotential": ExpotentialNoisyLinearRegression,
        "poisson": PoissonNoisyLinearRegression,
    }
    if noise_name in noise_names_to_classes:
        task_cls = noise_names_to_classes[noise_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError



class GaussianNoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(GaussianNoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy



class UniformNoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(UniformNoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.rand_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class ExpotentialNoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(ExpotentialNoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        poisson_numpy = np.random.exponential(self.noise_std, ys_b.shape) - self.noise_std
        ys_b_noisy = ys_b + torch.from_numpy(poisson_numpy.astype(np.float32))
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy



class PoissonNoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(PoissonNoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        poisson_numpy = np.random.poisson(self.noise_std**2, ys_b.shape) - self.noise_std**2
        ys_b_noisy = ys_b + torch.from_numpy(poisson_numpy.astype(np.float32))
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy

