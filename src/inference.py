import torch
import matplotlib.pyplot as plt

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
        xs_scaled = xs * x_scale
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

def visualize_inference_noise_std(sparsity,baseline,noise_levels,loss_result):
    for i in range(len(noise_levels)):
        noise_level = noise_levels[i]
        loss = loss_result[i]
        plt.plot(loss.mean(axis=0), lw=1, label="Transformer(noise_std = {})".format(noise_level))
    plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.legend()
    plt.show()

def visualize_inference_scale(sparsity,baseline,x_scales,loss_result):
    for i in range(len(x_scales)):
        x_scale = x_scales[i]
        loss = loss_result[i]
        plt.plot(loss.mean(axis=0), lw=1, label="Transformer(x_scale = {})".format(x_scale))
    plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.legend()
    plt.show()