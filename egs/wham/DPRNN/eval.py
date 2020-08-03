import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import queue
import time
import glob

from asteroid import DPRNNTasNet
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex",
    type=int,
    default=100,
    help="Number of audio examples to save, -1 means all",
)
parser.add_argument("--n_workers", type=int, default=5)

compute_metrics = ["si_sdr"]  # , 'sdr', 'sir', 'sar', 'stoi']


def process_single(
    conf, model, model_device, loss_func, idx, mix, sources, save_as_wav
):
    # Forward the network on the mixture.
    mix, sources = tensors_to_device(
        [torch.from_numpy(mix), torch.from_numpy(sources)], device=model_device,
    )
    est_sources = model(mix[None, None])
    loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
    mix_np = mix[None].cpu().data.numpy()
    sources_np = sources.cpu().data.numpy()
    est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
    utt_metrics = get_metrics(
        mix_np,
        sources_np,
        est_sources_np,
        sample_rate=conf["sample_rate"],
        metrics_list=compute_metrics,
    )

    # Save some examples in a folder. Wav files and metrics as text.
    if save_as_wav:
        local_save_dir = os.path.join(conf["exp_dir"], "examples", f"ex_{idx}")
        os.makedirs(local_save_dir, exist_ok=True)
        sf.write(
            os.path.join(local_save_dir, "mixture.wav"), mix_np[0], conf["sample_rate"],
        )
        # Loop over the sources and estimates
        for src_idx, src in enumerate(sources_np):
            sf.write(
                os.path.join(local_save_dir, f"s{src_idx+1}.wav"),
                src,
                conf["sample_rate"],
            )
        for src_idx, est_src in enumerate(est_sources_np):
            sf.write(
                os.path.join(local_save_dir, f"s{src_idx+1}_estimate.wav"),
                est_src,
                conf["sample_rate"],
            )
        # Write local metrics to the example folder.
        with open(os.path.join(local_save_dir, "metrics.json"), "w") as f:
            json.dump(utt_metrics, f, indent=0)

    return utt_metrics


def worker(worker_id, conf, input_queue, output_queue):
    if 0:
        model_path = os.path.join(conf["exp_dir"], "best_model.pth")
        model = DPRNNTasNet.from_pretrained(model_path)
    else:
        model_path = sorted(glob.glob(os.path.join(conf["exp_dir"], "_ckp*")))[-1]
        print("Using ckpt", model_path)
        model = DPRNNTasNet(
            **conf["train_conf"]["filterbank"], **conf["train_conf"]["masknet"], n_src=1
        )
        if "NO_GPU" in os.environ:
            checkpoint = torch.load(model_path, map_location="cpu")
        else:
            checkpoint = torch.load(model_path)
        model.load_state_dict(
            {k.split(".", 1)[1]: v for k, v in checkpoint["state_dict"].items()}
        )

    # Handle device placement
    if conf["use_gpu"] and not "NO_GPU" in os.environ:
        print("Using GPU")
        model.cuda()
    model_device = next(model.parameters()).device
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    while input_queue.empty():
        print("Waiting for producer")
        time.sleep(3)

    try:
        with torch.no_grad():
            while True:
                idx, mix_path, mix, sources, save_as_wav = input_queue.get()
                try:
                    utt_metrics = process_single(
                        conf, model, model_device, loss_func, idx, mix, sources, save_as_wav
                    )
                except Exception as err:
                    raise RuntimeError(f"Error processing f{mix_path} {mix.shape}/{sources.shape}") from err
                utt_metrics["mix_path"] = mix_path
                output_queue.put((worker_id, None, (idx, utt_metrics)))
    except Exception as err:
        output_queue.put((worker_id, err, None))
        raise


def main(conf):
    input_queue = torch.multiprocessing.Queue(maxsize=2 * conf["n_workers"])
    output_queue = torch.multiprocessing.Queue(maxsize=2 * conf["n_workers"])

    processes = [
        torch.multiprocessing.Process(
            target=worker, args=(n, conf, input_queue, output_queue), daemon=True
        )
        for n in range(conf["n_workers"])
    ]
    for process in processes:
        process.start()

    import ds

    test_set = ds.getds(True, {"data": {"sample_rate": conf["sample_rate"]}})

    # Randomly choose the indexes of sentences to save.
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])

    series_list = [None] * len(test_set)
    last_input_idx = 0
    input_idx = 0
    progress = tqdm(total=len(test_set))
    while True:
        done = input_idx >= len(test_set)
        if last_input_idx != input_idx:
            progress.update(input_idx - last_input_idx)
            last_input_idx = input_idx
        while not done and input_idx < len(test_set) and not input_queue.full():
            mix_path = test_set.ds.vctk_files[test_set.ds.items[input_idx][0]]
            mix, sources = test_set[input_idx]
            if mix.shape[0] < 1_000:
                print(f"Error with {mix_path}, skipping.")
            else:
                input_queue.put(
                    (
                        input_idx,
                        (test_set.ds.items[input_idx], mix_path),
                        mix, sources, input_idx in save_idx,
                    )
                )
            input_idx += 1
        while not output_queue.empty():
            worker_id, err, data = output_queue.get()
            if err:
                for process in processes:
                    process.terminate()
                raise RuntimeError(f"Error in worker {worker_id}") from err
            else:
                output_idx, utt_metrics = data
                series_list[output_idx] = pd.Series(utt_metrics)
        if done:
            break
        else:
            time.sleep(0.1)

    print("Queue sizes (in/out):", input_queue.qsize(), output_queue.qsize())

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame([x for x in series_list if x is not None])
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(conf["exp_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
