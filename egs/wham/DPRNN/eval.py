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

from asteroid import DPRNNTasNet
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device


parser = argparse.ArgumentParser()

compute_metrics = ['si_sdr', 'stoi']


def main(conf):
    conf.update(conf['eval'])
    conf['sample_rate'] = conf['data']['sample_rate']
    from asteroid.data.vbd_dataset import VBDDataset
    test_set = VBDDataset(
        conf['data']['test_clean_dir'],
        conf['data']['test_noisy_dir'],
        sr=conf['data']['sample_rate']
    )
    model = DPRNNTasNet(
        **conf["filterbank"], **conf["masknet"], n_src=1
    )
    checkpoint = torch.load(os.environ['M'])
    model.load_state_dict(
        {k.split(".", 1)[1]: v for k, v in checkpoint["state_dict"].items()}
    )

    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model_device = next(model.parameters()).device

    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix[None, None])
        loss, reordered_sources = loss_func(est_sources, sources[None],
                                            return_est=True)
        mix_np = mix[None].cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        utt_metrics = get_metrics(mix_np, sources_np, est_sources_np,
                                  sample_rate=conf['sample_rate'],
                                  metrics_list=compute_metrics)
        utt_metrics['mix_path'] = test_set.mix[idx][0]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, 'ex_{}/'.format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np[0],
                     conf['sample_rate'])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx+1), src,
                         conf['sample_rate'])
            for src_idx, est_src in enumerate(est_sources_np):
                sf.write(local_save_dir + "s{}_estimate.wav".format(src_idx+1),
                         est_src, conf['sample_rate'])
            # Write local metrics to the example folder.
            with open(local_save_dir + 'metrics.json', 'w') as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf['exp_dir'], 'all_metrics.csv'))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = 'input_' + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + '_imp'] = ldf.mean()
    print('Overall metrics :')
    pprint(final_results)
    with open(os.path.join(conf['exp_dir'], 'final_metrics.json'), 'w') as f:
        json.dump(final_results, f, indent=0)
    model_dict = torch.load(model_path, map_location='cpu')

    publishable = save_publishable(
        os.path.join(conf['exp_dir'], 'publish_dir'), model_dict,
        metrics=final_results, train_conf=train_conf
    )

if __name__ == '__main__':
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
