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
from pathlib import Path
import torch
torch.backends.cudnn.enabled=False
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import DCCRNet_mini
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import WERTracker, MockWERTracker



COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi",'pesq']



def main():
    wer_tracker = (MockWERTracker())
    model_path = '/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD/checkpoint/the_best_model.pth'
    model = DCCRNet_mini.from_pretrained(model_path)
    # Handle device placement
    model_device = next(model.parameters()).device
    test_set = LibriMix(
        csv_dir='/root/NTH_student/Speech_Enhancement_new/asteroid/egs/librimix/DCCRNet/data/wav16k/max/test',
        task='enh_single',
        sample_rate=16000,
        n_src=1,
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
  
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        est_sources = model(mix.unsqueeze(0))
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=16000,
            # metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
        utt_metrics.update(
            **wer_tracker(
                mix=mix_np,
                clean=sources_np,
                estimate=est_sources_np_normalized,
                wav_id=ids,
                sample_rate=16000,
            )
        )
        series_list.append(pd.Series(utt_metrics))

        

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    #all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in COMPUTE_METRICS:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)

    with open('Speech_Enhancement_new/knowledge_distillation_CLSKD/results/All_metric.json','w') as fp:
        json.dump(final_results,fp) 


if __name__ == "__main__":
    main()
