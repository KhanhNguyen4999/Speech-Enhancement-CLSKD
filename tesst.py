from asteroid.metrics import get_metrics
import pprint
import numpy as np

mix = np.random.randn(1, 16000)
clean = np.random.randn(2, 16000)
est = np.random.randn(2, 16000)
metrics_dict = get_metrics(mix, clean, est, sample_rate=8000,metrics_list='all')
pprint.pprint(metrics_dict)