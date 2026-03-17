import faulthandler, time
faulthandler.enable()
faulthandler.dump_traceback_later(5, repeat=True)
from data.generators import SyntheticConfig, generate_synthetic
from grrhs.models.gigg_regression import GIGGRegression
print('imports ok', flush=True)
cfg=SyntheticConfig(n=40,p=8,G=2,group_sizes=[4,4],signal={'blueprint':[{'groups':[0],'components':[{'distribution':'constant','count':2,'value':1.0,'sign':'positive'}]}]},noise_sigma=0.2,seed=10)
data=generate_synthetic(cfg)
print('data ok', flush=True)
model=GIGGRegression(method='fixed',n_burn_in=1,n_samples=1,n_thin=1,seed=0,store_lambda=True)
print('model ok', flush=True)
model.fit(data.X,data.y,groups=data.groups)
print('fit done', flush=True)
