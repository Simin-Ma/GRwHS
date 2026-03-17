import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(5, repeat=True)
print('before', flush=True)
import sklearn
print('after', flush=True)
