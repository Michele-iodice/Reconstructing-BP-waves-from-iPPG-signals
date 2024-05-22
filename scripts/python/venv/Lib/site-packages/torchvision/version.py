__version__ = '0.18.0+cpu'
git_version = '6043bc250768b129e90a5321e318c1d51ee48a5c'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
