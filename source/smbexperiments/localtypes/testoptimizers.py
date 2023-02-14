from enum import Enum
from typing import Dict

from sls.sls import Sls
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from smbsolve.smb import SMB


class TestOptimizer(Enum):
    SMB = SMB
    SLS = Sls
    ADAM = Adam
    SGD = SGD

    def config_from_dict(self, params, data: Dict):
        return self.value(params, **data)
