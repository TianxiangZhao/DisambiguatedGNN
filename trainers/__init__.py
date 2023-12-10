from .DGI_trainer import Trainer_DGI
from .GRACE_trainer import Trainer_GRACE
from .MVGRL_trainer import Trainer_MVGRL
from .SUP_trainer import Trainer_SUP
from .SupBalance_trainer import Trainer_BALANCE
from .BoundCont_trainer import Trainer_BoundCont

__all__ = {
    'Trainer_DGI',
    'Trainer_GRACE',
    'Trainer_MVGRL',
    'Trainer_SUP',
    'Trainer_BALANCE',
    'Trainer_BoundCont'
}