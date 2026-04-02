from src.agent.ddpg import DDPG
from src.agent.her import HER
from src.agent.gcsl import GCSL
from src.agent.wgcsl import WGCSL
from src.agent.sac import SAC
from src.agent.sac_her import SAC_HER
from src.agent.mher import MHER
from src.agent.ppo_her import PPO_HER
from src.agent.td3_her import TD3_HER
from src.agent.gchr import GCHR
from src.agent.td_infonce import TDInfoNCE
from src.agent.qrl import QRL
from src.agent.contrastive_rl import ContrastiveRL   # <-- NEW
from src.agent.ris import RIS                         # <-- NEW

__all__ = [
    'DDPG',
    'HER',
    'GCSL',
    'WGCSL',
    'SAC',
    'SAC_HER',
    'MHER',
    'PPO_HER',
    'TD3_HER',
    'TDInfoNCE',
    'QRL',
    'GCHR',
    'ContrastiveRL',    # <-- NEW
    'RIS',              # <-- NEW
]
