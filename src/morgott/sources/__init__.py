from .boundary import _agentic_boundary_rows
from .finance import (
    _financebench_rows,
    _harper_valley_rows,
    _tatqa_rows,
)
from .security import (
    _browsesafe_rows,
    _gandalf_rows,
    _hackaprompt_rows,
    _llmail_rows,
    _tensor_trust_rows,
    _wildguard_rows,
    _wildjailbreak_rows,
)
from .tasks import (
    _banking77_rows,
    _coconot_rows,
    _false_reject_rows,
    _jbb_benign_rows,
    _lmsys_arena_rows,
    _massive_rows,
    _mind2web_rows,
    _schema_guided_dialogue_rows,
    _swebench_verified_rows,
    _taskmaster_rows,
)

__all__ = ["LOADERS"]

LOADERS = {
    "gandalf": _gandalf_rows,
    "llmail": _llmail_rows,
    "tensor_trust_raw": _tensor_trust_rows,
    "browsesafe": _browsesafe_rows,
    "hackaprompt": _hackaprompt_rows,
    "wildjailbreak": _wildjailbreak_rows,
    "wildguardmix": _wildguard_rows,
    "harper_valley_bank": _harper_valley_rows,
    "tatqa": _tatqa_rows,
    "financebench": _financebench_rows,
    "mind2web": _mind2web_rows,
    "swebench_verified": _swebench_verified_rows,
    "taskmaster": _taskmaster_rows,
    "banking77": _banking77_rows,
    "false_reject": _false_reject_rows,
    "schema_guided_dialogue": _schema_guided_dialogue_rows,
    "massive_en": _massive_rows,
    "coconot": _coconot_rows,
    "jbb_benign": _jbb_benign_rows,
    "lmsys_arena": _lmsys_arena_rows,
    "agentic_boundary_pairs": _agentic_boundary_rows,
}
