"""Backend wrapper."""
from enum import Enum

from qiskit.providers.fake_provider import FakeBelemV2, FakeNairobiV2, FakeQuitoV2


class IBMQBackend(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    BELEM = FakeBelemV2
    NAIROBI = FakeNairobiV2
    QUITO = FakeQuitoV2
