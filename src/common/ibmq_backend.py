"""Backend wrapper."""
from enum import Enum

from qiskit.providers.fake_provider import FakeBelemV2, FakeNairobiV2, FakeQuitoV2


class IBMQBackend(Enum):
    """Wraps three common backends from IBMQ."""

    BELEM = FakeBelemV2()
    NAIROBI = FakeNairobiV2()
    QUITO = FakeQuitoV2()
