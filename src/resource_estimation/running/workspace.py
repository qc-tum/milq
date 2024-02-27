"""Qautnum workspace manager."""

from os import environ
import multiprocessing

from azure.identity import EnvironmentCredential
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider


class WorkspaceManager:
    """Singleton wrapper for the Azure Quantum workspace.

    Returns:
        Workspace: Azure Quantum workspace context manager.
    """

    _instance = None
    _workspace = None
    provider = None
    _lock = multiprocessing.Lock()

    def __new__(cls, *args, **kwargs) -> "WorkspaceManager":
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._workspace = _load_workspace()
                    cls.provider = AzureQuantumProvider(workspace=cls._workspace)
        return cls._instance

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


def _load_workspace() -> Workspace:
    resource_id = environ.get("AZURE_QUANTUM_WORKSPACE_RESOURCE_ID", "")
    location = environ.get("AZURE_QUANTUM_WORKSPACE_LOCATION", "")
    credential = EnvironmentCredential()
    return Workspace(
        resource_id=resource_id,
        location=location,
        credential=credential
    )
