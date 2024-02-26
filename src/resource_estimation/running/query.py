"""Interface to query the resource estimator for a given QIR program."""

from azure.quantum.qiskit.job import AzureQuantumJob
from azure.quantum.target.microsoft import MicrosoftEstimatorResult
from qiskit.tools.monitor import job_monitor

from .workspace import WorkspaceManager


def query_resource_estimator(bitcode: bytes, **kwargs) -> MicrosoftEstimatorResult:
    """Query the resource estimator given the QIR program.

    For details see https://learn.microsoft.com/en-us/azure/quantum/tutorial-resource-estimator-qir.

    Args:
        bitcode (bytes): The QIR program to estimate the runtime for.
        **kwargs: Additional parameters to pass to the resource estimator.
            Expects error_budget.

    Returns:
        MicrosoftEstimatorResult: The result of the resource estimation.
    """
    name = kwargs.get("name", "runtime-estimation-job")
    with WorkspaceManager() as manager:
        backend = manager.provider.get_backend("microsoft.estimator")
        config = backend.configuration()
        blob_name = config.azure["blob_name"]
        content_type = config.azure["content_type"]
        provider_id = config.azure["provider_id"]
        output_data_format = config.azure["output_data_format"]

        job = AzureQuantumJob(
            backend=backend,
            target=backend.name(),
            name=name,
            input_data=bitcode,
            blob_name=blob_name,
            content_type=content_type,
            provider_id=provider_id,
            input_data_format="qir.v1",
            output_data_format=output_data_format,
            input_params=kwargs,
            metadata={},
        )
    job_monitor(job)
    return job.result()
