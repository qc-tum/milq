from mqt.predictor import qcompile
from qiskit import QuantumCircuit


def predict_device(
    cicuit: QuantumCircuit, available_devices: list[str] | None = None
) -> str | None:
    """Predicts the device for a given circuit.

    Args:
        cicuit (QuantumCircuit): The circuit to predict the device for.
        available_devices (list[str] | None, optional): List of available devices.
            Defaults to None.

    Returns:
        str | None: The predicted device or None if no device is available
            or the predicted device isn't available
    """
    if available_devices is None:
        return None
    success = qcompile(cicuit)
    if not success:
        return None
    (_, _, device) = success
    if device in available_devices:
        return device
    return None
