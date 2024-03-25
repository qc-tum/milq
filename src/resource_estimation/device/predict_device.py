from mqt.predictor import qcompile
from qiskit import QuantumCircuit


def predict_device(
    cicuit: QuantumCircuit, available_devices: list[str] | None = None
) -> str | None:
    """Predicts the device for a given circuit.
    
    FIXME: Doesn't work with single-qubit circuits. This if fixed with mqt-bench 
    v1.1.0 which is however incompatible with the most recent mqt-predictor (v2.0.0).

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
