"""
Shared DAQ / F71 helpers for the calibration and validation GUIs.

Wraps:
  - CoilDriver: persistent 3-channel analog output task (Dev3/AO0-2).
  - F71Analog:  reads the F71 magnetometer analog output via DAQ AI (Dev2/ai0).
  - F71SCPI:    switches which axis the F71 routes to its analog output.
  - load_calibration() / voltage_for_field(): compute coil voltages for a
    desired field given a saved M (and optional ambient B0).
"""

import numpy as np
from pathlib import Path

try:
    import nidaqmx
    HAVE_NIDAQMX = True
except ImportError:
    HAVE_NIDAQMX = False

try:
    import pyvisa
    HAVE_PYVISA = True
except ImportError:
    HAVE_PYVISA = False


V_MIN, V_MAX = -10.0, 10.0

DEFAULT_COIL_DEVICE = "Dev3"
DEFAULT_COIL_CHANS = ("ao0", "ao1", "ao2")
DEFAULT_READOUT_DEVICE = "Dev2"
DEFAULT_READOUT_CHAN = "ai0"
DEFAULT_F71_RESOURCE = "ASRL18::INSTR"
DEFAULT_CORRECTION_V_PER_MT = 0.1

AXIS_SCPI = {"x": "XCOR", "y": "YCOR", "z": "ZCOR"}


class CoilDriver:
    """Persistent 3-channel AO task. Outputs stay latched between write()s."""

    def __init__(self, device: str = DEFAULT_COIL_DEVICE,
                 chans: tuple = DEFAULT_COIL_CHANS):
        self.device = device
        self.chans = chans
        self.task = None

    def open(self):
        if not HAVE_NIDAQMX:
            return
        self.task = nidaqmx.Task()
        for ch in self.chans:
            self.task.ao_channels.add_ao_voltage_chan(
                f"{self.device}/{ch}", min_val=V_MIN, max_val=V_MAX
            )
        self.task.write([0.0, 0.0, 0.0])

    def write(self, va: float, vb: float, vc: float):
        if self.task is not None:
            self.task.write([float(va), float(vb), float(vc)])

    def close(self):
        if self.task is not None:
            try:
                self.task.write([0.0, 0.0, 0.0])
            finally:
                self.task.close()
                self.task = None


class F71Analog:
    """Read the F71's analog-out via NI-DAQ AI. Returns the active axis in mT."""

    def __init__(self, device: str = DEFAULT_READOUT_DEVICE,
                 chan: str = DEFAULT_READOUT_CHAN,
                 correction_v_per_mT: float = DEFAULT_CORRECTION_V_PER_MT,
                 samples: int = 20, rate: int = 1000):
        self.device = device
        self.chan = chan
        self.k = float(correction_v_per_mT)
        self.samples = samples
        self.rate = rate

    def read_mT(self) -> float:
        if not HAVE_NIDAQMX:
            return 0.0
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.device}/{self.chan}")
            task.timing.cfg_samp_clk_timing(
                rate=self.rate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=self.samples,
            )
            data = task.read(number_of_samples_per_channel=self.samples)
            return float(np.mean(data)) / self.k


class F71SCPI:
    """Tell the F71 which axis to route to its analog output."""

    def __init__(self, resource: str = DEFAULT_F71_RESOURCE):
        self.resource = resource
        self.inst = None

    def open(self):
        if not HAVE_PYVISA:
            raise RuntimeError("pyvisa is not installed")
        rm = pyvisa.ResourceManager()
        inst = rm.open_resource(self.resource)
        inst.baud_rate = 115200
        inst.data_bits = 8
        inst.parity = pyvisa.constants.Parity.none
        inst.stop_bits = pyvisa.constants.StopBits.one
        inst.flow_control = pyvisa.constants.ControlFlow.rts_cts
        inst.timeout = 3000
        inst.write_termination = "\n"
        inst.read_termination = "\n"
        self.inst = inst

    def set_axis(self, axis: str):
        if self.inst is None:
            raise RuntimeError("SCPI not open")
        self.inst.write(f"SYST:ANAL:SIG {AXIS_SCPI[axis.lower()]}")

    def close(self):
        if self.inst is not None:
            try:
                self.inst.close()
            finally:
                self.inst = None


def load_calibration(path) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, M_inv). Raises on error."""
    M = np.load(path)
    if M.shape != (3, 3):
        raise ValueError(f"Calibration matrix must be 3x3, got {M.shape}")
    Minv = np.linalg.inv(M)
    return M, Minv


def load_b0(path):
    """Return B0 (3,) or None if missing/malformed."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        b0 = np.load(p)
    except Exception:
        return None
    if b0.shape != (3,):
        return None
    return np.asarray(b0, dtype=float)


def voltage_for_field(B_mT, Minv, B0=None) -> np.ndarray:
    """Compute coil voltages needed to produce B_mT (optionally subtracting B0)."""
    B = np.asarray(B_mT, dtype=float).reshape(3)
    if B0 is not None:
        B = B - np.asarray(B0, dtype=float).reshape(3)
    return Minv @ B
