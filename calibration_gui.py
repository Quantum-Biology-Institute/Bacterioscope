"""
Magnetic field calibration GUI.

Drives a set of test voltage vectors on the 3-axis coils, reads the F71
magnetometer per-axis, and fits a 3x3 calibration matrix M such that

    B_mT = M @ V_volts

Two readout modes:
  - SCPI (automatic): F71 axis is switched via pyvisa between reads.
  - Manual: operator switches the F71 display manually; GUI runs 3 passes.

Saves calibration_M.npy. Existing file is backed up with a timestamp.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import queue
import time
import shutil
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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


DEFAULT_COIL_DEVICE = "Dev3"
DEFAULT_COIL_CHANS = ("ao0", "ao1", "ao2")
DEFAULT_READOUT_DEVICE = "Dev2"
DEFAULT_READOUT_CHAN = "ai0"
DEFAULT_F71_RESOURCE = "ASRL18::INSTR"
DEFAULT_CORRECTION_V_PER_MT = 0.1

AXIS_SCPI = {"x": "XCOR", "y": "YCOR", "z": "ZCOR"}

V_MIN, V_MAX = -10.0, 10.0
DAQ_SAMPLE_RATE = 1000
DAQ_SAMPLES = 50
COIL_SETTLE_S = 1.0
SCPI_SETTLE_S = 0.6

CAL_PATH = "calibration_M.npy"
B0_PATH = "calibration_B0.npy"


# =========================
#  VOLTAGE PROTOCOL
# =========================

def build_protocol(max_v: float) -> list[tuple[float, float, float]]:
    """Return test voltage vectors. Includes zero, single-axis sweeps at
    ±max and ±max/2, and two-axis combinations at ±max/2."""
    s = float(max_v)
    h = s / 2.0
    vecs: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for ax in range(3):
        for v in (s, -s, h, -h):
            vec = [0.0, 0.0, 0.0]
            vec[ax] = v
            vecs.append(tuple(vec))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for a, b in [(h, h), (h, -h), (-h, h), (-h, -h)]:
            vec = [0.0, 0.0, 0.0]
            vec[i] = a
            vec[j] = b
            vecs.append(tuple(vec))
    return vecs


# =========================
#  HARDWARE WRAPPERS
# =========================

class CoilDriver:
    def __init__(self, device: str, chans: tuple):
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
    def __init__(self, device: str, chan: str, correction_v_per_mT: float):
        self.device = device
        self.chan = chan
        self.k = float(correction_v_per_mT)

    def read_mT(self) -> float:
        if not HAVE_NIDAQMX:
            return 0.0
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.device}/{self.chan}")
            task.timing.cfg_samp_clk_timing(
                rate=DAQ_SAMPLE_RATE,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=DAQ_SAMPLES,
            )
            data = task.read(number_of_samples_per_channel=DAQ_SAMPLES)
            return float(np.mean(data)) / self.k


class F71SCPI:
    def __init__(self, resource: str):
        self.resource = resource
        self.inst = None

    def open(self):
        if not HAVE_PYVISA:
            raise RuntimeError("pyvisa not installed")
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


# =========================
#  FIT
# =========================

def fit_M(V: np.ndarray, B: np.ndarray):
    """Solve B = V @ M.T in least squares. Returns M (3,3) and fit quality."""
    M_T, *_ = np.linalg.lstsq(V, B, rcond=None)
    M = M_T.T
    B_pred = V @ M.T
    residuals = B - B_pred
    rmse_per_axis = np.sqrt(np.mean(residuals ** 2, axis=0))
    return M, B_pred, residuals, rmse_per_axis


def max_field_per_axis(M: np.ndarray, v_max: float) -> dict:
    """For each world axis ê, compute max |B| before any coil hits v_max."""
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    out = {}
    for name, e in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        V_req_per_mT = Minv @ np.array(e, dtype=float)
        peak = float(np.max(np.abs(V_req_per_mT)))
        out[name] = (v_max / peak) if peak > 0 else 0.0
    return out


# =========================
#  WORKER
# =========================

class CalibrationWorker(threading.Thread):
    """Run a calibration sequence in a background thread. Posts events to a
    queue for the GUI to consume on the Tk main loop."""

    def __init__(
        self,
        voltages: list,
        coil: CoilDriver,
        reader: F71Analog,
        scpi: F71SCPI | None,
        mode: str,
        events: queue.Queue,
        abort: threading.Event,
    ):
        super().__init__(daemon=True)
        self.voltages = voltages
        self.coil = coil
        self.reader = reader
        self.scpi = scpi
        self.mode = mode  # "scpi" or "manual"
        self.events = events
        self.abort = abort

    def _post(self, kind: str, **data):
        self.events.put({"kind": kind, **data})

    def run(self):
        try:
            N = len(self.voltages)
            B = np.full((N, 3), np.nan, dtype=float)

            if self.mode == "scpi":
                for idx, vec in enumerate(self.voltages):
                    if self.abort.is_set():
                        self._post("aborted")
                        return
                    self.coil.write(*vec)
                    time.sleep(COIL_SETTLE_S)
                    for ax_i, ax in enumerate(("x", "y", "z")):
                        self.scpi.set_axis(ax)
                        time.sleep(SCPI_SETTLE_S)
                        B[idx, ax_i] = self.reader.read_mT()
                    self._post("sample", idx=idx, V=vec, B=B[idx].tolist())
            else:
                for ax_i, ax in enumerate(("x", "y", "z")):
                    if self.abort.is_set():
                        self._post("aborted")
                        return
                    self._post("prompt_axis", axis=ax)
                    while not self.abort.is_set():
                        if self._wait_for_ack(timeout=0.25):
                            break
                    if self.abort.is_set():
                        self._post("aborted")
                        return
                    for idx, vec in enumerate(self.voltages):
                        if self.abort.is_set():
                            self._post("aborted")
                            return
                        self.coil.write(*vec)
                        time.sleep(COIL_SETTLE_S)
                        B[idx, ax_i] = self.reader.read_mT()
                        self._post(
                            "sample",
                            idx=idx,
                            V=vec,
                            B=B[idx].tolist(),
                            axis=ax,
                        )
            self.coil.write(0.0, 0.0, 0.0)
            self._post("done", B=B.tolist(), V=[list(v) for v in self.voltages])
        except Exception as e:
            try:
                self.coil.write(0.0, 0.0, 0.0)
            except Exception:
                pass
            self._post("error", message=str(e))

    # Manual-mode acknowledgement: the GUI sets this event when user clicks
    # "Continue" after switching the F71 axis.
    _ack: threading.Event = None  # set from outside

    def bind_ack(self, ack: threading.Event):
        self._ack = ack

    def _wait_for_ack(self, timeout: float) -> bool:
        if self._ack is None:
            return True
        if self._ack.wait(timeout):
            self._ack.clear()
            return True
        return False


# =========================
#  GUI
# =========================

class CalibrationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Magnet Calibration")
        self.root.geometry("1200x760")

        self.events: queue.Queue = queue.Queue()
        self.abort = threading.Event()
        self.ack = threading.Event()
        self.worker: CalibrationWorker | None = None

        self.V_recorded: np.ndarray | None = None
        self.B_recorded: np.ndarray | None = None
        self.M_fit: np.ndarray | None = None
        self.B0_fit: np.ndarray | None = None

        self._build_ui()
        self.root.after(100, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_controls(left)
        self._build_plots(right)

    def _build_controls(self, parent):
        # Hardware config
        hw = ttk.LabelFrame(parent, text="Hardware", padding=8)
        hw.pack(fill="x", pady=(0, 6))

        self.var_coil_dev = tk.StringVar(value=DEFAULT_COIL_DEVICE)
        self.var_ai_dev = tk.StringVar(value=DEFAULT_READOUT_DEVICE)
        self.var_ai_chan = tk.StringVar(value=DEFAULT_READOUT_CHAN)
        self.var_k = tk.DoubleVar(value=DEFAULT_CORRECTION_V_PER_MT)
        self.var_visa = tk.StringVar(value=DEFAULT_F71_RESOURCE)
        self.var_mode = tk.StringVar(value="scpi" if HAVE_PYVISA else "manual")

        grid = ttk.Frame(hw)
        grid.pack(fill="x")
        for col in range(2):
            grid.columnconfigure(col, weight=1)

        def row(r, label, widget):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="ew", pady=2)

        row(0, "Coil device", ttk.Entry(grid, textvariable=self.var_coil_dev, width=14))
        row(1, "F71 DAQ device", ttk.Entry(grid, textvariable=self.var_ai_dev, width=14))
        row(2, "F71 DAQ channel", ttk.Entry(grid, textvariable=self.var_ai_chan, width=14))
        row(3, "F71 analog scale (V/mT)", ttk.Entry(grid, textvariable=self.var_k, width=14))
        row(4, "F71 VISA resource", ttk.Entry(grid, textvariable=self.var_visa, width=14))

        ttk.Label(grid, text="Readout mode").grid(row=5, column=0, sticky="w", pady=2)
        mode_row = ttk.Frame(grid)
        mode_row.grid(row=5, column=1, sticky="ew")
        ttk.Radiobutton(mode_row, text="SCPI (auto)", variable=self.var_mode,
                        value="scpi").pack(side="left")
        ttk.Radiobutton(mode_row, text="Manual", variable=self.var_mode,
                        value="manual").pack(side="left", padx=(8, 0))

        # Protocol
        proto = ttk.LabelFrame(parent, text="Test protocol", padding=8)
        proto.pack(fill="x", pady=6)
        self.var_max_v = tk.DoubleVar(value=2.0)
        r = ttk.Frame(proto)
        r.pack(fill="x")
        ttk.Label(r, text="Max |V| per coil:").pack(side="left")
        ttk.Entry(r, width=6, textvariable=self.var_max_v, justify="right").pack(side="left", padx=4)
        ttk.Label(r, text="V").pack(side="left")
        ttk.Button(r, text="Preview", command=self._preview_protocol).pack(side="right")
        self.lbl_proto = ttk.Label(proto, text="Click Preview to count vectors.")
        self.lbl_proto.pack(anchor="w", pady=(4, 0))

        # Run
        run = ttk.LabelFrame(parent, text="Run", padding=8)
        run.pack(fill="x", pady=6)
        self.btn_start = ttk.Button(run, text="Start Calibration", command=self._start)
        self.btn_start.pack(fill="x")
        self.btn_abort = ttk.Button(run, text="Abort", command=self._abort, state="disabled")
        self.btn_abort.pack(fill="x", pady=(4, 0))
        self.btn_continue = ttk.Button(run, text="Axis switched — continue",
                                       command=self._ack_axis, state="disabled")
        self.btn_continue.pack(fill="x", pady=(4, 0))
        self.progress = ttk.Progressbar(run, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))

        # Fit / save
        out = ttk.LabelFrame(parent, text="Fit & save", padding=8)
        out.pack(fill="x", pady=6)
        self.btn_fit = ttk.Button(out, text="Fit matrix from measurements",
                                  command=self._fit, state="disabled")
        self.btn_fit.pack(fill="x")
        self.btn_save = ttk.Button(out, text=f"Save as {CAL_PATH}",
                                   command=self._save, state="disabled")
        self.btn_save.pack(fill="x", pady=(4, 0))
        self.btn_save_as = ttk.Button(out, text="Save as...",
                                      command=self._save_as, state="disabled")
        self.btn_save_as.pack(fill="x", pady=(4, 0))
        ttk.Button(out, text="Zero coils", command=self._zero_coils).pack(fill="x", pady=(6, 0))

        self.txt = tk.Text(parent, height=14, width=48, wrap="word")
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

    def _build_plots(self, parent):
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.ax_bx = self.fig.add_subplot(311)
        self.ax_by = self.fig.add_subplot(312)
        self.ax_bz = self.fig.add_subplot(313)
        for ax, lbl in ((self.ax_bx, "Bx"), (self.ax_by, "By"), (self.ax_bz, "Bz")):
            ax.set_ylabel(f"{lbl} (mT)")
            ax.grid(True, alpha=0.3)
        self.ax_bz.set_xlabel("sample index")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ---------- log ----------

    def _log(self, msg: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    # ---------- protocol ----------

    def _current_protocol(self):
        return build_protocol(float(self.var_max_v.get()))

    def _preview_protocol(self):
        try:
            vecs = self._current_protocol()
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Invalid", str(e))
            return
        self.lbl_proto.config(text=f"{len(vecs)} vectors, |V| ≤ {self.var_max_v.get():.2f} V")

    # ---------- run ----------

    def _start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        if not HAVE_NIDAQMX:
            if not messagebox.askyesno(
                "No DAQ", "nidaqmx is not installed. Run in simulation mode?"
            ):
                return

        try:
            max_v = float(self.var_max_v.get())
            if not (0.1 <= max_v <= V_MAX):
                raise ValueError(f"Max |V| must be in [0.1, {V_MAX}] V")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Invalid", str(e))
            return

        vecs = self._current_protocol()
        if any(abs(v) > V_MAX for vec in vecs for v in vec):
            messagebox.showerror("Out of range",
                                 f"Protocol requires |V| > {V_MAX}")
            return

        mode = self.var_mode.get()
        if mode == "scpi" and not HAVE_PYVISA:
            messagebox.showerror(
                "pyvisa missing",
                "pyvisa not installed. Switch to Manual mode.",
            )
            return

        coil = CoilDriver(self.var_coil_dev.get(), DEFAULT_COIL_CHANS)
        try:
            coil.open()
        except Exception as e:
            messagebox.showerror("DAQ open failed", f"Coil driver: {e}")
            return

        reader = F71Analog(self.var_ai_dev.get(), self.var_ai_chan.get(),
                           float(self.var_k.get()))
        scpi = None
        if mode == "scpi":
            scpi = F71SCPI(self.var_visa.get())
            try:
                scpi.open()
            except Exception as e:
                coil.close()
                messagebox.showerror("SCPI open failed", str(e))
                return

        self._coil = coil
        self._scpi = scpi

        self.abort.clear()
        self.ack.clear()
        self.progress["maximum"] = len(vecs) * (3 if mode == "manual" else 1)
        self.progress["value"] = 0

        self._n_vecs = len(vecs)
        self._samples_seen = 0
        self._B_running = np.full((len(vecs), 3), np.nan, dtype=float)
        self._V_running = np.array(vecs, dtype=float)

        self._log(f"\n=== Starting calibration ({mode} mode, {len(vecs)} vectors) ===")
        self._log(f"Coil: {self.var_coil_dev.get()}/[AO0-2], "
                  f"Readout: {self.var_ai_dev.get()}/{self.var_ai_chan.get()}, "
                  f"k = {self.var_k.get():.3f} V/mT")

        self.worker = CalibrationWorker(
            voltages=vecs,
            coil=coil,
            reader=reader,
            scpi=scpi,
            mode=mode,
            events=self.events,
            abort=self.abort,
        )
        self.worker.bind_ack(self.ack)
        self.worker.start()

        self.btn_start.config(state="disabled")
        self.btn_abort.config(state="normal")
        self.btn_fit.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.btn_save_as.config(state="disabled")

    def _abort(self):
        if self.worker is not None:
            self.abort.set()
            self.ack.set()
            self._log("Abort requested.")

    def _ack_axis(self):
        self.ack.set()
        self.btn_continue.config(state="disabled")

    # ---------- events ----------

    def _drain_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                self._handle_event(ev)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _handle_event(self, ev):
        kind = ev["kind"]
        if kind == "sample":
            idx = ev["idx"]
            if "axis" in ev:
                ax_i = "xyz".index(ev["axis"])
                self._B_running[idx, ax_i] = ev["B"][ax_i]
            else:
                self._B_running[idx] = ev["B"]
            self._samples_seen += 1
            self.progress["value"] = self._samples_seen
            V = ev["V"]
            B = ev["B"]
            axis_note = f" [{ev['axis']}]" if "axis" in ev else ""
            self._log(f"  [{idx+1:02d}/{self._n_vecs}] V={V} → "
                      f"B=({B[0]:+.3f}, {B[1]:+.3f}, {B[2]:+.3f}) mT{axis_note}")
            self._redraw_measured()
        elif kind == "prompt_axis":
            ax = ev["axis"].upper()
            self._log(f"→ Switch F71 analog output to {ax}-axis, then click 'Axis switched — continue'.")
            self.btn_continue.config(state="normal")
        elif kind == "done":
            self.V_recorded = np.array(ev["V"], dtype=float)
            self.B_recorded = np.array(ev["B"], dtype=float)
            self._log("Measurement complete.")
            self._finish_run(success=True)
            self._fit()  # auto-fit after a complete run
        elif kind == "aborted":
            self._log("Aborted.")
            self._finish_run(success=False)
        elif kind == "error":
            self._log(f"ERROR: {ev['message']}")
            messagebox.showerror("Calibration error", ev["message"])
            self._finish_run(success=False)

    def _finish_run(self, success: bool):
        try:
            if hasattr(self, "_coil") and self._coil is not None:
                self._coil.close()
        finally:
            self._coil = None
        if hasattr(self, "_scpi") and self._scpi is not None:
            try:
                self._scpi.close()
            finally:
                self._scpi = None
        self.btn_start.config(state="normal")
        self.btn_abort.config(state="disabled")
        self.btn_continue.config(state="disabled")
        if success and self.B_recorded is not None:
            self.btn_fit.config(state="normal")

    # ---------- fit ----------

    def _fit(self):
        if self.V_recorded is None or self.B_recorded is None:
            return
        V = self.V_recorded
        B = self.B_recorded
        if np.isnan(B).any():
            messagebox.showerror("Incomplete data", "Some measurements are NaN; cannot fit.")
            return

        zero_rows = np.all(V == 0.0, axis=1)
        B0 = B[zero_rows].mean(axis=0) if zero_rows.any() else np.zeros(3)
        B_centered = B - B0

        M, B_pred, resid, rmse = fit_M(V[~zero_rows], B_centered[~zero_rows])
        cond = float(np.linalg.cond(M))
        col_norms = np.linalg.norm(M, axis=0)
        maxB = max_field_per_axis(M, V_MAX - 0.02)

        self.M_fit = M
        self.B0_fit = B0

        self._log("\n--- Fit results ---")
        self._log(f"Ambient B0 (mT):  ({B0[0]:+.3f}, {B0[1]:+.3f}, {B0[2]:+.3f})")
        self._log("Calibration matrix M (mT/V):")
        for i in range(3):
            self._log(f"  [{M[i,0]:+7.3f}, {M[i,1]:+7.3f}, {M[i,2]:+7.3f}]")
        self._log(f"Column norms (per-coil gain): "
                  f"A={col_norms[0]:.2f}, B={col_norms[1]:.2f}, C={col_norms[2]:.2f} mT/V")
        self._log(f"RMSE per axis (mT): ({rmse[0]:.3f}, {rmse[1]:.3f}, {rmse[2]:.3f})")
        self._log(f"Condition number:   {cond:.2f}  "
                  f"({'OK' if cond < 20 else 'HIGH — check for weak/colinear coils'})")
        self._log(f"Max |B| achievable at ±{V_MAX:.1f} V: "
                  f"X ≤ {maxB['x']:.1f}, Y ≤ {maxB['y']:.1f}, Z ≤ {maxB['z']:.1f} mT")
        self._log("Suggested experimental range: stay within ~70% of those maxima for headroom.")

        self._redraw_fit(V[~zero_rows], B_centered[~zero_rows], B_pred)
        self.btn_save.config(state="normal")
        self.btn_save_as.config(state="normal")

    # ---------- plots ----------

    def _redraw_measured(self):
        if self.B_recorded is None:
            B = self._B_running
        else:
            B = self.B_recorded
        idx = np.arange(B.shape[0])
        for ax, col in ((self.ax_bx, 0), (self.ax_by, 1), (self.ax_bz, 2)):
            ax.clear()
            ax.plot(idx, B[:, col], "o-", label="measured")
            ax.set_ylabel(f"B{'xyz'[col]} (mT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        self.ax_bz.set_xlabel("sample index")
        self.canvas.draw_idle()

    def _redraw_fit(self, V, B_meas, B_pred):
        idx = np.arange(B_meas.shape[0])
        for ax, col in ((self.ax_bx, 0), (self.ax_by, 1), (self.ax_bz, 2)):
            ax.clear()
            ax.plot(idx, B_meas[:, col], "o-", label="measured")
            ax.plot(idx, B_pred[:, col], "x--", label="predicted")
            ax.set_ylabel(f"B{'xyz'[col]} (mT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        self.ax_bz.set_xlabel("sample index (non-zero protocol rows)")
        self.canvas.draw_idle()

    # ---------- save ----------

    def _save(self):
        self._save_to(Path(CAL_PATH))

    def _save_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy")],
            initialfile=CAL_PATH,
        )
        if path:
            self._save_to(Path(path))

    def _save_to(self, path: Path):
        if self.M_fit is None:
            return
        if path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_name(f"{path.stem}.backup_{ts}.npy")
            shutil.copy(path, backup)
            self._log(f"Backed up existing {path.name} → {backup.name}")
        np.save(path, self.M_fit)
        if self.B0_fit is not None:
            np.save(path.with_name(B0_PATH), self.B0_fit)
        self._log(f"Saved M → {path.name}  and B0 → {B0_PATH}")
        messagebox.showinfo("Saved", f"Calibration written to {path.name}")

    # ---------- misc ----------

    def _zero_coils(self):
        if not HAVE_NIDAQMX:
            return
        try:
            c = CoilDriver(self.var_coil_dev.get(), DEFAULT_COIL_CHANS)
            c.open()
            c.write(0.0, 0.0, 0.0)
            c.close()
            self._log("Coils zeroed.")
        except Exception as e:
            messagebox.showerror("Zero failed", str(e))

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.abort.set()
            self.ack.set()
            self.worker.join(timeout=2.0)
        try:
            if hasattr(self, "_coil") and self._coil is not None:
                self._coil.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_scpi") and self._scpi is not None:
                self._scpi.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CalibrationGUI(root)
    root.mainloop()
