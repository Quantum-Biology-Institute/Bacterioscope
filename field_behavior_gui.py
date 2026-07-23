"""
Dynamic field behavior validator.

Drives a pre-defined behavior through the 3-axis coils while reading the F71
magnetometer axis-by-axis (via SCPI switching), then plots measured vs
predicted field and reports RMSE.

Behaviors
---------
  - Rotating field in a chosen plane (XY / YZ / XZ) at a given frequency and
    amplitude for a chosen duration.
  - Linear ramp along a single axis from start to end values over a duration.

Timing model
------------
The loop writes the commanded voltage, waits until the next scheduled target
time, then reads the F71. Total sample count is N = update_rate * duration.
Frequency is capped at update_rate / MIN_SAMPLES_PER_CYCLE so there are
enough points per cycle for the measurement to be meaningful.

Three passes per run (one per F71 axis). Between passes, the F71 analog
output is switched via SCPI.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import threading
import queue
import time

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from hardware_io import (
    CoilDriver, F71Analog, F71SCPI,
    V_MIN, V_MAX,
    DEFAULT_COIL_DEVICE, DEFAULT_READOUT_DEVICE, DEFAULT_READOUT_CHAN,
    DEFAULT_F71_RESOURCE, DEFAULT_CORRECTION_V_PER_MT,
    HAVE_NIDAQMX, HAVE_PYVISA,
    load_calibration, load_b0,
)


DEFAULT_UPDATE_RATE_HZ = 20.0
MAX_UPDATE_RATE_HZ = 40.0
MIN_UPDATE_RATE_HZ = 5.0
MIN_SAMPLES_PER_CYCLE = 6

AI_SAMPLES_PER_READ = 10
AI_RATE_HZ = 1000


# =========================
#  WAVEFORM BUILDERS
# =========================

def build_rotation(plane: str, freq_hz: float, amp_mT: float,
                   duration_s: float, update_rate_hz: float):
    N = max(2, int(round(duration_s * update_rate_hz)))
    t = np.linspace(0, duration_s, N, endpoint=False)
    omega = 2 * np.pi * freq_hz
    c, s = np.cos(omega * t), np.sin(omega * t)
    B = np.zeros((N, 3))
    idx = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}[plane.lower()]
    B[:, idx[0]] = amp_mT * c
    B[:, idx[1]] = amp_mT * s
    return t, B


def build_ramp(axis: str, start_mT: float, end_mT: float,
               duration_s: float, update_rate_hz: float):
    N = max(2, int(round(duration_s * update_rate_hz)))
    t = np.linspace(0, duration_s, N)
    B = np.zeros((N, 3))
    B[:, "xyz".index(axis.lower())] = np.linspace(start_mT, end_mT, N)
    return t, B


# =========================
#  WORKER
# =========================

class BehaviorWorker(threading.Thread):
    def __init__(self, t, B_pred, Minv, B0_sub, coil, reader, scpi,
                 update_rate_hz, events, abort, mode="scpi", ack=None):
        super().__init__(daemon=True)
        self.t = t
        self.B_pred = B_pred
        self.Minv = Minv
        self.B0_sub = B0_sub
        self.coil = coil
        self.reader = reader
        self.scpi = scpi
        self.period_s = 1.0 / float(update_rate_hz)
        self.events = events
        self.abort = abort
        self.mode = mode  # "scpi" or "manual"
        self.ack = ack    # threading.Event for manual acknowledgement

    def _post(self, kind, **data):
        self.events.put({"kind": kind, **data})

    def _wait_for_ack(self):
        while not self.abort.is_set():
            if self.ack.wait(timeout=0.2):
                self.ack.clear()
                return True
        return False

    def run(self):
        try:
            N = self.B_pred.shape[0]
            B_meas = np.full((N, 3), np.nan, dtype=float)

            B_cmd = self.B_pred.copy()
            if self.B0_sub is not None:
                B_cmd = B_cmd - self.B0_sub
            V_wave = (self.Minv @ B_cmd.T).T  # (N, 3)

            if np.any(np.abs(V_wave) > V_MAX):
                peak = float(np.max(np.abs(V_wave)))
                self._post("error",
                           message=f"Behavior requires |V| up to {peak:.2f} V "
                                   f"(max ±{V_MAX} V). Reduce amplitude.")
                return

            for ax_i, axis in enumerate(("x", "y", "z")):
                if self.abort.is_set():
                    self._zero_quiet()
                    self._post("aborted")
                    return
                if self.mode == "scpi":
                    self.scpi.set_axis(axis)
                    time.sleep(0.6)
                else:
                    self._post("prompt_axis", axis=axis)
                    if not self._wait_for_ack():
                        self._zero_quiet()
                        self._post("aborted")
                        return
                self._post("pass_start", axis=axis, N=N)

                t0 = time.monotonic()
                for i in range(N):
                    if self.abort.is_set():
                        self._zero_quiet()
                        self._post("aborted")
                        return
                    self.coil.write(*V_wave[i])
                    target = t0 + (i + 1) * self.period_s
                    now = time.monotonic()
                    if target > now:
                        time.sleep(target - now)
                    B_meas[i, ax_i] = self.reader.read_mT()
                    if (i % max(1, N // 40)) == 0 or i == N - 1:
                        self._post("progress",
                                   axis=axis, i=i, N=N,
                                   B_meas_axis=B_meas[:, ax_i].tolist())
                self._post("pass_done", axis=axis,
                           B_meas_axis=B_meas[:, ax_i].tolist())

            self._zero_quiet()
            self._post("done", B_meas=B_meas.tolist())
        except Exception as e:
            self._zero_quiet()
            self._post("error", message=str(e))

    def _zero_quiet(self):
        try:
            self.coil.write(0.0, 0.0, 0.0)
        except Exception:
            pass


# =========================
#  GUI
# =========================

class FieldBehaviorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Field behavior validator")
        self.root.geometry("1200x780")

        self.M = None
        self.Minv = None
        self.B0 = None

        self.events: queue.Queue = queue.Queue()
        self.abort = threading.Event()
        self.ack = threading.Event()
        self.worker: BehaviorWorker | None = None

        self._t = None
        self._B_pred = None
        self._B_meas = None
        self._coil = None
        self._scpi = None

        self._build_ui()
        self.root.after(80, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_file(Path("calibration_M.npy"))

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
        cal = ttk.LabelFrame(parent, text="Calibration", padding=8)
        cal.pack(fill="x")
        r = ttk.Frame(cal); r.pack(fill="x")
        self.var_path = tk.StringVar(value="calibration_M.npy")
        ttk.Entry(r, textvariable=self.var_path, width=24).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="Browse...", command=self._browse).pack(side="left", padx=4)
        ttk.Button(r, text="Reload",
                   command=lambda: self._load_file(Path(self.var_path.get()))).pack(side="left")
        self.var_sub_b0 = tk.BooleanVar(value=False)
        ttk.Checkbutton(cal, text="Subtract ambient B0", variable=self.var_sub_b0
                        ).pack(anchor="w", pady=(4, 0))
        self.lbl_cal = ttk.Label(cal, text="(not loaded)", foreground="#555")
        self.lbl_cal.pack(anchor="w")

        hw = ttk.LabelFrame(parent, text="Hardware", padding=8)
        hw.pack(fill="x", pady=6)
        self.var_coil = tk.StringVar(value=DEFAULT_COIL_DEVICE)
        self.var_ai_dev = tk.StringVar(value=DEFAULT_READOUT_DEVICE)
        self.var_ai_chan = tk.StringVar(value=DEFAULT_READOUT_CHAN)
        self.var_k = tk.DoubleVar(value=DEFAULT_CORRECTION_V_PER_MT)
        self.var_visa = tk.StringVar(value=DEFAULT_F71_RESOURCE)
        grid = ttk.Frame(hw); grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        def row(r, label, w):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=1)
            w.grid(row=r, column=1, sticky="ew", pady=1)
        row(0, "Coil device", ttk.Entry(grid, textvariable=self.var_coil))
        row(1, "F71 AI device", ttk.Entry(grid, textvariable=self.var_ai_dev))
        row(2, "F71 AI channel", ttk.Entry(grid, textvariable=self.var_ai_chan))
        row(3, "F71 scale (V/mT)", ttk.Entry(grid, textvariable=self.var_k))
        row(4, "F71 VISA", ttk.Entry(grid, textvariable=self.var_visa))

        self.var_mode = tk.StringVar(value="manual")
        mode_row = ttk.Frame(hw); mode_row.pack(fill="x", pady=(6, 0))
        ttk.Label(mode_row, text="Axis switching:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Manual", variable=self.var_mode,
                        value="manual").pack(side="left", padx=(4, 0))
        ttk.Radiobutton(mode_row, text="SCPI", variable=self.var_mode,
                        value="scpi").pack(side="left", padx=(8, 0))

        beh = ttk.LabelFrame(parent, text="Behavior", padding=8)
        beh.pack(fill="x", pady=6)
        self.var_kind = tk.StringVar(value="rotation")
        ttk.Radiobutton(beh, text="Rotating field",
                        variable=self.var_kind, value="rotation",
                        command=self._refresh_behavior).pack(anchor="w")
        ttk.Radiobutton(beh, text="Linear ramp",
                        variable=self.var_kind, value="ramp",
                        command=self._refresh_behavior).pack(anchor="w")

        self.rot_frame = ttk.Frame(beh)
        self.ramp_frame = ttk.Frame(beh)

        self.var_plane = tk.StringVar(value="xy")
        self.var_freq = tk.DoubleVar(value=0.5)
        self.var_amp = tk.DoubleVar(value=5.0)
        self.var_dur_rot = tk.DoubleVar(value=6.0)
        r = ttk.Frame(self.rot_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Plane").pack(side="left")
        ttk.Combobox(r, textvariable=self.var_plane, values=["xy", "yz", "xz"],
                     width=6, state="readonly").pack(side="left", padx=4)
        r = ttk.Frame(self.rot_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Frequency (Hz)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_freq, width=8, justify="right").pack(side="left", padx=4)
        r = ttk.Frame(self.rot_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Amplitude (mT)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_amp, width=8, justify="right").pack(side="left", padx=4)
        r = ttk.Frame(self.rot_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Duration (s)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_dur_rot, width=8, justify="right").pack(side="left", padx=4)

        self.var_axis = tk.StringVar(value="x")
        self.var_start = tk.DoubleVar(value=0.0)
        self.var_end = tk.DoubleVar(value=10.0)
        self.var_dur_rmp = tk.DoubleVar(value=5.0)
        r = ttk.Frame(self.ramp_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Axis").pack(side="left")
        ttk.Combobox(r, textvariable=self.var_axis, values=["x", "y", "z"],
                     width=4, state="readonly").pack(side="left", padx=4)
        r = ttk.Frame(self.ramp_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Start (mT)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_start, width=8, justify="right").pack(side="left", padx=4)
        r = ttk.Frame(self.ramp_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="End (mT)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_end, width=8, justify="right").pack(side="left", padx=4)
        r = ttk.Frame(self.ramp_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Duration (s)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_dur_rmp, width=8, justify="right").pack(side="left", padx=4)

        self._refresh_behavior()

        timing = ttk.LabelFrame(parent, text="Timing", padding=8)
        timing.pack(fill="x", pady=6)
        self.var_rate = tk.DoubleVar(value=DEFAULT_UPDATE_RATE_HZ)
        r = ttk.Frame(timing); r.pack(fill="x")
        ttk.Label(r, text="Update rate (Hz)").pack(side="left")
        ttk.Entry(r, textvariable=self.var_rate, width=8, justify="right").pack(side="left", padx=4)
        ttk.Label(timing,
                  text=(f"Range: {MIN_UPDATE_RATE_HZ:.0f}–{MAX_UPDATE_RATE_HZ:.0f} Hz.  "
                        f"Rotation freq ≤ rate / {MIN_SAMPLES_PER_CYCLE}."),
                  foreground="#666").pack(anchor="w", pady=(4, 0))

        run = ttk.LabelFrame(parent, text="Run", padding=8)
        run.pack(fill="x", pady=6)
        self.btn_start = ttk.Button(run, text="Start", command=self._start)
        self.btn_start.pack(fill="x")
        self.btn_abort = ttk.Button(run, text="Abort", command=self._abort, state="disabled")
        self.btn_abort.pack(fill="x", pady=(4, 0))
        self.btn_continue = ttk.Button(run, text="Axis switched — continue",
                                       command=self._ack_axis, state="disabled")
        self.btn_continue.pack(fill="x", pady=(4, 0))
        self.progress = ttk.Progressbar(run, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))

        self.txt = tk.Text(parent, height=10, width=46, wrap="word")
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

    def _build_plots(self, parent):
        self.fig = Figure(figsize=(7, 7), tight_layout=True)
        self.axs = [self.fig.add_subplot(311),
                    self.fig.add_subplot(312),
                    self.fig.add_subplot(313)]
        for ax, lbl in zip(self.axs, ("Bx", "By", "Bz")):
            ax.set_ylabel(f"{lbl} (mT)")
            ax.grid(True, alpha=0.3)
        self.axs[-1].set_xlabel("time (s)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _refresh_behavior(self):
        if self.var_kind.get() == "rotation":
            self.ramp_frame.pack_forget()
            self.rot_frame.pack(fill="x", pady=(4, 0))
        else:
            self.rot_frame.pack_forget()
            self.ramp_frame.pack(fill="x", pady=(4, 0))

    # ---------- cal ----------

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("NumPy array", "*.npy")],
            initialfile="calibration_M.npy")
        if p:
            self.var_path.set(p)
            self._load_file(Path(p))

    def _load_file(self, path: Path):
        try:
            M, Minv = load_calibration(path)
        except Exception as e:
            self.M = self.Minv = self.B0 = None
            self.lbl_cal.config(text=f"Failed: {e}", foreground="red")
            return
        self.M = M
        self.Minv = Minv
        self.B0 = load_b0(path.parent / "calibration_B0.npy")
        cond = np.linalg.cond(M)
        extra = f"   B0 loaded" if self.B0 is not None else ""
        self.lbl_cal.config(text=f"Loaded {path.name}   cond = {cond:.2f}{extra}",
                            foreground="black")

    # ---------- log ----------

    def _log(self, msg):
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    # ---------- start ----------

    def _start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        if self.Minv is None:
            messagebox.showerror("No calibration", "Load a calibration matrix first.")
            return
        mode = self.var_mode.get()
        if mode == "scpi" and not HAVE_PYVISA:
            messagebox.showerror("pyvisa missing",
                                 "pyvisa is required for SCPI axis switching. "
                                 "Choose Manual mode instead.")
            return

        try:
            rate = float(self.var_rate.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid", "Update rate is not a number")
            return
        if not (MIN_UPDATE_RATE_HZ <= rate <= MAX_UPDATE_RATE_HZ):
            messagebox.showerror(
                "Invalid",
                f"Update rate must be in [{MIN_UPDATE_RATE_HZ}, {MAX_UPDATE_RATE_HZ}] Hz")
            return

        try:
            if self.var_kind.get() == "rotation":
                freq = float(self.var_freq.get())
                amp = float(self.var_amp.get())
                dur = float(self.var_dur_rot.get())
                plane = self.var_plane.get()
                max_freq = rate / MIN_SAMPLES_PER_CYCLE
                if freq <= 0 or freq > max_freq:
                    raise ValueError(
                        f"Frequency must be in (0, {max_freq:.2f}] Hz given rate {rate:.1f} Hz")
                if dur <= 0.2:
                    raise ValueError("Duration must be > 0.2 s")
                if amp <= 0:
                    raise ValueError("Amplitude must be > 0 mT")
                t, B_pred = build_rotation(plane, freq, amp, dur, rate)
                descr = f"rotation in {plane.upper()}, {freq:.2f} Hz, ±{amp:.2f} mT, {dur:.2f} s"
            else:
                axis = self.var_axis.get()
                start = float(self.var_start.get())
                end = float(self.var_end.get())
                dur = float(self.var_dur_rmp.get())
                if dur <= 0.2:
                    raise ValueError("Duration must be > 0.2 s")
                t, B_pred = build_ramp(axis, start, end, dur, rate)
                descr = f"ramp on {axis.upper()}: {start:+.2f} → {end:+.2f} mT over {dur:.2f} s"
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Invalid", str(e))
            return

        coil = CoilDriver(self.var_coil.get())
        try:
            coil.open()
        except Exception as e:
            messagebox.showerror("DAQ", f"Coil driver open failed: {e}")
            return
        reader = F71Analog(self.var_ai_dev.get(), self.var_ai_chan.get(),
                           float(self.var_k.get()),
                           samples=AI_SAMPLES_PER_READ, rate=AI_RATE_HZ)
        scpi = None
        if mode == "scpi":
            scpi = F71SCPI(self.var_visa.get())
            try:
                scpi.open()
            except Exception as e:
                coil.close()
                messagebox.showerror("SCPI", f"F71 SCPI open failed: {e}")
                return

        self._coil, self._scpi = coil, scpi
        self._t = t
        self._B_pred = B_pred
        self._B_meas = np.full_like(B_pred, np.nan)
        self.abort.clear()
        self.ack.clear()
        self.progress["maximum"] = 3 * len(t)
        self.progress["value"] = 0

        B0_sub = self.B0 if (self.var_sub_b0.get() and self.B0 is not None) else None

        self._log(f"\n=== {descr} ===")
        self._log(f"N = {len(t)} samples at {rate:.1f} Hz "
                  f"(total run ≈ {3*len(t)/rate:.1f} s across 3 axes)")

        self._draw_predicted()

        self.worker = BehaviorWorker(
            t=t, B_pred=B_pred, Minv=self.Minv, B0_sub=B0_sub,
            coil=coil, reader=reader, scpi=scpi,
            update_rate_hz=rate, events=self.events, abort=self.abort,
            mode=mode, ack=self.ack,
        )
        self.worker.start()
        self.btn_start.config(state="disabled")
        self.btn_abort.config(state="normal")

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
        k = ev["kind"]
        if k == "prompt_axis":
            ax = ev["axis"].upper()
            self._log(f"→ Switch F71 analog output to {ax}-axis, then click "
                      f"'Axis switched — continue'.")
            self.btn_continue.config(state="normal")
        elif k == "pass_start":
            self._log(f"  → Pass: measuring B{ev['axis'].upper()}")
        elif k == "progress":
            ax_i = "xyz".index(ev["axis"])
            self._B_meas[:, ax_i] = np.array(ev["B_meas_axis"])
            self.progress["value"] = ax_i * ev["N"] + ev["i"] + 1
            self._redraw()
        elif k == "pass_done":
            ax_i = "xyz".index(ev["axis"])
            self._B_meas[:, ax_i] = np.array(ev["B_meas_axis"])
            self._redraw()
        elif k == "done":
            self._B_meas = np.array(ev["B_meas"])
            self._redraw()
            self._finish(success=True)
            self._report_rmse()
        elif k == "aborted":
            self._log("Aborted.")
            self._finish(success=False)
        elif k == "error":
            self._log(f"ERROR: {ev['message']}")
            messagebox.showerror("Error", ev["message"])
            self._finish(success=False)

    def _finish(self, success: bool):
        if self._coil is not None:
            try: self._coil.close()
            finally: self._coil = None
        if self._scpi is not None:
            try: self._scpi.close()
            finally: self._scpi = None
        self.btn_start.config(state="normal")
        self.btn_abort.config(state="disabled")
        self.btn_continue.config(state="disabled")

    # ---------- plots ----------

    def _draw_predicted(self):
        if self._t is None or self._B_pred is None:
            return
        for ax, col in zip(self.axs, range(3)):
            ax.clear()
            ax.plot(self._t, self._B_pred[:, col], "--", label="predicted", color="#0a6")
            ax.set_ylabel(f"B{'xyz'[col]} (mT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        self.axs[-1].set_xlabel("time (s)")
        self.canvas.draw_idle()

    def _redraw(self):
        if self._t is None or self._B_pred is None:
            return
        for ax, col in zip(self.axs, range(3)):
            ax.clear()
            ax.plot(self._t, self._B_pred[:, col], "--", label="predicted", color="#0a6")
            if self._B_meas is not None:
                meas = self._B_meas[:, col]
                if not np.all(np.isnan(meas)):
                    ax.plot(self._t, meas, "o", markersize=3, label="measured", color="#c33")
            ax.set_ylabel(f"B{'xyz'[col]} (mT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        self.axs[-1].set_xlabel("time (s)")
        self.canvas.draw_idle()

    def _report_rmse(self):
        if self._B_pred is None or self._B_meas is None:
            return
        resid = self._B_meas - self._B_pred
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=0))
        pk = np.nanmax(np.abs(self._B_pred), axis=0)
        pct = np.where(pk > 1e-6, 100 * rmse / pk, np.nan)
        self._log("--- Fit quality ---")
        for ax, col in zip("xyz", range(3)):
            self._log(f"  B{ax.upper()}: RMSE = {rmse[col]:.3f} mT"
                      + (f" ({pct[col]:.1f}% of peak)" if not np.isnan(pct[col]) else ""))

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.abort.set()
            self.ack.set()
            self.worker.join(timeout=2.0)
        if self._coil is not None:
            try: self._coil.close()
            except Exception: pass
        if self._scpi is not None:
            try: self._scpi.close()
            except Exception: pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    FieldBehaviorGUI(root)
    root.mainloop()
