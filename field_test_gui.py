"""
Static field validation GUI.

Enter a target (Bx, By, Bz) in mT, the GUI computes the coil voltages via the
chosen calibration matrix and applies them. Check the magnetometer display to
confirm the actual field matches.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import numpy as np

from hardware_io import (
    CoilDriver,
    V_MAX,
    HAVE_NIDAQMX,
    load_calibration,
    load_b0,
    voltage_for_field,
)


class FieldTestGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Field validation (static)")
        self.root.geometry("620x440")

        self.M = None
        self.Minv = None
        self.B0 = None

        self.controller = CoilDriver()
        self._daq_ok = True
        try:
            self.controller.open()
        except Exception as e:
            self._daq_ok = False
            messagebox.showwarning(
                "DAQ unavailable",
                f"Could not open coil driver:\n{e}\n\nGUI will run but nothing will be applied.",
            )

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_file(Path("calibration_M.npy"))

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        cal = ttk.LabelFrame(frm, text="Calibration matrix", padding=8)
        cal.pack(fill="x")
        pick = ttk.Frame(cal)
        pick.pack(fill="x")
        self.var_path = tk.StringVar(value="calibration_M.npy")
        ttk.Entry(pick, textvariable=self.var_path).pack(side="left", fill="x", expand=True)
        ttk.Button(pick, text="Browse...", command=self._browse).pack(side="left", padx=4)
        ttk.Button(pick, text="Reload",
                   command=lambda: self._load_file(Path(self.var_path.get()))).pack(side="left")

        self.var_subtract_b0 = tk.BooleanVar(value=False)
        ttk.Checkbutton(cal, text="Subtract ambient B0 (if loaded) from target",
                        variable=self.var_subtract_b0).pack(anchor="w", pady=(4, 0))

        self.lbl_status = ttk.Label(cal, text="(no calibration loaded)", foreground="#555")
        self.lbl_status.pack(anchor="w", pady=(4, 0))

        inp = ttk.LabelFrame(frm, text="Target field (mT)", padding=8)
        inp.pack(fill="x", pady=8)
        self.vars_B = [tk.DoubleVar(value=0.0) for _ in range(3)]
        for i, lbl in enumerate(("Bx", "By", "Bz")):
            r = ttk.Frame(inp)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=lbl, width=4).pack(side="left")
            ttk.Entry(r, textvariable=self.vars_B[i], width=12, justify="right").pack(side="left")
            ttk.Label(r, text="mT").pack(side="left", padx=(4, 0))

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Compute & Apply", command=self._apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Zero", command=self._zero).pack(side="left", padx=4)
        ttk.Button(btns, text="Quit", command=self._on_close).pack(side="right", padx=4)

        out = ttk.LabelFrame(frm, text="Output", padding=8)
        out.pack(fill="both", expand=True, pady=(8, 0))
        self.lbl_voltage = ttk.Label(out, text="Voltages: — / — / — V",
                                     font=("TkFixedFont", 10))
        self.lbl_voltage.pack(anchor="w")
        self.lbl_warn = ttk.Label(out, text="", foreground="red")
        self.lbl_warn.pack(anchor="w", pady=(4, 0))
        self.lbl_applied = ttk.Label(out, text="", foreground="#0a0")
        self.lbl_applied.pack(anchor="w", pady=(4, 0))

        hint = (
            "Tip: enter a target, click Apply, then read the magnetometer.\n"
            "Validation is good if the F71 reads within a few % of the target on all axes."
        )
        ttk.Label(out, text=hint, foreground="#666", justify="left").pack(anchor="w", pady=(8, 0))

    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("NumPy array", "*.npy")],
            initialfile="calibration_M.npy",
        )
        if path:
            self.var_path.set(path)
            self._load_file(Path(path))

    def _load_file(self, path: Path):
        try:
            M, Minv = load_calibration(path)
        except Exception as e:
            self.M = self.Minv = self.B0 = None
            self.lbl_status.config(text=f"Failed to load {path.name}: {e}", foreground="red")
            return
        self.M = M
        self.Minv = Minv
        self.B0 = load_b0(path.parent / "calibration_B0.npy")
        cond = np.linalg.cond(M)
        col_norms = np.linalg.norm(M, axis=0)
        status = (f"Loaded {path.name}   cond(M) = {cond:.2f}   "
                  f"col norms = [{col_norms[0]:.1f}, {col_norms[1]:.1f}, {col_norms[2]:.1f}] mT/V")
        if self.B0 is not None:
            status += (f"   B0 = ({self.B0[0]:+.3f}, "
                       f"{self.B0[1]:+.3f}, {self.B0[2]:+.3f}) mT")
        else:
            status += "   (no B0 file)"
        self.lbl_status.config(text=status, foreground="black")

    def _read_target(self):
        vals = []
        for i, var in enumerate(self.vars_B):
            try:
                vals.append(float(var.get()))
            except (tk.TclError, ValueError):
                raise ValueError(f"B{'xyz'[i]} is not a number")
        return np.array(vals)

    def _apply(self):
        if self.Minv is None:
            messagebox.showerror("No calibration", "Load a calibration matrix first.")
            return
        try:
            B = self._read_target()
        except ValueError as e:
            messagebox.showerror("Invalid", str(e))
            return

        B0 = self.B0 if (self.var_subtract_b0.get() and self.B0 is not None) else None
        V = voltage_for_field(B, self.Minv, B0=B0)
        self.lbl_voltage.config(
            text=f"Voltages: {V[0]:+.3f} / {V[1]:+.3f} / {V[2]:+.3f} V"
        )
        if np.any(np.abs(V) > V_MAX):
            self.lbl_warn.config(text=f"OUT OF RANGE: |V| > {V_MAX:.1f} V — not applied")
            self.lbl_applied.config(text="")
            return
        self.lbl_warn.config(text="")

        if not self._daq_ok:
            self.lbl_applied.config(text="(DAQ not available — no output)")
            return
        try:
            self.controller.write(*V)
        except Exception as e:
            messagebox.showerror("Write failed", str(e))
            return
        self.lbl_applied.config(
            text=(f"Applied. Target B = "
                  f"({B[0]:+.3f}, {B[1]:+.3f}, {B[2]:+.3f}) mT. "
                  f"Check the magnetometer.")
        )

    def _zero(self):
        for var in self.vars_B:
            var.set(0.0)
        try:
            self.controller.write(0.0, 0.0, 0.0)
        except Exception:
            pass
        self.lbl_voltage.config(text="Voltages: 0.000 / 0.000 / 0.000 V")
        self.lbl_warn.config(text="")
        self.lbl_applied.config(text="Output zeroed.")

    def _on_close(self):
        try:
            self.controller.close()
        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    FieldTestGUI(root)
    root.mainloop()
