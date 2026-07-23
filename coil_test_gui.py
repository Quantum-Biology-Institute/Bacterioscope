"""
Simple manual coil voltage control.
Writes static DC voltages to Dev3/AO0-2 for hardware sanity checking.
"""

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import nidaqmx
    HAVE_NIDAQMX = True
except ImportError:
    HAVE_NIDAQMX = False


DEVICE = "Dev3"
CHANNELS = ("ao0", "ao1", "ao2")
LABELS = ("Coil A (Dev3/AO0)", "Coil B (Dev3/AO1)", "Coil C (Dev3/AO2)")
V_MIN, V_MAX = -10.0, 10.0


class CoilController:
    def __init__(self):
        self.task = None
        if HAVE_NIDAQMX:
            self.task = nidaqmx.Task()
            for ch in CHANNELS:
                self.task.ao_channels.add_ao_voltage_chan(
                    f"{DEVICE}/{ch}", min_val=V_MIN, max_val=V_MAX
                )
            self.task.write([0.0, 0.0, 0.0])

    def write(self, va, vb, vc):
        if self.task is None:
            return
        self.task.write([float(va), float(vb), float(vc)])

    def close(self):
        if self.task is not None:
            try:
                self.task.write([0.0, 0.0, 0.0])
            finally:
                self.task.close()
                self.task = None


class CoilTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coil Voltage Test")
        self.root.geometry("520x340")

        self.controller = CoilController()
        self.vars = [tk.DoubleVar(value=0.0) for _ in CHANNELS]
        self._syncing = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        header = "Manual coil drive"
        if not HAVE_NIDAQMX:
            header += "  —  nidaqmx NOT available (GUI-only mode)"
        ttk.Label(frm, text=header, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(frm, text=f"Voltage range: {V_MIN:+.1f} to {V_MAX:+.1f} V").pack(anchor="w", pady=(0, 10))

        for i, (label, var) in enumerate(zip(LABELS, self.vars)):
            row = ttk.Frame(frm)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=22).pack(side="left")

            entry = ttk.Entry(row, width=8, textvariable=var, justify="right")
            entry.pack(side="left", padx=(0, 8))

            scale = ttk.Scale(
                row, from_=V_MIN, to=V_MAX, orient="horizontal",
                variable=var, length=260,
                command=lambda v, idx=i: self._on_scale(idx, v),
            )
            scale.pack(side="left", fill="x", expand=True)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(14, 4))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Zero All", command=self._zero).pack(side="left", padx=4)
        ttk.Button(btns, text="Quit", command=self._on_close).pack(side="right", padx=4)

        test_row = ttk.Frame(frm)
        test_row.pack(fill="x", pady=(8, 4))
        ttk.Label(test_row, text="Single-coil test at").pack(side="left")
        self.test_v = tk.DoubleVar(value=1.0)
        ttk.Entry(test_row, width=6, textvariable=self.test_v, justify="right").pack(side="left", padx=4)
        ttk.Label(test_row, text="V (others = 0):").pack(side="left", padx=(0, 8))
        ttk.Button(test_row, text="Test A", command=lambda: self._test_only(0)).pack(side="left", padx=2)
        ttk.Button(test_row, text="Test B", command=lambda: self._test_only(1)).pack(side="left", padx=2)
        ttk.Button(test_row, text="Test C", command=lambda: self._test_only(2)).pack(side="left", padx=2)

        self.status = ttk.Label(frm, text="Output: 0.000, 0.000, 0.000 V", foreground="#555")
        self.status.pack(anchor="w", pady=(10, 0))

    def _on_scale(self, idx, value):
        if self._syncing:
            return
        try:
            rounded = round(float(value), 3)
            self._syncing = True
            self.vars[idx].set(rounded)
        finally:
            self._syncing = False

    def _read_voltages(self):
        vs = []
        for i, var in enumerate(self.vars):
            try:
                v = float(var.get())
            except (tk.TclError, ValueError):
                raise ValueError(f"{LABELS[i]}: not a number")
            if not (V_MIN <= v <= V_MAX):
                raise ValueError(f"{LABELS[i]}: {v:+.3f} V is outside [{V_MIN:+.1f}, {V_MAX:+.1f}] V")
            vs.append(v)
        return vs

    def _apply(self):
        try:
            vs = self._read_voltages()
        except ValueError as e:
            messagebox.showerror("Invalid voltage", str(e))
            return
        try:
            self.controller.write(*vs)
        except Exception as e:
            messagebox.showerror("DAQ write failed", str(e))
            return
        mode = "" if HAVE_NIDAQMX else "  (simulated)"
        self.status.config(text=f"Output: {vs[0]:+.3f}, {vs[1]:+.3f}, {vs[2]:+.3f} V{mode}")

    def _zero(self):
        for var in self.vars:
            var.set(0.0)
        self._apply()

    def _test_only(self, idx):
        try:
            v = float(self.test_v.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid voltage", "Test voltage is not a number")
            return
        if not (V_MIN <= v <= V_MAX):
            messagebox.showerror("Invalid voltage", f"Test voltage must be in [{V_MIN:+.1f}, {V_MAX:+.1f}] V")
            return
        for i, var in enumerate(self.vars):
            var.set(v if i == idx else 0.0)
        self._apply()

    def _on_close(self):
        try:
            self.controller.close()
        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CoilTestGUI(root)
    root.mainloop()
