"""
Magnetofluorescence Experiment GUI
Provides a unified interface for configuring and running both ON/OFF cycling
and field ramp experiments.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import threading
import time
from PIL import Image, ImageTk


# =========================
# CONFIGURATION DATACLASSES
# =========================

@dataclass
class CommonConfig:
    """Parameters shared by all experiment types"""
    # Timing
    snap_interval_s: float = 2.0
    led_advance_s: float = 0.10
    led_pulse_s: float = 1.6
    trig_pulse_s: float = 0.005

    # LED intensity
    led_intensity_mA: float = 4000.0  # Max hardware intensity

    # Calibration
    calibration_matrix_path: str = "calibration_M.npy"

    # Devices
    dev_mag: str = "Dev3"
    ao_mag_chans: Tuple[str, str, str] = ("ao0", "ao1", "ao2")
    dev_ao: str = "Dev2"
    ao_led_chan: str = "ao0"
    ao_cam_chan: str = "ao1"
    ao_vmin: float = -10.0
    ao_vmax: float = 10.0
    ao_headroom_v: float = 0.02
    ttl_low_v: float = 0.0
    ttl_high_v: float = 5.0
    sample_rate: float = 10000.0

    # Saving
    save_dir: str = field(default_factory=lambda: f"C:\\data\\{datetime.now().strftime('%Y_%m_%d')}")
    fiji_exe: str = r"C:\Program Files\fiji-latest-win64-jdk\Fiji\fiji-windows-x64.exe"
    mm_show_display: bool = False
    mm_set_exposure_ms: float = 400.0

    # Sample
    sample_name: str = "sample"

    # Experiment notes (saved in metadata)
    notes: str = ""

    # Safety
    acq_timeout_margin_s: float = 5.0
    daq_start_delay_s: float = 0.4


@dataclass
class OnOffConfig(CommonConfig):
    """Configuration for ON/OFF cycling experiment"""
    schedule_mode: str = "start_off"  # or "start_on"
    target_field_mT: Tuple[float, float, float] = (10.0, 0.0, 0.0)
    ambient_field_mT: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    baseline_first_s: float = 300.0
    cycles_n: int = 6
    dwell_s: float = 60.0


@dataclass
class RampConfig(CommonConfig):
    """Configuration for field ramp experiment"""
    max_field_mT: Tuple[float, float, float] = (20.0, 0.0, 0.0)
    step_mT: float = 2
    ramp_first: str = "down"  # or "up"
    off_first: bool = True
    baseline_first_s: float = 300.0  # Initial baseline period
    segment_duration_s: float = 90.0


@dataclass
class CustomConfig(CommonConfig):
    """Configuration for custom field sequence experiment"""
    csv_file_path: str = ""  # Path to CSV with Bx,By,Bz columns
    field_sequence: List[Tuple[float, float, float]] = field(default_factory=list)
    baseline_first_s: float = 300.0  # Initial baseline period
    segment_duration_s: float = 90.0  # Duration for each field
    off_between_fields: bool = True  # Include OFF segment between fields


# =========================
# DURATION CALCULATORS
# =========================

def calculate_onoff_duration(config: OnOffConfig) -> float:
    """Calculate total duration for ON/OFF experiment"""
    return config.baseline_first_s + 2 * config.cycles_n * config.dwell_s


def calculate_ramp_duration(config: RampConfig) -> float:
    """Calculate total duration for field ramp experiment"""
    # Calculate number of magnitude levels
    Bmax_mag = float(np.linalg.norm(config.max_field_mT))
    n_steps = int(np.floor(Bmax_mag / config.step_mT))

    # Levels in one direction (including max)
    n_levels = n_steps + 1

    # Total segments: optional initial OFF + baseline + (ON + OFF) for each level in both directions
    n_segments = (1 if config.off_first else 0) + 2 * n_levels * 2

    return config.baseline_first_s + n_segments * config.segment_duration_s


def calculate_custom_duration(config: CustomConfig) -> float:
    """Calculate total duration for custom field sequence experiment"""
    n_fields = len(config.field_sequence)
    if n_fields == 0:
        return config.baseline_first_s

    # Baseline + (optional initial OFF) + n_fields * (ON + optional OFF)
    if config.off_between_fields:
        # Initial OFF + (ON + OFF) for each field
        n_segments = 1 + n_fields * 2
    else:
        # Just ON for each field
        n_segments = n_fields

    return config.baseline_first_s + n_segments * config.segment_duration_s


# =========================
# GUI APPLICATION
# =========================

class ExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Magnetofluorescence Experiment Control")
        self.root.geometry("1200x900")

        # Current configuration
        self.experiment_type = tk.StringVar(value="onoff")
        self.config = OnOffConfig()
        self.dry_run = tk.BooleanVar(value=False)

        # Progress tracking
        self.progress_tracker = None
        self.progress_poll_id = None
        self.progress_line = None
        self.field_text = None
        self.led_indicator = None
        self.experiment_start_time = None

        # Progress window widgets
        self.progress_window = None
        self.progress_bar = None
        self.progress_pct_label = None
        self.time_elapsed_label = None
        self.time_remaining_label = None
        self.time_finish_label = None
        self.frame_progress_label = None
        self.field_display_label = None
        self.led_display_canvas = None

        # Create main layout
        self.create_widgets()

        # Initial update
        self.on_experiment_type_change()

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Left panel: Controls
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Right panel: Previews
        preview_frame = ttk.Frame(main_frame, padding="5")
        preview_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        # === CONTROL PANEL ===

        # QBI Logo at the top
        try:
            logo_path = Path("QBI_logo.png")
            if logo_path.exists():
                logo_image = Image.open(logo_path)
                # Resize logo to fit nicely (max width 200px, maintain aspect ratio)
                logo_image.thumbnail((200, 80), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_image)
                logo_label = ttk.Label(control_frame, image=logo_photo)
                logo_label.image = logo_photo  # Keep a reference to prevent garbage collection
                logo_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        except Exception as e:
            print(f"[GUI] Could not load logo: {e}")

        # Experiment Type Selection
        type_frame = ttk.LabelFrame(control_frame, text="Experiment Type", padding="10")
        type_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Radiobutton(type_frame, text="ON/OFF Cycling", variable=self.experiment_type,
                       value="onoff", command=self.on_experiment_type_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(type_frame, text="Field Ramp", variable=self.experiment_type,
                       value="ramp", command=self.on_experiment_type_change).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Radiobutton(type_frame, text="Custom", variable=self.experiment_type,
                       value="custom", command=self.on_experiment_type_change).grid(row=0, column=2, sticky=tk.W, padx=(20, 0))

        # Scrollable parameters frame
        canvas = tk.Canvas(control_frame, width=450, height=400)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        self.params_container = ttk.Frame(canvas)

        self.params_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.params_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
        control_frame.rowconfigure(2, weight=1)

        # Duration display
        duration_frame = ttk.LabelFrame(control_frame, text="Experiment Duration", padding="10")
        duration_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.duration_label = ttk.Label(duration_frame, text="0.0 s (0.0 min)", font=('Arial', 12, 'bold'))
        self.duration_label.pack()

        self.frames_label = ttk.Label(duration_frame, text="0 frames", font=('Arial', 10))
        self.frames_label.pack()

        # Action buttons
        button_frame = ttk.Frame(control_frame, padding="5")
        button_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="Load Config", command=self.load_config).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).grid(row=0, column=1, padx=5)

        # Dry run checkbox
        ttk.Checkbutton(button_frame, text="Dry Run", variable=self.dry_run).grid(row=0, column=2, padx=5)

        ttk.Button(button_frame, text="Run Experiment", command=self.run_experiment,
                  style='Accent.TButton').grid(row=0, column=3, padx=5)

        # === PREVIEW PANEL ===

        # Acquisition timing preview
        timing_frame = ttk.LabelFrame(preview_frame, text="Acquisition Timing Preview", padding="5")
        timing_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.timing_fig = Figure(figsize=(6, 3), dpi=100)
        self.timing_ax = self.timing_fig.add_subplot(111)
        self.timing_canvas = FigureCanvasTkAgg(self.timing_fig, master=timing_frame)
        self.timing_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Experiment preview
        experiment_frame = ttk.LabelFrame(preview_frame, text="Experiment Preview", padding="5")
        experiment_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.experiment_fig = Figure(figsize=(6, 3), dpi=100)
        self.experiment_ax = self.experiment_fig.add_subplot(111)
        self.experiment_canvas = FigureCanvasTkAgg(self.experiment_fig, master=experiment_frame)
        self.experiment_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Storage for parameter widgets
        self.param_widgets = {}

    def create_parameter_widgets(self):
        """Create parameter input widgets based on experiment type"""
        # Clear existing widgets
        for widget in self.params_container.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        row = 0

        if self.experiment_type.get() == "onoff":
            # ON/OFF Cycling Parameters
            config = self.config

            # Schedule mode
            frame = ttk.LabelFrame(self.params_container, text="Schedule", padding="10")
            frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
            row += 1

            schedule_var = tk.StringVar(value=config.schedule_mode)
            ttk.Radiobutton(frame, text="Start OFF", variable=schedule_var,
                           value="start_off", command=lambda: self.update_param('schedule_mode', schedule_var.get())).grid(row=0, column=0)
            ttk.Radiobutton(frame, text="Start ON", variable=schedule_var,
                           value="start_on", command=lambda: self.update_param('schedule_mode', schedule_var.get())).grid(row=0, column=1, padx=(20, 0))
            self.param_widgets['schedule_mode'] = schedule_var

            # Field vectors
            self.add_vector_input("Target Field (ON)", 'target_field_mT', config.target_field_mT, row)
            row += 1
            self.add_vector_input("Ambient Field (OFF)", 'ambient_field_mT', config.ambient_field_mT, row)
            row += 1

            # Timing parameters
            self.add_float_input("Baseline Duration (s)", 'baseline_first_s', config.baseline_first_s, row)
            row += 1
            self.add_int_input("Number of Cycles", 'cycles_n', config.cycles_n, row)
            row += 1
            self.add_float_input("Dwell Time (s)", 'dwell_s', config.dwell_s, row)
            row += 1

        elif self.experiment_type.get() == "ramp":
            # Field Ramp Parameters
            config = self.config

            # Ramp direction
            frame = ttk.LabelFrame(self.params_container, text="Ramp Configuration", padding="10")
            frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
            row += 1

            ramp_var = tk.StringVar(value=config.ramp_first)
            ttk.Radiobutton(frame, text="Down First (max→min→max)", variable=ramp_var,
                           value="down", command=lambda: self.update_param('ramp_first', ramp_var.get())).grid(row=0, column=0, sticky=tk.W)
            ttk.Radiobutton(frame, text="Up First (min→max→min)", variable=ramp_var,
                           value="up", command=lambda: self.update_param('ramp_first', ramp_var.get())).grid(row=1, column=0, sticky=tk.W)
            self.param_widgets['ramp_first'] = ramp_var

            off_first_var = tk.BooleanVar(value=config.off_first)
            ttk.Checkbutton(frame, text="Start with OFF segment", variable=off_first_var,
                           command=lambda: self.update_param('off_first', off_first_var.get())).grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
            self.param_widgets['off_first'] = off_first_var

            # Field parameters
            self.add_vector_input("Max Field", 'max_field_mT', config.max_field_mT, row)
            row += 1
            self.add_float_input("Step Size (mT)", 'step_mT', config.step_mT, row)
            row += 1
            self.add_float_input("Baseline Duration (s)", 'baseline_first_s', config.baseline_first_s, row)
            row += 1
            self.add_float_input("Segment Duration (s)", 'segment_duration_s', config.segment_duration_s, row)
            row += 1

        else:  # custom
            # Custom Field Sequence Parameters
            config = self.config

            # CSV file selection
            csv_frame = ttk.LabelFrame(self.params_container, text="Field Sequence (CSV)", padding="10")
            csv_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
            row += 1

            self.add_csv_input("CSV File (Bx,By,Bz)", 'csv_file_path', config.csv_file_path, 0, parent=csv_frame)

            # Show field count label
            self.csv_info_label = ttk.Label(csv_frame, text="No file loaded", foreground='gray')
            self.csv_info_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

            # Configuration options
            config_frame = ttk.LabelFrame(self.params_container, text="Custom Configuration", padding="10")
            config_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
            row += 1

            off_between_var = tk.BooleanVar(value=config.off_between_fields)
            ttk.Checkbutton(config_frame, text="Include OFF segment between fields", variable=off_between_var,
                           command=lambda: self.update_param('off_between_fields', off_between_var.get())).grid(row=0, column=0, sticky=tk.W)
            self.param_widgets['off_between_fields'] = off_between_var

            self.add_float_input("Baseline Duration (s)", 'baseline_first_s', config.baseline_first_s, row)
            row += 1
            self.add_float_input("Segment Duration (s)", 'segment_duration_s', config.segment_duration_s, row)
            row += 1

        # Common parameters
        common_frame = ttk.LabelFrame(self.params_container, text="Common Parameters", padding="10")
        common_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.add_float_input("Snap Interval (s)", 'snap_interval_s', config.snap_interval_s, row, parent=common_frame)
        row += 1
        self.add_float_input("LED Intensity (mA)", 'led_intensity_mA', config.led_intensity_mA, row, parent=common_frame)
        row += 1
        self.add_float_input("LED Advance (s)", 'led_advance_s', config.led_advance_s, row, parent=common_frame)
        row += 1
        self.add_float_input("LED Pulse Duration (s)", 'led_pulse_s', config.led_pulse_s, row, parent=common_frame)
        row += 1
        self.add_float_input("Camera Exposure (ms)", 'mm_set_exposure_ms', config.mm_set_exposure_ms, row, parent=common_frame)
        row += 1

        # Sample and paths
        paths_frame = ttk.LabelFrame(self.params_container, text="Sample & Paths", padding="10")
        paths_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.add_string_input("Sample Name", 'sample_name', config.sample_name, 0, parent=paths_frame)
        self.add_path_input("Save Directory", 'save_dir', config.save_dir, 1, parent=paths_frame)
        self.add_path_input("Calibration Matrix", 'calibration_matrix_path', config.calibration_matrix_path, 2, parent=paths_frame, file_mode=True)

        # Notes section
        notes_frame = ttk.LabelFrame(self.params_container, text="Experiment Notes", padding="10")
        notes_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.add_notes_input(config.notes, parent=notes_frame)

    def add_float_input(self, label, param_name, default_value, row, parent=None):
        """Add a float input field"""
        if parent is None:
            parent = self.params_container

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        var = tk.StringVar(value=str(default_value))
        entry = ttk.Entry(frame, textvariable=var, width=15)
        entry.grid(row=0, column=1, sticky=tk.E)

        var.trace_add('write', lambda *args: self.update_param(param_name, self.safe_float(var.get(), default_value)))
        self.param_widgets[param_name] = var

    def add_int_input(self, label, param_name, default_value, row, parent=None):
        """Add an integer input field"""
        if parent is None:
            parent = self.params_container

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        var = tk.StringVar(value=str(default_value))
        entry = ttk.Entry(frame, textvariable=var, width=15)
        entry.grid(row=0, column=1, sticky=tk.E)

        var.trace_add('write', lambda *args: self.update_param(param_name, self.safe_int(var.get(), default_value)))
        self.param_widgets[param_name] = var

    def add_string_input(self, label, param_name, default_value, row, parent=None):
        """Add a string input field"""
        if parent is None:
            parent = self.params_container

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(frame, textvariable=var, width=30)
        entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

        var.trace_add('write', lambda *args: self.update_param(param_name, var.get()))
        self.param_widgets[param_name] = var

    def add_path_input(self, label, param_name, default_value, row, parent=None, file_mode=False):
        """Add a path input field with browse button"""
        if parent is None:
            parent = self.params_container

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(frame, textvariable=var, width=25)
        entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

        def browse():
            if file_mode:
                path = filedialog.askopenfilename(initialdir=Path(var.get()).parent if var.get() else ".")
            else:
                path = filedialog.askdirectory(initialdir=var.get() if var.get() else ".")
            if path:
                var.set(path)

        ttk.Button(frame, text="Browse", command=browse, width=8).grid(row=0, column=2, padx=(5, 0))

        var.trace_add('write', lambda *args: self.update_param(param_name, var.get()))
        self.param_widgets[param_name] = var

    def add_vector_input(self, label, param_name, default_value, row, parent=None):
        """Add a 3D vector input (Bx, By, Bz)"""
        if parent is None:
            parent = self.params_container

        frame = ttk.LabelFrame(parent, text=label, padding="5")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)

        vars = []
        for i, axis in enumerate(['Bx (mT)', 'By (mT)', 'Bz (mT)']):
            ttk.Label(frame, text=axis).grid(row=0, column=i*2, padx=(0, 5))
            var = tk.StringVar(value=str(default_value[i]))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=0, column=i*2+1, padx=(0, 10))
            vars.append(var)

            var.trace_add('write', lambda *args: self.update_vector_param(param_name, vars, default_value))

        self.param_widgets[param_name] = vars

    def add_csv_input(self, label, param_name, default_value, row, parent=None):
        """Add a CSV file input field with browse button and load functionality"""
        if parent is None:
            parent = self.params_container

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(frame, textvariable=var, width=25)
        entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

        def browse_and_load():
            path = filedialog.askopenfilename(
                initialdir=Path(var.get()).parent if var.get() else ".",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if path:
                var.set(path)
                self.load_csv_fields(path)

        ttk.Button(frame, text="Browse", command=browse_and_load, width=8).grid(row=0, column=2, padx=(5, 0))

        var.trace_add('write', lambda *args: self.update_param(param_name, var.get()))
        self.param_widgets[param_name] = var

    def load_csv_fields(self, csv_path):
        """Load field sequence from CSV and update config"""
        try:
            from experiment_runners import load_field_sequence_from_csv
            fields = load_field_sequence_from_csv(csv_path)
            self.config.field_sequence = fields
            self.config.csv_file_path = csv_path

            # Update info label
            if hasattr(self, 'csv_info_label'):
                self.csv_info_label.config(
                    text=f"Loaded {len(fields)} field points",
                    foreground='green'
                )

            self.update_previews()
        except Exception as e:
            if hasattr(self, 'csv_info_label'):
                self.csv_info_label.config(
                    text=f"Error: {str(e)[:40]}...",
                    foreground='red'
                )
            print(f"[CSV] Error loading CSV: {e}")

    def add_notes_input(self, default_value, parent=None):
        """Add a multi-line text input for experiment notes"""
        if parent is None:
            parent = self.params_container

        # Create text widget with scrollbar
        text_frame = ttk.Frame(parent)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.notes_text = tk.Text(text_frame, width=50, height=4, wrap=tk.WORD)
        self.notes_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.notes_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.notes_text.configure(yscrollcommand=scrollbar.set)

        # Insert default value
        if default_value:
            self.notes_text.insert('1.0', default_value)

        # Bind text changes to update config
        def on_text_change(event=None):
            notes = self.notes_text.get('1.0', 'end-1c')
            self.config.notes = notes

        self.notes_text.bind('<KeyRelease>', on_text_change)
        self.notes_text.bind('<FocusOut>', on_text_change)

    def update_vector_param(self, param_name, vars, default_value):
        """Update a vector parameter from three input fields"""
        try:
            values = tuple(float(v.get()) if v.get() else default_value[i] for i, v in enumerate(vars))
            setattr(self.config, param_name, values)
            self.update_previews()
        except ValueError:
            pass

    def safe_float(self, value, default):
        """Safely convert string to float"""
        try:
            return float(value) if value else default
        except ValueError:
            return default

    def safe_int(self, value, default):
        """Safely convert string to int"""
        try:
            return int(value) if value else default
        except ValueError:
            return default

    def update_param(self, param_name, value):
        """Update a parameter and refresh previews"""
        setattr(self.config, param_name, value)
        self.update_previews()

    def on_experiment_type_change(self):
        """Handle experiment type change"""
        exp_type = self.experiment_type.get()
        if exp_type == "onoff":
            self.config = OnOffConfig()
        elif exp_type == "ramp":
            self.config = RampConfig()
        else:  # custom
            self.config = CustomConfig()

        self.create_parameter_widgets()
        self.update_previews()

    def update_previews(self):
        """Update all preview plots and duration display"""
        self.update_timing_preview()
        self.update_experiment_preview()
        self.update_duration_display()

    def update_timing_preview(self):
        """Update the acquisition timing preview (LED + Camera)"""
        self.timing_ax.clear()

        config = self.config
        snap_interval = config.snap_interval_s
        led_advance = config.led_advance_s
        led_pulse = config.led_pulse_s
        exposure = config.mm_set_exposure_ms / 1000.0  # Convert to seconds

        # Show 3 complete cycles
        t_max = 3 * snap_interval

        # Plot LED pulses for first 3 complete cycles
        for i in range(3):
            t_start = i * snap_interval - led_advance
            # For the first pulse, if it starts before t=0, show only the visible portion
            if i == 0 and t_start < 0:
                visible_start = 0
                visible_duration = led_pulse - led_advance
            else:
                visible_start = max(0, t_start)
                visible_duration = led_pulse
            self.timing_ax.barh(1, visible_duration, left=visible_start, height=0.6,
                               color='#6B7FDE', label='LED' if i == 0 else '')

        # Add partial LED pulse at the end (beginning of 4th pulse)
        if led_advance > 0:
            partial_start = t_max - led_advance
            partial_duration = led_advance
            self.timing_ax.barh(1, partial_duration, left=partial_start, height=0.6,
                               color='#6B7FDE', alpha=0.5)  # Semi-transparent to show it's partial

        # Plot camera exposures for first 3 cycles
        for i in range(3):
            t_start = i * snap_interval
            t_end = t_start + exposure
            self.timing_ax.barh(1, exposure, left=t_start, height=0.4,
                               color='#E74C3C', label='Camera' if i == 0 else '')

        self.timing_ax.set_xlim(0, t_max)
        self.timing_ax.set_ylim(0.5, 1.5)
        self.timing_ax.set_xlabel('t (s)', fontsize=10)
        self.timing_ax.set_yticks([])  # No y-axis labels
        self.timing_ax.legend(loc='upper right')
        self.timing_ax.grid(True, alpha=0.3)

        self.timing_fig.tight_layout()
        self.timing_canvas.draw()

    def update_experiment_preview(self):
        """Update the experiment preview (field vs time)"""
        self.experiment_ax.clear()

        exp_type = self.experiment_type.get()
        if exp_type == "onoff":
            self.plot_onoff_preview()
        elif exp_type == "ramp":
            self.plot_ramp_preview()
        else:  # custom
            self.plot_custom_preview()

        self.experiment_fig.tight_layout()
        self.experiment_canvas.draw()

    def plot_onoff_preview(self):
        """Plot ON/OFF cycling experiment preview"""
        config = self.config

        # Calculate field magnitude
        B_on = float(np.linalg.norm(config.target_field_mT))
        B_off = float(np.linalg.norm(config.ambient_field_mT))

        # Build time windows
        t = 0
        times = [t]
        fields = []

        if config.schedule_mode == "start_off":
            fields.append(B_off)
            t += config.baseline_first_s
            times.append(t)

            for _ in range(config.cycles_n):
                fields.append(B_on)
                t += config.dwell_s
                times.append(t)
                fields.append(B_off)
                t += config.dwell_s
                times.append(t)
        else:  # start_on
            fields.append(B_on)
            t += config.baseline_first_s
            times.append(t)

            for _ in range(config.cycles_n):
                fields.append(B_off)
                t += config.dwell_s
                times.append(t)
                fields.append(B_on)
                t += config.dwell_s
                times.append(t)

        # Plot as step function
        self.experiment_ax.step(times[:-1], fields, where='post', linewidth=2, color='#6B7FDE')
        self.experiment_ax.fill_between(times[:-1], fields, step='post', alpha=0.6, color='#6B7FDE')

        self.experiment_ax.set_xlabel('t (s)', fontsize=10)
        self.experiment_ax.set_ylabel('Magnetic Field (mT)', fontsize=10)
        self.experiment_ax.set_xlim(0, times[-1])
        self.experiment_ax.set_ylim(0, max(B_on, B_off) * 1.1 if max(B_on, B_off) > 0 else 1)
        self.experiment_ax.grid(True, alpha=0.3)

    def plot_ramp_preview(self):
        """Plot field ramp experiment preview"""
        config = self.config

        # Calculate magnitude levels
        Bmax_vec = np.array(config.max_field_mT)
        Bmax_mag = float(np.linalg.norm(Bmax_vec))
        step = config.step_mT

        if Bmax_mag <= 0 or step <= 0:
            self.experiment_ax.text(0.5, 0.5, 'Invalid parameters',
                                   ha='center', va='center', transform=self.experiment_ax.transAxes)
            return

        # Build magnitude list (descending)
        n_steps = int(np.floor(Bmax_mag / step))
        mags_desc = [Bmax_mag] + [k * step for k in range(n_steps, 0, -1)]
        mags_desc = sorted(set(mags_desc), reverse=True)

        # Build full sequence
        if config.ramp_first == "down":
            mags_full = mags_desc + list(reversed(mags_desc))
        else:
            mags_asc = list(reversed(mags_desc))
            mags_full = mags_asc + list(reversed(mags_asc))

        # Build time series with OFF segments
        times = [0]
        fields = []
        t = 0
        seg_dur = config.segment_duration_s

        # Add baseline period
        fields.append(0)
        t += config.baseline_first_s
        times.append(t)

        if config.off_first:
            fields.append(0)
            t += seg_dur
            times.append(t)

        for mag in mags_full:
            # ON segment
            fields.append(mag)
            t += seg_dur
            times.append(t)

            # OFF segment after each ON
            fields.append(0)
            t += seg_dur
            times.append(t)

        # Plot
        self.experiment_ax.step(times[:-1], fields, where='post', linewidth=2, color='#6B7FDE')
        self.experiment_ax.fill_between(times[:-1], fields, step='post', alpha=0.6, color='#6B7FDE')

        # Add magnitude labels on bars
        for i, mag in enumerate(mags_full):
            if mag > 0:
                t_mid = config.baseline_first_s + (config.off_first * seg_dur) + i * 2 * seg_dur + seg_dur / 2
                self.experiment_ax.text(t_mid, mag + Bmax_mag * 0.02, f'{mag:.1f}',
                                       ha='center', va='bottom', fontsize=8)

        self.experiment_ax.set_xlabel('t (s)', fontsize=10)
        self.experiment_ax.set_ylabel('Magnetic Field (mT)', fontsize=10)
        self.experiment_ax.set_xlim(0, times[-1])
        self.experiment_ax.set_ylim(0, Bmax_mag * 1.15)
        self.experiment_ax.grid(True, alpha=0.3)

    def plot_custom_preview(self):
        """Plot custom field sequence experiment preview"""
        config = self.config

        # Check if field sequence is loaded
        if not config.field_sequence:
            self.experiment_ax.text(0.5, 0.5, 'No CSV file loaded\nSelect a CSV file with Bx,By,Bz columns',
                                   ha='center', va='center', transform=self.experiment_ax.transAxes,
                                   fontsize=10, color='gray')
            return

        fields_seq = config.field_sequence
        seg_dur = config.segment_duration_s

        # Calculate magnitudes
        mags = [float(np.linalg.norm(B)) for B in fields_seq]
        max_mag = max(mags) if mags else 1.0

        # Build time series
        times = [0]
        fields = []
        t = 0

        # Baseline
        fields.append(0)
        t += config.baseline_first_s
        times.append(t)

        # Initial OFF if configured
        if config.off_between_fields:
            fields.append(0)
            t += seg_dur
            times.append(t)

        for mag in mags:
            # ON segment
            fields.append(mag)
            t += seg_dur
            times.append(t)

            # OFF segment if configured
            if config.off_between_fields:
                fields.append(0)
                t += seg_dur
                times.append(t)

        # Plot
        self.experiment_ax.step(times[:-1], fields, where='post', linewidth=2, color='#6B7FDE')
        self.experiment_ax.fill_between(times[:-1], fields, step='post', alpha=0.6, color='#6B7FDE')

        # Add some labels for first few fields
        n_labeled = min(len(mags), 10)
        for i in range(n_labeled):
            if config.off_between_fields:
                t_mid = config.baseline_first_s + seg_dur + i * 2 * seg_dur + seg_dur / 2
            else:
                t_mid = config.baseline_first_s + i * seg_dur + seg_dur / 2

            if mags[i] > 0:
                self.experiment_ax.text(t_mid, mags[i] + max_mag * 0.02, f'{mags[i]:.1f}',
                                       ha='center', va='bottom', fontsize=7)

        self.experiment_ax.set_xlabel('t (s)', fontsize=10)
        self.experiment_ax.set_ylabel('|B| (mT)', fontsize=10)
        self.experiment_ax.set_xlim(0, times[-1])
        self.experiment_ax.set_ylim(0, max_mag * 1.15 if max_mag > 0 else 1)
        self.experiment_ax.grid(True, alpha=0.3)

        # Add info text
        self.experiment_ax.set_title(f'Custom: {len(mags)} field points', fontsize=9)

    def update_duration_display(self):
        """Update duration and frame count display"""
        if isinstance(self.config, OnOffConfig):
            duration = calculate_onoff_duration(self.config)
        elif isinstance(self.config, RampConfig):
            duration = calculate_ramp_duration(self.config)
        else:  # CustomConfig
            duration = calculate_custom_duration(self.config)

        # Calculate number of frames
        n_frames = int(np.floor(duration / self.config.snap_interval_s)) + 1

        minutes = duration / 60
        hours = duration / 3600

        if hours >= 1:
            time_str = f"{duration:.1f} s ({hours:.2f} h)"
        else:
            time_str = f"{duration:.1f} s ({minutes:.1f} min)"

        self.duration_label.config(text=time_str)
        self.frames_label.config(text=f"{n_frames} frames @ {self.config.snap_interval_s}s interval")

    def save_config(self):
        """Save current configuration to JSON file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config_dict = asdict(self.config)
                with open(filename, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        """Load configuration from JSON file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_dict = json.load(f)

                # Determine experiment type from config
                if 'schedule_mode' in config_dict:
                    self.experiment_type.set("onoff")
                    self.config = OnOffConfig(**config_dict)
                elif 'csv_file_path' in config_dict:
                    self.experiment_type.set("custom")
                    # Convert field_sequence list of lists back to tuples
                    if 'field_sequence' in config_dict and config_dict['field_sequence']:
                        config_dict['field_sequence'] = [tuple(f) for f in config_dict['field_sequence']]
                    self.config = CustomConfig(**config_dict)
                else:
                    self.experiment_type.set("ramp")
                    self.config = RampConfig(**config_dict)

                self.create_parameter_widgets()
                self.update_previews()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def run_experiment(self):
        """Run the selected experiment"""
        # Determine experiment type name and duration
        if isinstance(self.config, OnOffConfig):
            exp_type = "ON/OFF Cycling"
            duration = calculate_onoff_duration(self.config)
        elif isinstance(self.config, RampConfig):
            exp_type = "Field Ramp"
            duration = calculate_ramp_duration(self.config)
        else:  # CustomConfig
            exp_type = "Custom Field Sequence"
            duration = calculate_custom_duration(self.config)
            # Validate CSV is loaded for custom experiments
            if not self.config.field_sequence:
                messagebox.showerror("Error", "Please load a CSV file with field values before running a custom experiment.")
                return

        dry_run_text = " (DRY RUN - No hardware will be used)" if self.dry_run.get() else ""
        msg = f"Run {exp_type} experiment{dry_run_text}?\n\nDuration: {duration/60:.1f} minutes\nSample: {self.config.sample_name}"

        if not messagebox.askyesno("Confirm Experiment", msg):
            return

        # Create progress tracker
        from experiment_runners import ExperimentProgress
        self.progress_tracker = ExperimentProgress()

        # Start progress polling
        self.start_progress_polling()

        # Run in separate thread to avoid blocking GUI
        def run_thread():
            try:
                if isinstance(self.config, OnOffConfig):
                    self.run_onoff_experiment()
                elif isinstance(self.config, RampConfig):
                    self.run_ramp_experiment()
                else:  # CustomConfig
                    self.run_custom_experiment()

                success_msg = "Dry run completed successfully!" if self.dry_run.get() else "Experiment completed successfully!"
                self.root.after(0, lambda: messagebox.showinfo("Success", success_msg))
            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showerror("Error", f"Experiment failed: {err}"))
            finally:
                self.root.after(0, self.stop_progress_polling)

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

        start_msg = "Dry run started! Check console for details." if self.dry_run.get() else "Experiment started! Check console for progress."
        messagebox.showinfo("Running", start_msg)

    def run_onoff_experiment(self):
        """Import and run ON/OFF cycling experiment"""
        from experiment_runners import run_onoff_experiment
        run_onoff_experiment(self.config, dry_run=self.dry_run.get(), progress=self.progress_tracker)

    def run_ramp_experiment(self):
        """Import and run field ramp experiment"""
        from experiment_runners import run_ramp_experiment
        run_ramp_experiment(self.config, dry_run=self.dry_run.get(), progress=self.progress_tracker)

    def run_custom_experiment(self):
        """Import and run custom field sequence experiment"""
        from experiment_runners import run_custom_experiment
        run_custom_experiment(self.config, dry_run=self.dry_run.get(), progress=self.progress_tracker)

    def create_progress_window(self):
        """Create a progress monitoring window"""
        if self.progress_window is not None:
            return  # Already exists

        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Experiment Progress")
        self.progress_window.geometry("520x480")  # Increased height to show all content
        self.progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing during experiment

        # Main container
        main_frame = ttk.Frame(self.progress_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Experiment in Progress", font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))

        # Progress bar
        progress_frame = ttk.LabelFrame(main_frame, text="Overall Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 15))

        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_pct_label = ttk.Label(progress_frame, text="0.0%", font=('Arial', 12, 'bold'))
        self.progress_pct_label.pack()

        # Time information
        time_frame = ttk.LabelFrame(main_frame, text="Timing", padding="10")
        time_frame.pack(fill=tk.X, pady=(0, 15))

        self.time_elapsed_label = ttk.Label(time_frame, text="Elapsed: 00:00:00", font=('Arial', 10))
        self.time_elapsed_label.pack(anchor=tk.W, pady=2)

        self.time_remaining_label = ttk.Label(time_frame, text="Remaining: --:--:--", font=('Arial', 10))
        self.time_remaining_label.pack(anchor=tk.W, pady=2)

        self.time_finish_label = ttk.Label(time_frame, text="Estimated finish: --:--:--", font=('Arial', 10))
        self.time_finish_label.pack(anchor=tk.W, pady=2)

        # Frame progress
        frame_frame = ttk.LabelFrame(main_frame, text="Frame Progress", padding="10")
        frame_frame.pack(fill=tk.X, pady=(0, 15))

        self.frame_progress_label = ttk.Label(frame_frame, text="Frame: 0 / 0", font=('Arial', 10))
        self.frame_progress_label.pack()

        # Current status
        status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True)

        # Magnetic field display
        field_label_title = ttk.Label(status_frame, text="Magnetic Field:", font=('Arial', 10, 'bold'))
        field_label_title.pack(anchor=tk.W, pady=(0, 5))

        self.field_display_label = ttk.Label(status_frame, text="0.00 mT (0.00, 0.00, 0.00)",
                                             font=('Arial', 11), foreground='#2C5F8D')
        self.field_display_label.pack(anchor=tk.W, padx=10)

        # LED status display
        led_status_frame = ttk.Frame(status_frame)
        led_status_frame.pack(anchor=tk.W, pady=(10, 0))

        led_label_title = ttk.Label(led_status_frame, text="LED Status:", font=('Arial', 10, 'bold'))
        led_label_title.pack(side=tk.LEFT, padx=(0, 10))

        self.led_display_canvas = tk.Canvas(led_status_frame, width=80, height=30,
                                            highlightthickness=2, highlightbackground='black')
        self.led_display_canvas.pack(side=tk.LEFT)

        # Draw initial OFF state
        self.led_display_canvas.create_rectangle(2, 2, 78, 28, fill='#D3D3D3', outline='black', width=2, tags='led_rect')
        self.led_display_canvas.create_text(40, 15, text='OFF', font=('Arial', 10, 'bold'), tags='led_text')

    def close_progress_window(self):
        """Close the progress monitoring window"""
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
            self.progress_bar = None
            self.progress_pct_label = None
            self.time_elapsed_label = None
            self.time_remaining_label = None
            self.time_finish_label = None
            self.frame_progress_label = None
            self.field_display_label = None
            self.led_display_canvas = None

    def start_progress_polling(self):
        """Start polling progress and updating display"""
        self.experiment_start_time = time.time()
        self.create_progress_window()
        self.poll_progress()

    def stop_progress_polling(self):
        """Stop polling progress"""
        if self.progress_poll_id:
            self.root.after_cancel(self.progress_poll_id)
            self.progress_poll_id = None
        # Clear visual indicators
        if self.progress_line:
            self.progress_line.remove()
            self.progress_line = None
        if self.field_text:
            self.field_text.remove()
            self.field_text = None
        if self.led_indicator:
            self.led_indicator.remove()
            self.led_indicator = None
        self.experiment_canvas.draw()
        # Close progress window
        self.close_progress_window()
        self.experiment_start_time = None

    def poll_progress(self):
        """Poll progress tracker and update display"""
        if not self.progress_tracker:
            return

        state = self.progress_tracker.get_state()

        # Always update display if we have a progress tracker
        self.update_progress_display(state)

        # Continue polling - keep window open until manually stopped
        self.progress_poll_id = self.root.after(50, self.poll_progress)  # 20 Hz

    def update_progress_display(self, state):
        """Update visual progress indicators"""
        # Update progress window widgets if they exist
        if self.progress_bar and self.progress_window and self.progress_window.winfo_exists():
            progress_pct = state['progress_pct']
            self.progress_bar['value'] = progress_pct
            self.progress_pct_label.config(text=f"{progress_pct:.1f}%")

            # Calculate time information
            elapsed_time = state['elapsed_time']
            total_time = state['total_time']
            remaining_time = max(0, total_time - elapsed_time)

            # Format elapsed time
            elapsed_td = timedelta(seconds=int(elapsed_time))
            elapsed_str = str(elapsed_td)
            self.time_elapsed_label.config(text=f"Elapsed: {elapsed_str}")

            # Format remaining time
            remaining_td = timedelta(seconds=int(remaining_time))
            remaining_str = str(remaining_td)
            self.time_remaining_label.config(text=f"Remaining: {remaining_str}")

            # Calculate estimated finish time
            if self.experiment_start_time:
                finish_time = datetime.now() + timedelta(seconds=remaining_time)
                finish_str = finish_time.strftime("%H:%M:%S")
                self.time_finish_label.config(text=f"Estimated finish: {finish_str}")

            # Update frame progress
            current_frame = state['current_frame']
            total_frames = state['total_frames']
            self.frame_progress_label.config(text=f"Frame: {current_frame} / {total_frames}")

            # Update magnetic field display
            Bx, By, Bz = state['current_field']
            B_mag = np.linalg.norm(state['current_field'])
            field_str = f"{B_mag:.2f} mT ({Bx:.2f}, {By:.2f}, {Bz:.2f})"
            self.field_display_label.config(text=field_str)

            # Update LED indicator
            if self.led_display_canvas:
                if state['led_on']:
                    # Blue when on
                    self.led_display_canvas.itemconfig('led_rect', fill='#4A90E2')
                    self.led_display_canvas.itemconfig('led_text', text='ON')
                else:
                    # Gray when off
                    self.led_display_canvas.itemconfig('led_rect', fill='#D3D3D3')
                    self.led_display_canvas.itemconfig('led_text', text='OFF')

        # Update experiment preview graph with progress indicators
        # Remove old indicators
        if self.progress_line:
            self.progress_line.remove()
        if self.field_text:
            self.field_text.remove()
        if self.led_indicator:
            self.led_indicator.remove()

        # Add semi-transparent progress overlay on experiment preview
        current_time = state['elapsed_time']
        if current_time > 0:
            # Add vertical line showing current position
            self.progress_line = self.experiment_ax.axvline(current_time, color='red', linewidth=2.5,
                                                            linestyle='-', alpha=0.7, zorder=100)

            # Add semi-transparent overlay for completed portion (light blue)
            xlim = self.experiment_ax.get_xlim()
            ylim = self.experiment_ax.get_ylim()
            from matplotlib.patches import Rectangle
            completed_rect = Rectangle((xlim[0], ylim[0]), current_time - xlim[0], ylim[1] - ylim[0],
                                      facecolor='#87CEEB', alpha=0.15, zorder=1)  # Light blue (Sky Blue)
            self.experiment_ax.add_patch(completed_rect)

        # Add progress text at top (inside plot area)
        Bx, By, Bz = state['current_field']
        B_mag = np.linalg.norm(state['current_field'])
        progress_pct = state['progress_pct']
        field_str = f"{progress_pct:.1f}% | {B_mag:.2f} mT"

        # Add compact text at top of plot (position inside axes)
        self.field_text = self.experiment_ax.text(0.5, 0.97, field_str, transform=self.experiment_ax.transAxes,
                                                  ha='center', va='top', fontsize=9,
                                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))

        # Add LED indicator on graph
        if state['led_on']:
            led_color = '#4A90E2'  # Blue when on
            led_text = 'LED'
        else:
            led_color = '#D3D3D3'  # Gray when off
            led_text = 'LED'

        # Add small LED box in corner (position inside axes)
        self.led_indicator = self.experiment_ax.text(0.97, 0.97, led_text, transform=self.experiment_ax.transAxes,
                                                     ha='right', va='top', fontsize=8,
                                                     bbox=dict(boxstyle='round', facecolor=led_color,
                                                              edgecolor='black', linewidth=1.5, alpha=0.9, pad=0.3))

        self.experiment_canvas.draw()


# =========================
# MAIN
# =========================

def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
