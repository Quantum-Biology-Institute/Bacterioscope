# Bacterioscope — Control & Acquisition Code

Software for the **bacterioscope**: a magnetofluorescence microscope that exposes
live bacteria to programmable 3-axis magnetic fields while acquiring synchronized
fluorescence image stacks.

The codebase is small and self-contained. It is organized as a set of focused
Tkinter GUIs that all share two underlying modules:

- **`experiment_runners.py`** — the experiment engine (waveform generation, DAQ
  threads, Micro-Manager acquisition, metadata writing).
- **`hardware_io.py`** — thin wrappers around the NI-DAQ coils, the F71
  magnetometer (DAQ analog input + SCPI axis switching), and the calibration
  matrix.

Everything else is either a GUI that calls into those modules, calibration data,
or installation specs.

---

## 1. Folder layout

```
Bacterioscope_code_share/
├── README.md                      ← this file
├── environment.yml                ← conda environment spec
├── requirements.txt               ← pip dependency spec (alternative to conda)
│
├── experiment_gui.py              ← MAIN GUI: runs ON/OFF, ramp, or custom experiments
├── experiment_runners.py          ← Experiment engine used by experiment_gui.py
│
├── calibration_gui.py             ← Calibrate the 3-axis coils → produces calibration_M.npy
├── field_test_gui.py              ← Static field validation (one target at a time)
├── field_behavior_gui.py          ← Dynamic field validation (rotation / ramp playback)
├── coil_test_gui.py               ← Bare-metal manual coil voltage control
├── hardware_io.py                 ← Shared DAQ / F71 / calibration helpers
│
├── calibration_M.npy              ← 3×3 matrix: B_mT = M · V_volts  (current calibration)
├── calibration_B0.npy             ← Measured ambient field (mT) at V = 0
│
├── QBI_logo.png                   ← Logo displayed in experiment_gui.py
└── examples/
    └── sample_field_sequence.csv  ← Example CSV for the "Custom" experiment mode
```

There are **seven Python files** and **two NumPy calibration files**. Every file
on disk is described, in detail, in §3 below.

---

## 2. Installation

### Hardware prerequisites

1. **NI-DAQmx driver** installed on the host PC (download from National
   Instruments; the Python `nidaqmx` package only provides bindings, not the
   driver itself).
2. **Micro-Manager** installed and running. The camera must be configurable
   for **External Trigger** mode.
3. **Fiji** for automatic post-acquisition visualization (optional but
   recommended). Path is set in the experiment GUI.
4. **F71 magnetometer** (calibration only) reachable both as a DAQ analog
   input and as a serial/VISA SCPI device.

### Software environment

**Option A — conda (recommended):**

```bash
conda env create -f environment.yml
conda activate bacterioscope
```

**Option B — pip + venv:**

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Verify

```bash
python -c "import numpy, matplotlib, nidaqmx, pycromanager; print('OK')"
```

`pyvisa` is only required if you use the automatic (SCPI) F71 readout mode in
`calibration_gui.py` / `field_behavior_gui.py`. Install with `pip install pyvisa`
if needed.

---

## 3. What each file does

This section is the in-depth tour. Each entry covers: **purpose**, **how to
launch / use it**, and **key internals** so a reader of the paper can map the
methods text directly onto the code.

### 3.1 `experiment_runners.py` — experiment engine

The non-GUI core that actually performs an experiment. Imported by
`experiment_gui.py`. It contains:

- **`ExperimentProgress`** — thread-safe shared state used by the GUI to poll
  the running experiment (elapsed time, current frame, current field, LED
  state).
- **`load_calibration_matrix(path)`** — loads & validates the 3×3 matrix
  `M` (`B_mT = M · V_volts`) and returns `(M, M_inv)`. Falls back to an
  identity-style mapping with a printed warning if no calibration is present.
- **`map_field_to_voltage_with_limits(...)`** — converts a desired field
  vector into coil voltages; if the request exceeds the AO range, the vector
  is uniformly down-scaled and flagged as **saturated** in the per-frame
  metadata.
- **Waveform builders** (one per experiment type):
  - `generate_onoff_waveforms` — square-wave ON/OFF schedule.
  - `generate_ramp_waveforms` — staircase ramp (max→min→max or min→max→min)
    with optional OFF segments between levels.
  - `generate_custom_waveforms` — arbitrary field sequence loaded from CSV.
  
  All three render arrays at **10 kHz** on a common time grid. Three outputs
  are produced:
  - `ao_mag` (3×N): voltages for `Dev3/AO0..AO2` (X, Y, Z coils).
  - `ao_led` (1×N): TTL gate for the LED on `Dev2/AO0`.
  - `ao_cam` (1×N): TTL trigger to the camera on `Dev2/AO1`.

- **`run_onoff_experiment` / `run_ramp_experiment` / `run_custom_experiment`**
  — orchestrators. They:
  1. Build the waveforms and a per-frame metadata table.
  2. Start two threads: a **DAQ thread** that plays the finite waveforms via
     NI-DAQmx, and an **acquisition thread** that drives Micro-Manager.
  3. Inject per-frame metadata (`frame_index`, `planned_time_s`, `field_on`,
     `magneticField_mT`, `coilVoltages_V`) into the saved OME-TIFF
     `userData`.
  4. Write `experiment_summary.json` with the full configuration plus
     calibration info, device map, and the per-frame table.
  5. Launch Fiji on the resulting dataset.
  6. Zero the DAQ outputs in `finally` blocks (no residual current).

- **`load_field_sequence_from_csv(path)`** — reads `Bx,By,Bz` rows from a
  CSV (header optional) and returns the list used by the custom-mode runner.
  See `examples/sample_field_sequence.csv` for the expected format.

**Key timing rule:** all indices are computed as
`int(np.round(time_s * sample_rate))` to avoid cumulative rounding drift.
The very last sample is always forced to TTL-low so coils and LED cleanly
shut off.

### 3.2 `experiment_gui.py` — the main experiment GUI

The interface used for day-to-day experiments. Launch with:

```bash
python experiment_gui.py
```

Features:

- **Three experiment types** (radio buttons):
  - *ON/OFF Cycling* — baseline period followed by N ON/OFF cycles at a
    fixed target field.
  - *Field Ramp* — staircase of N field magnitudes, with OFF segments
    optionally interleaved.
  - *Custom* — arbitrary field sequence loaded from a CSV
    (`Bx,By,Bz` columns).
- **Live previews** (matplotlib): an acquisition-timing preview (LED vs
  camera pulses for one snapshot) and an experiment-timeline preview
  (field magnitude vs time).
- **Live duration / frame-count** display that updates as parameters change.
- **Save / Load Config** — round-trips the full parameter set to JSON.
- **Dry Run** — prints the configuration, field windows / levels, and total
  duration to the console without touching any hardware. *Always run this
  first when you change parameters.*
- **Threaded execution** — the experiment runs in a background thread so
  the GUI remains responsive; an `ExperimentProgress` object reports back.

Default device assignments (overridable in the GUI):

| Signal              | Device / channel | Notes                       |
|---------------------|------------------|-----------------------------|
| Coil X              | `Dev3/AO0`       | ±10 V                       |
| Coil Y              | `Dev3/AO1`       | ±10 V                       |
| Coil Z              | `Dev3/AO2`       | ±10 V                       |
| LED gate            | `Dev2/AO0`       | 0/5 V TTL                   |
| Camera trigger      | `Dev2/AO1`       | 0/5 V TTL → SMA #1          |

Output for each run is a timestamped folder containing the OME-TIFF stack and
`experiment_summary.json`. Fiji opens automatically.

### 3.3 `calibration_gui.py` — coil calibration

Drives a sweep of test voltage vectors on the coils, reads the F71 magnetometer
per axis, and least-squares fits the 3×3 matrix `M` such that

> **B_mT = M · V_volts**

Launch:

```bash
python calibration_gui.py
```

Workflow:

1. Set the **Max |V| per coil** for the sweep (typical: 2 V).
2. Choose the readout mode:
   - **SCPI (auto)** — the GUI switches the F71's analog output to X / Y / Z
     itself via pyvisa. Fastest, requires `pyvisa` installed and the F71
     reachable on a serial resource (`ASRL18::INSTR` by default).
   - **Manual** — the GUI runs three passes; between passes it asks you to
     turn the F71 knob to the next axis and click **continue**.
3. Click **Start Calibration**. A background thread writes each test
   voltage, settles for 1 s, reads the field, and updates the live plots.
4. When the sweep finishes, the GUI auto-fits `M`, the ambient field offset
   `B0` (taken from `V = 0` rows), and reports:
   - per-axis RMSE,
   - per-coil column norm in mT/V,
   - `cond(M)` (a high value indicates weak or co-linear coils),
   - the maximum |B| achievable on each world axis without saturating any
     coil at ±10 V.
5. Click **Save** to write `calibration_M.npy` and `calibration_B0.npy`. Any
   pre-existing `calibration_M.npy` is auto-backed-up with a timestamp.

The test protocol (`build_protocol`) includes the zero vector, single-axis
sweeps at ±max and ±max/2, and two-axis combinations at ±max/2 — enough rank
to solve `M` robustly while keeping the run short.

### 3.4 `field_test_gui.py` — static field validation

A small one-screen GUI: enter `(Bx, By, Bz)` in mT, click **Compute & Apply**.
The GUI multiplies by `M⁻¹` (using the loaded calibration) and writes the
resulting voltages to the coils. Read the F71 manually to verify the field
matches the request.

```bash
python field_test_gui.py
```

Use this to spot-check a freshly-fitted calibration on a few hand-picked
targets before committing to a long experiment.

Options:

- **Browse / Reload** — pick a different `.npy` calibration file.
- **Subtract ambient B0** — if you want the coils to *cancel* the loaded
  ambient field, enable this so the applied voltage compensates for `B0`.

Out-of-range targets (any |V| > 10 V) are blocked with a red warning and
**not** applied.

### 3.5 `field_behavior_gui.py` — dynamic field validation

Drives a *time-varying* field through the coils and simultaneously records
what the F71 sees, then plots measured vs. commanded for each axis and
reports RMSE.

```bash
python field_behavior_gui.py
```

Two built-in behaviors:

- **Rotating field** in the XY, YZ, or XZ plane at a chosen frequency and
  amplitude for a chosen duration.
- **Linear ramp** along one axis between two values over a chosen duration.

Each run does three passes (one per F71 axis), because the F71 only outputs
one axis at a time on its analog port. Pass-to-pass axis switching is either
automatic (SCPI) or manual.

Timing model: a fixed update rate (5–40 Hz). The loop writes a sample, sleeps
until the next scheduled tick, then reads the F71 — so the achieved cadence
is at most the chosen update rate. Frequency is capped at
`update_rate / 6` so each rotation cycle has enough sample points to be
meaningful.

Use this when you need to confirm that the coils faithfully reproduce the
dynamic field used in an experiment (e.g. before publishing a rotating-field
result).

### 3.6 `coil_test_gui.py` — manual coil voltage

The lowest-level GUI: three sliders + entries, one per coil, that latch a DC
voltage on `Dev3/AO0..AO2`. Use it for hardware bring-up, sanity-checking
amplifier polarity, or measuring static fields by hand at known voltages.

```bash
python coil_test_gui.py
```

No calibration involved — voltages go straight to the coils.

### 3.7 `hardware_io.py` — shared hardware helpers

Used by every GUI in §3.4–§3.6. Provides:

- **`CoilDriver`** — a persistent 3-channel NI-DAQ analog-output task that
  keeps the coil voltages latched between writes (writes happen on demand,
  not continuously).
- **`F71Analog`** — reads the F71 magnetometer via DAQ AI (`Dev2/ai0` by
  default), applies the V→mT correction factor, returns mT.
- **`F71SCPI`** — opens a pyvisa serial session and switches the F71's
  analog output between X / Y / Z.
- **`load_calibration(path)`** → `(M, M_inv)`.
- **`load_b0(path)`** → the ambient field vector, or `None` if absent.
- **`voltage_for_field(B_mT, M_inv, B0=None)`** → coil voltages required to
  realize `B_mT` (optionally pre-subtracting an ambient `B0`).

All defaults (device names, F71 VISA resource, V/mT correction) are module
constants at the top of the file; change them there if your hardware
configuration differs.

### 3.8 Calibration data files

- **`calibration_M.npy`** — 3×3 float array. Loaded by every GUI that needs
  to convert a field request into coil voltages. Re-generated by
  `calibration_gui.py`.
- **`calibration_B0.npy`** — 3-vector (mT). The ambient field measured at
  `V = 0` during the most recent calibration. Optional; only used when a
  GUI's "subtract ambient B0" option is enabled.

### 3.9 `examples/sample_field_sequence.csv`

A small example of the CSV format consumed by the experiment GUI's "Custom"
mode. One header row (`Bx,By,Bz`) followed by one (Bx, By, Bz) triplet per
field level, in mT. Each row becomes one ON segment in the timeline; OFF
segments between them are configured in the GUI.

### 3.10 `QBI_logo.png`

Used by `experiment_gui.py` purely for branding. Safe to swap or remove if
you don't want it shown — the GUI checks for its presence and skips it if
absent.

### 3.11 `environment.yml` / `requirements.txt`

Two equivalent installation manifests. `environment.yml` pins Python 3.11
and uses conda-forge for numpy/matplotlib (and pip for the
hardware-specific packages). `requirements.txt` is the pip-only
alternative. Both produce a working environment for every GUI in this
folder.

---

## 4. User guide — typical workflows

### 4.1 First-time setup on a new instrument

1. Install the environment (§2).
2. Power on the NI-DAQ chassis, the coil drivers, and the F71 magnetometer.
3. **Bring-up check** — launch `coil_test_gui.py`, write +1 V then -1 V to
   each coil in turn, and confirm with a handheld gauss-meter that the
   field changes sign as expected. This verifies the wiring without any
   calibration assumptions.
4. **Calibrate** — see §4.2.
5. **Validate** — see §4.3.
6. **Run an experiment** — see §4.4.

### 4.2 Calibrating the coils

1. Mount the F71 probe at the sample position with all three axes aligned to
   the instrument frame.
2. Launch `python calibration_gui.py`.
3. Pick a **Max |V|** safely below the coil-driver limit (2 V is a common
   starting point — the field at 2 V tells you the coil constant, then you
   can re-run higher if you want a stronger fit far from zero).
4. Choose the readout mode (SCPI if your F71 is on a serial bus and pyvisa
   is installed; Manual otherwise).
5. Click **Start Calibration**.
6. When it finishes, read the **Fit results** block:
   - `cond(M)` should be **well under 20**. Higher values indicate that two
     coils are too closely aligned or one is much weaker than the others.
   - Per-axis **RMSE** should be a small fraction of the largest measured
     |B|.
7. Click **Save**. This writes `calibration_M.npy` and `calibration_B0.npy`
   into the working directory; any pre-existing `calibration_M.npy` is
   auto-backed-up with a timestamp.

### 4.3 Validating the calibration

**Static validation (`field_test_gui.py`):**

1. Launch the GUI; the saved calibration loads automatically.
2. Enter a handful of targets — at minimum: (1, 0, 0), (0, 1, 0), (0, 0, 1),
   and one combined like (1, 1, 1) mT.
3. For each, click **Compute & Apply**, read the F71, and confirm the three
   axis readings match the requested field to within a few percent.

**Dynamic validation (`field_behavior_gui.py`):**

1. Launch the GUI; the saved calibration loads automatically.
2. Try a slow XY rotation (e.g. 1 Hz, 1 mT, 5 s) first. After the three
   passes finish, the GUI plots measured vs. predicted and prints RMSE.
3. Increase the frequency or amplitude until you see the achievable field
   bandwidth of the system.

### 4.4 Running an experiment

1. Start Micro-Manager. Configure the camera for **External Trigger** mode.
2. Launch `python experiment_gui.py`.
3. Pick an experiment type:
   - **ON/OFF Cycling** — baseline duration, number of cycles, dwell per
     state, target field vector.
   - **Field Ramp** — direction vector, step size in mT, segment duration,
     ramp direction (down-first or up-first), and whether to interleave OFF
     segments.
   - **Custom** — click **Browse...** under *Field Sequence (CSV)* and
     point at a CSV like `examples/sample_field_sequence.csv`.
4. Set the imaging timing (snap interval, LED advance, LED pulse, exposure)
   and the sample name / save directory.
5. Check **Dry Run**, click **Run Experiment**, and read the printed
   timeline to confirm it matches your intent.
6. Uncheck **Dry Run** and click **Run Experiment** again. The DAQ and
   Micro-Manager threads start; progress is logged to the console.
7. When the run completes, Fiji opens on the resulting OME-TIFF stack and
   the timestamped folder contains `experiment_summary.json` with full
   per-frame metadata.

---

## 5. Data output

Each experiment run creates a timestamped folder under the configured save
directory. It contains:

- The **OME-TIFF stack** written by Micro-Manager.
- **`experiment_summary.json`** with:
  - The full configuration (every field of the dataclass).
  - The calibration matrix in use plus the calibration file path.
  - Device channel assignments.
  - A per-frame table: `frame_index`, `planned_time_s`, `field_on`
    (bool), `magneticField_mT` (3-vector), `coilVoltages_V` (3-vector),
    and a `saturated` flag if any voltage was clipped.

The per-frame metadata is *also* embedded into each TIFF frame's `userData`
during acquisition, so the same information survives in the image stack
itself.

---

## 6. Troubleshooting

| Symptom                                     | Likely cause / fix |
|---------------------------------------------|--------------------|
| `nidaqmx not installed` / `Device not found`| Install the **NI-DAQmx driver** (separate from the Python package) and verify device names (`Dev2`, `Dev3`) in NI MAX. |
| `pyvisa not installed` (SCPI mode)          | `pip install pyvisa`. Or switch the calibration / behavior GUI to **Manual** mode. |
| Micro-Manager connection fails              | Launch Micro-Manager *before* the experiment GUI. Confirm the pycromanager version matches your MM build. |
| Calibration warning at experiment start     | `calibration_M.npy` is missing. Run `calibration_gui.py` and save it. The experiment will otherwise run with a 1 mT → 1 V dummy mapping (incorrect physical fields). |
| Frames dropped / timing drift               | Make sure `snap_interval_s` > camera exposure + readout. Check CPU load. All time-to-sample conversions in `experiment_runners.py` use `int(np.round(t * fs))` to avoid cumulative drift — don't change that. |
| Field saturates (`saturated: true` in JSON) | The requested |B| exceeds what the coils can deliver within ±10 V. The runner has already uniformly down-scaled the vector; reduce the requested field or improve the coil drive. |
| Static validation off by tens of percent    | Re-run calibration with the probe more carefully aligned to the instrument frame; large misalignment shows up as large off-diagonal entries in `M`. |

---

## 7. Hardware reference

Default device map (defined in `experiment_runners.py` and `hardware_io.py`,
overridable from each GUI):

```
Dev3   (coil drivers, ±10 V)
  AO0  → X coil
  AO1  → Y coil
  AO2  → Z coil

Dev2   (LED, camera, magnetometer)
  AO0  → LED gate     (0/5 V TTL)
  AO1  → Camera trig  (0/5 V TTL → SMA #1)
  AI0  → F71 magnetometer analog out (V/mT scale set in hardware_io.py)

ASRL18::INSTR  → F71 SCPI (calibration & dynamic-validation GUIs only)
```

Calibration identity: **`B_mT = M · V_volts`**, with `M ∈ ℝ^{3×3}` stored as
`calibration_M.npy`. Inverse mapping (used at runtime to realize a target
field): **`V = M⁻¹ · B_mT`**, optionally with `B0` pre-subtracted from the
target.
