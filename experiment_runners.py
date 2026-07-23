"""
Experiment Runners
Extracted core experiment logic from notebooks for GUI integration
"""

import numpy as np
import math
import threading
import json
import csv
from pathlib import Path
from typing import Tuple, List, Optional, Callable
from datetime import datetime
import subprocess
import time
import re

import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_writers import AnalogMultiChannelWriter

from pycromanager import Acquisition, multi_d_acquisition_events, Core


# =========================
# PROGRESS TRACKING
# =========================

class ExperimentProgress:
    """Shared state for tracking experiment progress"""
    def __init__(self):
        self.lock = threading.Lock()
        self.elapsed_time = 0.0
        self.total_time = 0.0
        self.current_field = (0.0, 0.0, 0.0)
        self.led_on = False
        self.is_running = False
        self.current_frame = 0
        self.total_frames = 0

    def update(self, elapsed_time=None, current_field=None, led_on=None, current_frame=None):
        """Thread-safe update of progress state"""
        with self.lock:
            if elapsed_time is not None:
                self.elapsed_time = elapsed_time
            if current_field is not None:
                self.current_field = current_field
            if led_on is not None:
                self.led_on = led_on
            if current_frame is not None:
                self.current_frame = current_frame

    def get_state(self):
        """Thread-safe get current state"""
        with self.lock:
            return {
                'elapsed_time': self.elapsed_time,
                'total_time': self.total_time,
                'current_field': self.current_field,
                'led_on': self.led_on,
                'is_running': self.is_running,
                'current_frame': self.current_frame,
                'total_frames': self.total_frames,
                'progress_pct': (self.elapsed_time / self.total_time * 100) if self.total_time > 0 else 0
            }

    def start(self, total_time, total_frames):
        """Mark experiment as started"""
        with self.lock:
            self.is_running = True
            self.total_time = total_time
            self.total_frames = total_frames
            self.elapsed_time = 0.0
            self.current_frame = 0

    def stop(self):
        """Mark experiment as stopped"""
        with self.lock:
            self.is_running = False


# =========================
# CALIBRATION
# =========================

def load_calibration_matrix(path: str):
    """Load calibration matrix and return M and M_inv"""
    try:
        M = np.load(path)
        if M.shape != (3, 3):
            raise ValueError(f"Calibration matrix must be 3×3, got {M.shape}")
        Minv = np.linalg.inv(M)
        return M, Minv
    except Exception as e:
        print(f"[CAL] WARNING: failed to load '{path}': {e}\n       Using dummy mapping (1 mT → 1 V).")
        return None, None


def field_mT_to_voltage_V(BmT: Tuple[float, float, float], Minv) -> Tuple[float, float, float]:
    """Convert magnetic field (mT) to voltage (V) using calibration"""
    B = np.array(BmT, dtype=float).reshape(3,)
    if Minv is None:
        return tuple(B.tolist())  # dummy: 1 mT -> 1 V
    return tuple((Minv @ B).tolist())


def map_field_to_voltage_with_limits(BmT, vmin, vmax, headroom, M, Minv):
    """Map B (mT)->V (V); scale if any component exceeds range"""
    B = np.array(BmT, dtype=float).reshape(3,)
    if Minv is None:
        V_req = B.copy()
    else:
        V_req = Minv @ B

    Vpos = vmax - headroom
    Vneg = vmin + headroom
    Vlim = min(Vpos, -Vneg)

    max_abs = np.max(np.abs(V_req)) if Vlim > 0 else np.inf
    scale = 1.0
    saturated = False
    if max_abs > Vlim + 1e-12:
        scale = Vlim / max_abs
        saturated = True

    V_used = V_req * scale
    if M is None:
        B_ach = B * scale
    else:
        B_ach = M @ V_used

    return {
        "V_used": tuple(V_used.tolist()),
        "B_ach": tuple(B_ach.tolist()),
        "scale": float(scale),
        "saturated": bool(saturated),
        "V_req": tuple(V_req.tolist()),
        "B_req": tuple(B.tolist()),
    }


# =========================
# COMPREHENSIVE METADATA BUILDER
# =========================

def build_comprehensive_metadata(config, experiment_type: str, M, Minv) -> dict:
    """Build comprehensive metadata dictionary from config object.

    Includes all experiment parameters for complete reproducibility.
    """
    # Common parameters present in all config types
    metadata = {
        # Experiment identification
        "experiment_type": experiment_type,
        "sample_name": config.sample_name,
        "notes": getattr(config, 'notes', ''),

        # Timing parameters
        "timing": {
            "snap_interval_s": config.snap_interval_s,
            "led_advance_s": config.led_advance_s,
            "led_pulse_s": config.led_pulse_s,
            "trig_pulse_s": config.trig_pulse_s,
            "sample_rate_hz": config.sample_rate,
        },

        # LED settings
        "led_settings": {
            "intensity_mA": config.led_intensity_mA,
            "pulse_duration_s": config.led_pulse_s,
            "advance_before_trigger_s": config.led_advance_s,
            "ttl_high_v": config.ttl_high_v,
            "ttl_low_v": config.ttl_low_v,
        },

        # Camera settings
        "camera_settings": {
            "exposure_ms": config.mm_set_exposure_ms,
            "show_display": config.mm_show_display,
            "trigger_pulse_s": config.trig_pulse_s,
        },

        # DAQ device configuration
        "daq_configuration": {
            "magnet_device": config.dev_mag,
            "magnet_channels": list(config.ao_mag_chans),
            "control_device": config.dev_ao,
            "led_channel": config.ao_led_chan,
            "camera_trigger_channel": config.ao_cam_chan,
            "voltage_range": {
                "min_v": config.ao_vmin,
                "max_v": config.ao_vmax,
            },
            "headroom_v": config.ao_headroom_v,
        },

        # Calibration information
        "calibration": {
            "matrix_path": config.calibration_matrix_path,
            "matrix_loaded": M is not None,
            "calibration_matrix": M.tolist() if M is not None else None,
            "inverse_matrix": Minv.tolist() if Minv is not None else None,
        },

        # Safety parameters
        "safety": {
            "acq_timeout_margin_s": config.acq_timeout_margin_s,
            "daq_start_delay_s": config.daq_start_delay_s,
        },

        # File paths
        "paths": {
            "save_directory": config.save_dir,
            "fiji_executable": config.fiji_exe,
        },
    }

    return metadata


def add_onoff_specific_metadata(metadata: dict, config) -> dict:
    """Add ON/OFF cycling specific parameters to metadata."""
    metadata["magnetic_field_parameters"] = {
        "schedule_mode": config.schedule_mode,
        "target_field_mT": {
            "Bx": config.target_field_mT[0],
            "By": config.target_field_mT[1],
            "Bz": config.target_field_mT[2],
            "magnitude": float(np.linalg.norm(config.target_field_mT)),
        },
        "ambient_field_mT": {
            "Bx": config.ambient_field_mT[0],
            "By": config.ambient_field_mT[1],
            "Bz": config.ambient_field_mT[2],
            "magnitude": float(np.linalg.norm(config.ambient_field_mT)),
        },
        "baseline_duration_s": config.baseline_first_s,
        "number_of_cycles": config.cycles_n,
        "dwell_time_s": config.dwell_s,
    }
    return metadata


def add_ramp_specific_metadata(metadata: dict, config, levels, mapping_on) -> dict:
    """Add field ramp specific parameters to metadata."""
    Bmax_mag = float(np.linalg.norm(config.max_field_mT))

    # Check for saturation
    saturated_levels = [i for i, m in enumerate(mapping_on) if m['saturated']]

    metadata["magnetic_field_parameters"] = {
        "max_field_mT": {
            "Bx": config.max_field_mT[0],
            "By": config.max_field_mT[1],
            "Bz": config.max_field_mT[2],
            "magnitude": Bmax_mag,
        },
        "step_size_mT": config.step_mT,
        "ramp_direction_first": config.ramp_first,
        "start_with_off_segment": config.off_first,
        "baseline_duration_s": config.baseline_first_s,
        "segment_duration_s": config.segment_duration_s,
        "number_of_levels": len(levels),
        "unique_magnitudes_mT": sorted(set([float(np.linalg.norm(L)) for L in levels if np.linalg.norm(L) > 0]), reverse=True),
        "levels_sequence_mT": [{"Bx": L[0], "By": L[1], "Bz": L[2], "magnitude": float(np.linalg.norm(L))} for L in levels],
        "voltage_saturation": {
            "any_saturated": len(saturated_levels) > 0,
            "num_saturated_levels": len(saturated_levels),
            "saturated_level_indices": saturated_levels,
        },
    }
    return metadata


def add_custom_specific_metadata(metadata: dict, config, levels, mapping_on) -> dict:
    """Add custom field sequence specific parameters to metadata."""
    # Check for saturation
    saturated_levels = [i for i, m in enumerate(mapping_on) if m['saturated']]

    metadata["magnetic_field_parameters"] = {
        "csv_file_path": getattr(config, 'csv_file_path', ''),
        "baseline_duration_s": config.baseline_first_s,
        "segment_duration_s": config.segment_duration_s,
        "include_off_between_fields": config.off_between_fields,
        "number_of_field_points": len(levels),
        "field_sequence_mT": [{"Bx": L[0], "By": L[1], "Bz": L[2], "magnitude": float(np.linalg.norm(L))} for L in levels],
        "voltage_saturation": {
            "any_saturated": len(saturated_levels) > 0,
            "num_saturated_levels": len(saturated_levels),
            "saturated_level_indices": saturated_levels,
        },
    }
    return metadata


# =========================
# CSV FIELD LOADING
# =========================

def load_field_sequence_from_csv(csv_path: str) -> List[Tuple[float, float, float]]:
    """Load magnetic field sequence from CSV file.

    Expected CSV format:
    Bx,By,Bz
    1.0,0.0,0.0
    2.0,0.0,0.0
    ...

    Returns list of (Bx, By, Bz) tuples in mT.
    """
    fields = []
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)

        # Check for header
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV file is empty")

        # Try to parse first row - if it fails, it's probably a header
        try:
            bx, by, bz = float(first_row[0]), float(first_row[1]), float(first_row[2])
            fields.append((bx, by, bz))
        except (ValueError, IndexError):
            # First row is header, skip it
            pass

        # Read remaining rows
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 3:
                print(f"[CSV] Warning: Row {row_num} has fewer than 3 columns, skipping")
                continue
            try:
                bx = float(row[0].strip())
                by = float(row[1].strip())
                bz = float(row[2].strip())
                fields.append((bx, by, bz))
            except ValueError as e:
                print(f"[CSV] Warning: Could not parse row {row_num}: {row} - {e}")
                continue

    if not fields:
        raise ValueError("No valid field values found in CSV file")

    print(f"[CSV] Loaded {len(fields)} field points from {csv_path}")
    return fields


# =========================
# FIJI HELPERS
# =========================

def pick_newest(paths: List[Path]) -> Optional[Path]:
    """Return newest path by modification time"""
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_latest_image_under(folder: Path) -> Optional[Path]:
    """Find newest image file in folder"""
    patterns = ["*.ome.tif", "*.ome.tiff", "*MMStack*.ome.tif", "*.tif", "*.tiff"]
    candidates = []
    for pat in patterns:
        candidates.extend(folder.rglob(pat))
    return pick_newest(candidates) if candidates else None


def resolve_actual_run_dir(save_root: Path, run_base: str) -> Optional[Path]:
    """Find actual run directory (may have _1, _2, etc suffix from MM)"""
    rx = re.compile(r"^" + re.escape(run_base) + r"(?:_(\d+))?$")
    candidates = [p for p in save_root.iterdir() if p.is_dir() and rx.match(p.name)]
    return pick_newest(candidates)


def open_in_fiji(fiji_exe: Path, data_path: Path) -> None:
    """Launch Fiji with data"""
    if not fiji_exe.exists():
        print(f"[FIJI] Fiji not found at: {fiji_exe}")
        return
    target = data_path if data_path.is_file() else (find_latest_image_under(data_path) or data_path)
    try:
        subprocess.Popen([str(fiji_exe), str(target)])
        print(f"[FIJI] Launched Fiji with: {target}")
    except Exception as e:
        print(f"[FIJI] Launch failed: {e}")


# =========================
# RUN NAME GENERATION
# =========================

def sanitize_token(s: str) -> str:
    """Sanitize string for filename"""
    s = (s or "").strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s) or "sample"


def build_onoff_run_name(config) -> str:
    """Build run name for ON/OFF experiment"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample = sanitize_token(config.sample_name)
    B_mag = float(np.linalg.norm(config.target_field_mT))
    btag = f"{int(round(B_mag))}mT"
    return f"{ts}_{sample}_{btag}"


def build_ramp_run_name(config) -> str:
    """Build run name for ramp experiment"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample = sanitize_token(config.sample_name)
    ramp = "rampdown" if config.ramp_first.lower().startswith("down") else "rampup"
    first = "offfirst" if config.off_first else "onfirst"
    return f"{ts}_{sample}_MF_{ramp}_{first}"


def build_custom_run_name(config) -> str:
    """Build run name for custom field sequence experiment"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample = sanitize_token(config.sample_name)
    n_fields = len(getattr(config, 'field_sequence', [])) or 0
    return f"{ts}_{sample}_custom_{n_fields}pts"


# =========================
# ON/OFF CYCLING EXPERIMENT
# =========================

def build_onoff_field_windows(config):
    """Build field ON/OFF windows for cycling experiment"""
    t = 0.0
    w = []
    if config.schedule_mode.lower() == "start_on":
        w.append((0.0, config.baseline_first_s, True))
        t = config.baseline_first_s
        for _ in range(config.cycles_n):
            w.append((t, t + config.dwell_s, False))
            t += config.dwell_s
            w.append((t, t + config.dwell_s, True))
            t += config.dwell_s
    else:  # start_off
        w.append((0.0, config.baseline_first_s, False))
        t = config.baseline_first_s
        for _ in range(config.cycles_n):
            w.append((t, t + config.dwell_s, True))
            t += config.dwell_s
            w.append((t, t + config.dwell_s, False))
            t += config.dwell_s
    return w


def field_on_at(t: float, windows):
    """Check if field is ON at time t"""
    for (t0, t1, on) in windows:
        if t0 <= t < t1:
            return on
    return False


def total_duration(windows):
    """Get total duration from windows"""
    return max(t1 for (_, t1, _) in windows)


def build_snap_times(config, T: float):
    """Build snapshot times"""
    n = int(math.floor((T - 1e-12) / config.snap_interval_s)) + 1
    ts = [i * config.snap_interval_s for i in range(n)]
    return [t for t in ts if t < T - 1.0 / config.sample_rate]


def generate_onoff_waveforms(config, Minv):
    """Generate waveforms for ON/OFF experiment"""
    windows = build_onoff_field_windows(config)
    T = total_duration(windows)
    snap_ts = build_snap_times(config, T)

    sr = config.sample_rate
    N = int(np.round(T * sr)) + 1

    # Magnet AO 3×N
    V_on = np.array(field_mT_to_voltage_V(config.target_field_mT, Minv), dtype=float).reshape(3, 1)
    V_off = np.array(field_mT_to_voltage_V(config.ambient_field_mT, Minv), dtype=float).reshape(3, 1)

    t_axis = np.arange(N) / sr
    on_mask = np.array([field_on_at(t, windows) for t in t_axis], dtype=bool)[None, :]
    ao_mag = np.where(on_mask, V_on, V_off).astype(np.float64)

    # Dev2 AO TTL-like (2×N): row0=LED, row1=CAM
    ao2 = np.full((2, N), config.ttl_low_v, dtype=np.float64)

    def mark(arr: np.ndarray, t0: float, pw: float, high: float):
        i0 = int(np.round(t0 * sr))
        i1 = int(np.round((t0 + pw) * sr))
        i0 = max(0, min(N - 1, i0))
        i1 = max(0, min(N, i1))
        if i1 > i0:
            arr[i0:i1] = high

    for ts in snap_ts:
        mark(ao2[1], ts, config.trig_pulse_s, config.ttl_high_v)
        mark(ao2[0], max(0.0, ts - config.led_advance_s), config.led_pulse_s, config.ttl_high_v)

    ao2[:, -1] = config.ttl_low_v

    return ao_mag, ao2, T, snap_ts, windows


def build_onoff_frame_metadata(config, snap_ts, windows, Minv):
    """Build per-frame metadata for ON/OFF experiment"""
    meta = []
    for i, ts in enumerate(snap_ts):
        on = field_on_at(ts, windows)
        BmT = config.target_field_mT if on else config.ambient_field_mT
        V = field_mT_to_voltage_V(BmT, Minv)
        meta.append({
            "frame_index": int(i),
            "planned_time_s": float(ts),
            "field_on": bool(on),
            "magneticField_mT": {"Bx": float(BmT[0]), "By": float(BmT[1]), "Bz": float(BmT[2])},
            "coilVoltages_V": {"Vx": float(V[0]), "Vy": float(V[1]), "Vz": float(V[2])},
            "led_intensity_mA": float(config.led_intensity_mA),
        })
    return meta


def run_onoff_experiment(config, dry_run=False, progress: Optional[ExperimentProgress] = None):
    """Run ON/OFF cycling experiment

    Args:
        config: OnOffConfig object with experiment parameters
        dry_run: If True, generate waveforms and metadata but don't run hardware
        progress: Optional progress tracker for real-time updates
    """
    run_name = build_onoff_run_name(config)
    save_root = Path(config.save_dir)

    if not dry_run:
        save_root.mkdir(parents=True, exist_ok=True)

    # Load calibration
    M, Minv = load_calibration_matrix(config.calibration_matrix_path)

    # Generate waveforms
    ao_mag, ao2, T, snap_ts, windows = generate_onoff_waveforms(config, Minv)
    N = ao_mag.shape[1]
    sr = config.sample_rate
    frame_meta = build_onoff_frame_metadata(config, snap_ts, windows, Minv)

    mode_tag = "[DRY RUN]" if dry_run else "[RUN]"
    print(f"\n{mode_tag} {run_name}")
    print(f"{'='*80}")
    print(f"Experiment Type: ON/OFF Cycling")
    print(f"Schedule: {config.schedule_mode}")
    print(f"Total Duration: {T:.3f}s ({T/60:.1f} min)")
    print(f"Frames: {len(snap_ts)} @ {config.snap_interval_s}s interval")
    print(f"Target Field (ON): {config.target_field_mT} mT")
    print(f"Ambient Field (OFF): {config.ambient_field_mT} mT")
    print(f"Baseline: {config.baseline_first_s}s | Cycles: {config.cycles_n} | Dwell: {config.dwell_s}s")
    print(f"Waveform: {N} samples @ {sr} Hz")
    print(f"Save to: {save_root}\\{run_name}")

    if dry_run:
        print(f"\n[DRY RUN] Skipping hardware setup and execution")
        print(f"[DRY RUN] Would configure:")
        print(f"  - DAQ {config.dev_mag}: 3-axis magnet control ({config.ao_mag_chans})")
        print(f"  - DAQ {config.dev_ao}: LED gate ({config.ao_led_chan}) + Camera trigger ({config.ao_cam_chan})")
        print(f"  - Camera: External trigger, {config.mm_set_exposure_ms}ms exposure")
        print(f"\n[DRY RUN] Field windows:")
        for i, (t0, t1, on) in enumerate(windows):
            state = "ON " if on else "OFF"
            print(f"  {i:3d}. t=[{t0:7.1f}, {t1:7.1f}]s : {state}")
        print(f"\n[DRY RUN] Dry run complete - no hardware was used\n")
        print(f"{'='*80}\n")
        return

    print(f"{'='*80}\n")

    # Setup DAQ tasks
    mag_task = nidaqmx.Task("mag_ao_task")
    for ch in config.ao_mag_chans:
        mag_task.ao_channels.add_ao_voltage_chan(f"{config.dev_mag}/{ch}",
                                                  min_val=config.ao_vmin, max_val=config.ao_vmax)
    mag_task.timing.cfg_samp_clk_timing(rate=sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
    AnalogMultiChannelWriter(mag_task.out_stream, auto_start=False).write_many_sample(ao_mag)

    ao_task = nidaqmx.Task("dev2_ao_led_cam")
    ao_task.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_led_chan}",
                                            min_val=config.ao_vmin, max_val=config.ao_vmax)
    ao_task.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_cam_chan}",
                                            min_val=config.ao_vmin, max_val=config.ao_vmax)
    ao_task.timing.cfg_samp_clk_timing(rate=sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
    AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False).write_many_sample(ao2)

    # Configure camera
    try:
        core = Core()
        cam = core.get_camera_device()
        for prop, val in [("TriggerMode", "External"), ("Trigger Source", "External"),
                          ("trigger_mode", "External")]:
            try:
                core.set_property(cam, prop, val)
            except Exception:
                pass
        core.set_exposure(config.mm_set_exposure_ms)
        print("[MM] Camera set to External Trigger; exposure set.")
    except Exception as e:
        print(f"[MM] Could not set camera trigger via Core: {e}")

    # Initialize progress tracking
    if progress:
        progress.start(T, len(snap_ts))

    # DAQ thread with progress updates
    def run_daq():
        try:
            time.sleep(config.daq_start_delay_s)
            ao_task.start()
            mag_task.start()

            # Monitor progress periodically
            start_time = time.time()
            while mag_task.is_task_done() is False or ao_task.is_task_done() is False:
                elapsed = time.time() - start_time
                if progress and elapsed <= T:
                    # Determine current field and LED state
                    current_on = field_on_at(elapsed, windows)
                    current_field = config.target_field_mT if current_on else config.ambient_field_mT

                    # Check if LED should be on (within any LED pulse window)
                    led_is_on = False
                    for ts in snap_ts:
                        if elapsed >= max(0, ts - config.led_advance_s) and elapsed < (ts - config.led_advance_s + config.led_pulse_s):
                            led_is_on = True
                            break

                    progress.update(elapsed_time=elapsed, current_field=current_field, led_on=led_is_on)
                time.sleep(0.05)  # Update at 20 Hz

            mag_task.wait_until_done(timeout=T + 5.0)
            ao_task.wait_until_done(timeout=T + 5.0)
        finally:
            for t in (ao_task, mag_task):
                try:
                    t.stop()
                except Exception:
                    pass
                try:
                    t.close()
                except Exception:
                    pass
            # Zero outputs
            zero_daq_outputs(config)
            if progress:
                progress.stop()

    daq_thread = threading.Thread(target=run_daq, daemon=True)

    # MM acquisition thread
    acq_exc = []

    def run_acq():
        try:
            def ipf(image, metadata):
                idx = ipf.counter
                md = frame_meta[idx] if idx < len(frame_meta) else {"frame_index": int(idx)}
                ud = metadata.get("userData", {})
                ud.update(md)
                ud.setdefault("experiment_summary_file", "experiment_summary.json")
                metadata["userData"] = ud
                ipf.counter += 1
                # Update progress with current frame
                if progress:
                    progress.update(current_frame=ipf.counter)
                return image, metadata

            ipf.counter = 0

            events = multi_d_acquisition_events(num_time_points=len(snap_ts), time_interval_s=0)
            with Acquisition(directory=str(save_root), name=run_name,
                             show_display=config.mm_show_display, image_process_fn=ipf) as acq:
                acq.acquire(events)
        except Exception as e:
            acq_exc.append(e)

    acq_thread = threading.Thread(target=run_acq, daemon=True)

    # Start both
    acq_thread.start()
    daq_thread.start()
    deadline = time.time() + T + config.acq_timeout_margin_s
    while time.time() < deadline and (acq_thread.is_alive() or daq_thread.is_alive()):
        time.sleep(0.1)

    # Save summary with comprehensive metadata
    actual_run_dir = resolve_actual_run_dir(save_root, run_name)
    if actual_run_dir:
        # Build comprehensive metadata
        summary = build_comprehensive_metadata(config, "onoff_cycling", M, Minv)
        summary = add_onoff_specific_metadata(summary, config)

        # Add run-specific info
        summary["run_name"] = run_name
        summary["saved_path"] = str(actual_run_dir)
        summary["timestamp"] = datetime.now().isoformat()

        # Add duration and frame info
        summary["acquisition_info"] = {
            "total_duration_s": T,
            "total_duration_min": T / 60.0,
            "total_frames": len(snap_ts),
            "waveform_samples": N,
        }

        # Add field windows for reference
        summary["field_windows"] = [
            {"start_s": t0, "end_s": t1, "field_on": on}
            for t0, t1, on in windows
        ]

        # Add per-frame metadata
        summary["frames"] = frame_meta

        with open(actual_run_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[MM] Wrote summary: {actual_run_dir / 'experiment_summary.json'}")

        open_in_fiji(Path(config.fiji_exe), actual_run_dir)

    if acq_exc:
        raise acq_exc[0]


# =========================
# FIELD RAMP EXPERIMENT
# =========================

def build_ramp_levels(config):
    """Build magnitude levels for ramp"""
    Bmax_vec = np.array(config.max_field_mT, dtype=float)
    u = Bmax_vec / np.linalg.norm(Bmax_vec)
    Bmax_mag = float(np.linalg.norm(Bmax_vec))
    step = float(config.step_mT)

    n = int(np.floor(Bmax_mag / step))
    multiples = [k * step for k in range(n, 0, -1)]
    mags = [Bmax_mag]
    for m in multiples:
        if abs(m - Bmax_mag) > 1e-9:
            mags.append(m)
    mags = sorted(set(mags), reverse=True)

    if mags[-1] < step - 1e-9:
        mags.append(step)
        mags = sorted(set(mags), reverse=True)

    mags_asc = list(reversed(mags))

    if config.ramp_first.lower().startswith("down"):
        mags_full = mags + mags_asc
    else:
        mags_full = mags_asc + mags

    levels = [tuple((m * u).tolist()) for m in mags_full]
    return levels


def build_ramp_segments(config, levels):
    """Build segment list for ramp"""
    segs = []
    t = 0.0
    dur = config.segment_duration_s

    # Add baseline period
    segs.append((t, t + config.baseline_first_s, "baseline"))
    t += config.baseline_first_s

    if config.off_first:
        segs.append((t, t + dur, "off"))
        t += dur

    for i in range(len(levels)):
        segs.append((t, t + dur, i))
        t += dur
        segs.append((t, t + dur, "off"))
        t += dur

    return segs, t


def generate_ramp_waveforms(config, M, Minv):
    """Generate waveforms for ramp experiment"""
    levels = build_ramp_levels(config)
    segs, T = build_ramp_segments(config, levels)

    # Precompute mappings
    mapping_on = [map_field_to_voltage_with_limits(B, config.ao_vmin, config.ao_vmax,
                                                     config.ao_headroom_v, M, Minv) for B in levels]
    mapping_off = map_field_to_voltage_with_limits((0.0, 0.0, 0.0), config.ao_vmin,
                                                     config.ao_vmax, config.ao_headroom_v, M, Minv)

    sr = config.sample_rate
    N = int(np.round(T * sr)) + 1

    # Magnet AO
    ao_mag = np.zeros((3, N), dtype=np.float64)
    for (t0, t1, state) in segs:
        i0 = max(0, min(N - 1, int(np.round(t0 * sr))))
        i1 = max(0, min(N, int(np.round(t1 * sr))))
        if i1 <= i0:
            continue
        if state == "off" or state == "baseline":
            V = np.array(mapping_off["V_used"]).reshape(3, 1)
        else:
            V = np.array(mapping_on[state]["V_used"]).reshape(3, 1)
        ao_mag[:, i0:i1] = V

    # Dev2 AO
    ao2 = np.full((2, N), config.ttl_low_v, dtype=np.float64)

    num_snaps = int(math.floor((T - 1e-12) / config.snap_interval_s)) + 1
    snap_ts = [i * config.snap_interval_s for i in range(num_snaps)]
    snap_ts = [t for t in snap_ts if t < T - 1.0 / sr]

    def mark(arr: np.ndarray, t0: float, pw: float, high: float):
        i0 = int(np.round(t0 * sr))
        i1 = int(np.round((t0 + pw) * sr))
        i0 = max(0, min(N - 1, i0))
        i1 = max(0, min(N, i1))
        if i1 > i0:
            arr[i0:i1] = high

    for ts in snap_ts:
        mark(ao2[1], ts, config.trig_pulse_s, config.ttl_high_v)
        mark(ao2[0], max(0.0, ts - config.led_advance_s), config.led_pulse_s, config.ttl_high_v)

    ao2[:, -1] = config.ttl_low_v

    return ao_mag, ao2, T, snap_ts, segs, levels, mapping_on, mapping_off


def build_ramp_frame_metadata(config, snap_ts, segs, levels, mapping_on, mapping_off):
    """Build per-frame metadata for ramp experiment"""
    meta = []
    for i, ts in enumerate(snap_ts):
        state = "off"
        for (t0, t1, st) in segs:
            if t0 <= ts < t1:
                state = st
                break

        if state == "off" or state == "baseline":
            mapd = mapping_off
        else:
            mapd = mapping_on[state]

        B = mapd["B_ach"]
        V = mapd["V_used"]
        meta.append({
            "frame_index": int(i),
            "planned_time_s": float(ts),
            "field_on": (state != "off" and state != "baseline"),
            "magneticField_mT": {"Bx": float(B[0]), "By": float(B[1]), "Bz": float(B[2])},
            "coilVoltages_V": {"Vx": float(V[0]), "Vy": float(V[1]), "Vz": float(V[2])},
            "scale_factor": float(mapd["scale"]),
            "saturated": bool(mapd["saturated"]),
            "led_intensity_mA": float(config.led_intensity_mA),
        })
    return meta


def zero_daq_outputs(config):
    """Zero all DAQ outputs"""
    try:
        with nidaqmx.Task() as t2:
            t2.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_led_chan}",
                                               min_val=config.ao_vmin, max_val=config.ao_vmax)
            t2.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_cam_chan}",
                                               min_val=config.ao_vmin, max_val=config.ao_vmax)
            AnalogMultiChannelWriter(t2.out_stream, auto_start=True).write_many_sample(np.zeros((2, 1)))
    except Exception:
        pass
    try:
        with nidaqmx.Task() as tm:
            for ch in config.ao_mag_chans:
                tm.ao_channels.add_ao_voltage_chan(f"{config.dev_mag}/{ch}",
                                                   min_val=config.ao_vmin, max_val=config.ao_vmax)
            AnalogMultiChannelWriter(tm.out_stream, auto_start=True).write_many_sample(np.zeros((3, 1)))
    except Exception:
        pass


def run_ramp_experiment(config, dry_run=False, progress: Optional[ExperimentProgress] = None):
    """Run field ramp experiment

    Args:
        config: RampConfig object with experiment parameters
        dry_run: If True, generate waveforms and metadata but don't run hardware
        progress: Optional progress tracker for real-time updates
    """
    run_name = build_ramp_run_name(config)
    save_root = Path(config.save_dir)

    if not dry_run:
        save_root.mkdir(parents=True, exist_ok=True)

    # Load calibration
    M, Minv = load_calibration_matrix(config.calibration_matrix_path)

    # Generate waveforms
    ao_mag, ao2, T, snap_ts, segs, levels, mapping_on, mapping_off = generate_ramp_waveforms(config, M, Minv)
    N = ao_mag.shape[1]
    sr = config.sample_rate
    frame_meta = build_ramp_frame_metadata(config, snap_ts, segs, levels, mapping_on, mapping_off)

    Bmax_mag = float(np.linalg.norm(config.max_field_mT))
    mode_tag = "[DRY RUN]" if dry_run else "[RUN]"
    print(f"\n{mode_tag} {run_name}")
    print(f"{'='*80}")
    print(f"Experiment Type: Field Ramp")
    print(f"Ramp: {config.ramp_first} first | Start: {'OFF' if config.off_first else 'ON'}")
    print(f"|B|max={Bmax_mag:.3f} mT, step={config.step_mT:.3f} mT")
    print(f"Levels: {len(levels)} | Segments: {len(segs)}")
    print(f"Total Duration: {T:.3f}s ({T/60:.1f} min)")
    print(f"Frames: {len(snap_ts)} @ {config.snap_interval_s}s interval")
    print(f"Segment Duration: {config.segment_duration_s}s")
    print(f"Waveform: {N} samples @ {sr} Hz")
    print(f"Save to: {save_root}\\{run_name}")

    if dry_run:
        print(f"\n[DRY RUN] Skipping hardware setup and execution")
        print(f"[DRY RUN] Would configure:")
        print(f"  - DAQ {config.dev_mag}: 3-axis magnet control ({config.ao_mag_chans})")
        print(f"  - DAQ {config.dev_ao}: LED gate ({config.ao_led_chan}) + Camera trigger ({config.ao_cam_chan})")
        print(f"  - Camera: External trigger, {config.mm_set_exposure_ms}ms exposure")

        print(f"\n[DRY RUN] Field levels (magnitude in mT):")
        unique_mags = sorted(set([float(np.linalg.norm(L)) for L in levels if np.linalg.norm(L) > 0]), reverse=True)
        print(f"  Unique field strengths: {unique_mags}")

        print(f"\n[DRY RUN] Segment sequence (first 10):")
        for i, (t0, t1, state) in enumerate(segs[:10]):
            if state == "off":
                label = "OFF (0.0 mT)"
            else:
                B_vec = levels[state]
                B_mag = float(np.linalg.norm(B_vec))
                label = f"ON  ({B_mag:.2f} mT)"
            print(f"  {i:3d}. t=[{t0:7.1f}, {t1:7.1f}]s : {label}")
        if len(segs) > 10:
            print(f"  ... ({len(segs)-10} more segments)")

        # Check for voltage saturation
        saturated_levels = [i for i, m in enumerate(mapping_on) if m['saturated']]
        if saturated_levels:
            print(f"\n[DRY RUN] WARNING: {len(saturated_levels)} field levels will be voltage-saturated")
            for i in saturated_levels[:5]:
                m = mapping_on[i]
                print(f"  Level {i}: requested {m['B_req']} mT, achievable {m['B_ach']} mT (scale={m['scale']:.3f})")

        print(f"\n[DRY RUN] Dry run complete - no hardware was used\n")
        print(f"{'='*80}\n")
        return

    print(f"{'='*80}\n")

    # Setup DAQ tasks
    mag_task = nidaqmx.Task("mag_ao_task")
    for ch in config.ao_mag_chans:
        mag_task.ao_channels.add_ao_voltage_chan(f"{config.dev_mag}/{ch}",
                                                  min_val=config.ao_vmin, max_val=config.ao_vmax)
    mag_task.timing.cfg_samp_clk_timing(rate=sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
    AnalogMultiChannelWriter(mag_task.out_stream, auto_start=False).write_many_sample(ao_mag)

    ao_task = nidaqmx.Task("dev2_ao_led_cam")
    ao_task.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_led_chan}",
                                            min_val=config.ao_vmin, max_val=config.ao_vmax)
    ao_task.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_cam_chan}",
                                            min_val=config.ao_vmin, max_val=config.ao_vmax)
    ao_task.timing.cfg_samp_clk_timing(rate=sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
    AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False).write_many_sample(ao2)

    # Configure camera
    try:
        core = Core()
        cam = core.get_camera_device()
        for prop, val in [("TriggerMode", "External"), ("Trigger Source", "External"),
                          ("trigger_mode", "External")]:
            try:
                core.set_property(cam, prop, val)
            except Exception:
                pass
        core.set_exposure(config.mm_set_exposure_ms)
        print("[MM] Camera set to External Trigger; exposure set.")
    except Exception as e:
        print(f"[MM] Could not set camera trigger via Core: {e}")

    # Initialize progress tracking
    if progress:
        progress.start(T, len(snap_ts))

    # DAQ thread with progress updates
    def run_daq():
        try:
            time.sleep(config.daq_start_delay_s)
            ao_task.start()
            mag_task.start()

            # Monitor progress periodically
            start_time = time.time()
            while mag_task.is_task_done() is False or ao_task.is_task_done() is False:
                elapsed = time.time() - start_time
                if progress and elapsed <= T:
                    # Find current segment and field
                    current_field = (0.0, 0.0, 0.0)
                    for (t0, t1, state) in segs:
                        if t0 <= elapsed < t1:
                            if state == "off" or state == "baseline":
                                current_field = (0.0, 0.0, 0.0)
                            else:
                                current_field = levels[state]
                            break

                    # Check if LED should be on
                    led_is_on = False
                    for ts in snap_ts:
                        if elapsed >= max(0, ts - config.led_advance_s) and elapsed < (ts - config.led_advance_s + config.led_pulse_s):
                            led_is_on = True
                            break

                    progress.update(elapsed_time=elapsed, current_field=current_field, led_on=led_is_on)
                time.sleep(0.05)  # Update at 20 Hz

            mag_task.wait_until_done(timeout=T + 5.0)
            ao_task.wait_until_done(timeout=T + 5.0)
        finally:
            for t in (ao_task, mag_task):
                try:
                    t.stop()
                except Exception:
                    pass
                try:
                    t.close()
                except Exception:
                    pass
            zero_daq_outputs(config)
            if progress:
                progress.stop()

    daq_thread = threading.Thread(target=run_daq, daemon=True)

    # MM acquisition thread
    acq_exc = []

    def run_acq():
        try:
            def ipf(image, metadata):
                idx = ipf.counter
                md = frame_meta[idx] if idx < len(frame_meta) else {"frame_index": int(idx)}
                ud = metadata.get("userData", {})
                ud.update(md)
                ud.setdefault("experiment_summary_file", "experiment_summary.json")
                metadata["userData"] = ud
                ipf.counter += 1
                # Update progress with current frame
                if progress:
                    progress.update(current_frame=ipf.counter)
                return image, metadata

            ipf.counter = 0

            events = multi_d_acquisition_events(num_time_points=len(snap_ts), time_interval_s=0)
            with Acquisition(directory=str(save_root), name=run_name,
                             show_display=config.mm_show_display, image_process_fn=ipf) as acq:
                acq.acquire(events)
        except Exception as e:
            acq_exc.append(e)

    acq_thread = threading.Thread(target=run_acq, daemon=True)

    # Start both
    acq_thread.start()
    daq_thread.start()
    deadline = time.time() + T + config.acq_timeout_margin_s
    while time.time() < deadline and (acq_thread.is_alive() or daq_thread.is_alive()):
        time.sleep(0.1)

    # Save summary with comprehensive metadata
    actual_run_dir = resolve_actual_run_dir(save_root, run_name)
    if actual_run_dir:
        # Build comprehensive metadata
        summary = build_comprehensive_metadata(config, "field_ramp", M, Minv)
        summary = add_ramp_specific_metadata(summary, config, levels, mapping_on)

        # Add run-specific info
        summary["run_name"] = run_name
        summary["saved_path"] = str(actual_run_dir)
        summary["timestamp"] = datetime.now().isoformat()

        # Add duration and frame info
        summary["acquisition_info"] = {
            "total_duration_s": T,
            "total_duration_min": T / 60.0,
            "total_frames": len(snap_ts),
            "waveform_samples": N,
        }

        # Add segment sequence for reference
        summary["segment_sequence"] = [
            {
                "start_s": t0,
                "end_s": t1,
                "state": "off" if state == "off" else ("baseline" if state == "baseline" else "on"),
                "level_index": state if isinstance(state, int) else None,
                "field_mT": None if (state == "off" or state == "baseline") else {
                    "Bx": levels[state][0],
                    "By": levels[state][1],
                    "Bz": levels[state][2],
                    "magnitude": float(np.linalg.norm(levels[state]))
                }
            }
            for t0, t1, state in segs
        ]

        # Add per-frame metadata
        summary["frames"] = frame_meta

        with open(actual_run_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[MM] Wrote summary: {actual_run_dir / 'experiment_summary.json'}")

        open_in_fiji(Path(config.fiji_exe), actual_run_dir)

    if acq_exc:
        raise acq_exc[0]


# =========================
# CUSTOM FIELD SEQUENCE EXPERIMENT
# =========================

def build_custom_segments(config, levels):
    """Build segment list for custom field sequence experiment.

    Similar to ramp but uses a custom sequence of fields from CSV.
    """
    segs = []
    t = 0.0
    dur = config.segment_duration_s

    # Add baseline period
    segs.append((t, t + config.baseline_first_s, "baseline"))
    t += config.baseline_first_s

    # Add initial OFF segment if configured
    if config.off_between_fields:
        segs.append((t, t + dur, "off"))
        t += dur

    for i in range(len(levels)):
        # ON segment with this field
        segs.append((t, t + dur, i))
        t += dur

        # OFF segment after each ON (if configured)
        if config.off_between_fields:
            segs.append((t, t + dur, "off"))
            t += dur

    return segs, t


def generate_custom_waveforms(config, M, Minv):
    """Generate waveforms for custom field sequence experiment."""
    levels = config.field_sequence  # List of (Bx, By, Bz) tuples
    segs, T = build_custom_segments(config, levels)

    # Precompute mappings for all field levels
    mapping_on = [map_field_to_voltage_with_limits(B, config.ao_vmin, config.ao_vmax,
                                                    config.ao_headroom_v, M, Minv) for B in levels]
    mapping_off = map_field_to_voltage_with_limits((0.0, 0.0, 0.0), config.ao_vmin,
                                                    config.ao_vmax, config.ao_headroom_v, M, Minv)

    sr = config.sample_rate
    N = int(np.round(T * sr)) + 1

    # Magnet AO
    ao_mag = np.zeros((3, N), dtype=np.float64)
    for (t0, t1, state) in segs:
        i0 = max(0, min(N - 1, int(np.round(t0 * sr))))
        i1 = max(0, min(N, int(np.round(t1 * sr))))
        if i1 <= i0:
            continue
        if state == "off" or state == "baseline":
            V = np.array(mapping_off["V_used"]).reshape(3, 1)
        else:
            V = np.array(mapping_on[state]["V_used"]).reshape(3, 1)
        ao_mag[:, i0:i1] = V

    # Dev2 AO (LED + camera trigger)
    ao2 = np.full((2, N), config.ttl_low_v, dtype=np.float64)

    num_snaps = int(math.floor((T - 1e-12) / config.snap_interval_s)) + 1
    snap_ts = [i * config.snap_interval_s for i in range(num_snaps)]
    snap_ts = [t for t in snap_ts if t < T - 1.0 / sr]

    def mark(arr: np.ndarray, t0: float, pw: float, high: float):
        i0 = int(np.round(t0 * sr))
        i1 = int(np.round((t0 + pw) * sr))
        i0 = max(0, min(N - 1, i0))
        i1 = max(0, min(N, i1))
        if i1 > i0:
            arr[i0:i1] = high

    for ts in snap_ts:
        mark(ao2[1], ts, config.trig_pulse_s, config.ttl_high_v)
        mark(ao2[0], max(0.0, ts - config.led_advance_s), config.led_pulse_s, config.ttl_high_v)

    ao2[:, -1] = config.ttl_low_v

    return ao_mag, ao2, T, snap_ts, segs, levels, mapping_on, mapping_off


def build_custom_frame_metadata(config, snap_ts, segs, levels, mapping_on, mapping_off):
    """Build per-frame metadata for custom field sequence experiment."""
    meta = []
    for i, ts in enumerate(snap_ts):
        state = "off"
        for (t0, t1, st) in segs:
            if t0 <= ts < t1:
                state = st
                break

        if state == "off" or state == "baseline":
            mapd = mapping_off
        else:
            mapd = mapping_on[state]

        B = mapd["B_ach"]
        V = mapd["V_used"]
        meta.append({
            "frame_index": int(i),
            "planned_time_s": float(ts),
            "field_on": (state != "off" and state != "baseline"),
            "magneticField_mT": {"Bx": float(B[0]), "By": float(B[1]), "Bz": float(B[2])},
            "coilVoltages_V": {"Vx": float(V[0]), "Vy": float(V[1]), "Vz": float(V[2])},
            "scale_factor": float(mapd["scale"]),
            "saturated": bool(mapd["saturated"]),
            "led_intensity_mA": float(config.led_intensity_mA),
            "field_index": state if isinstance(state, int) else None,
        })
    return meta


def run_custom_experiment(config, dry_run=False, progress: Optional[ExperimentProgress] = None):
    """Run custom field sequence experiment.

    Args:
        config: CustomConfig object with experiment parameters
        dry_run: If True, generate waveforms and metadata but don't run hardware
        progress: Optional progress tracker for real-time updates
    """
    # Load field sequence from CSV if not already loaded
    if not hasattr(config, 'field_sequence') or not config.field_sequence:
        config.field_sequence = load_field_sequence_from_csv(config.csv_file_path)

    run_name = build_custom_run_name(config)
    save_root = Path(config.save_dir)

    if not dry_run:
        save_root.mkdir(parents=True, exist_ok=True)

    # Load calibration
    M, Minv = load_calibration_matrix(config.calibration_matrix_path)

    # Generate waveforms
    ao_mag, ao2, T, snap_ts, segs, levels, mapping_on, mapping_off = generate_custom_waveforms(config, M, Minv)
    N = ao_mag.shape[1]
    sr = config.sample_rate
    frame_meta = build_custom_frame_metadata(config, snap_ts, segs, levels, mapping_on, mapping_off)

    mode_tag = "[DRY RUN]" if dry_run else "[RUN]"
    print(f"\n{mode_tag} {run_name}")
    print(f"{'='*80}")
    print(f"Experiment Type: Custom Field Sequence")
    print(f"CSV Source: {config.csv_file_path}")
    print(f"Field Points: {len(levels)}")
    print(f"OFF between fields: {'Yes' if config.off_between_fields else 'No'}")
    print(f"Total Duration: {T:.3f}s ({T/60:.1f} min)")
    print(f"Frames: {len(snap_ts)} @ {config.snap_interval_s}s interval")
    print(f"Segment Duration: {config.segment_duration_s}s")
    print(f"Waveform: {N} samples @ {sr} Hz")
    print(f"Save to: {save_root}\\{run_name}")

    if dry_run:
        print(f"\n[DRY RUN] Skipping hardware setup and execution")
        print(f"[DRY RUN] Would configure:")
        print(f"  - DAQ {config.dev_mag}: 3-axis magnet control ({config.ao_mag_chans})")
        print(f"  - DAQ {config.dev_ao}: LED gate ({config.ao_led_chan}) + Camera trigger ({config.ao_cam_chan})")
        print(f"  - Camera: External trigger, {config.mm_set_exposure_ms}ms exposure")

        print(f"\n[DRY RUN] Field sequence (first 15):")
        for i, B in enumerate(levels[:15]):
            B_mag = float(np.linalg.norm(B))
            print(f"  {i:3d}. ({B[0]:7.3f}, {B[1]:7.3f}, {B[2]:7.3f}) mT  |B|={B_mag:.3f} mT")
        if len(levels) > 15:
            print(f"  ... ({len(levels)-15} more field points)")

        # Check for voltage saturation
        saturated_levels = [i for i, m in enumerate(mapping_on) if m['saturated']]
        if saturated_levels:
            print(f"\n[DRY RUN] WARNING: {len(saturated_levels)} field levels will be voltage-saturated")
            for i in saturated_levels[:5]:
                m = mapping_on[i]
                print(f"  Level {i}: requested {m['B_req']} mT, achievable {m['B_ach']} mT (scale={m['scale']:.3f})")

        print(f"\n[DRY RUN] Dry run complete - no hardware was used\n")
        print(f"{'='*80}\n")
        return

    print(f"{'='*80}\n")

    # Setup DAQ tasks
    mag_task = nidaqmx.Task("mag_ao_task")
    for ch in config.ao_mag_chans:
        mag_task.ao_channels.add_ao_voltage_chan(f"{config.dev_mag}/{ch}",
                                                  min_val=config.ao_vmin, max_val=config.ao_vmax)
    mag_task.timing.cfg_samp_clk_timing(rate=sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
    AnalogMultiChannelWriter(mag_task.out_stream, auto_start=False).write_many_sample(ao_mag)

    ao_task = nidaqmx.Task("dev2_ao_led_cam")
    ao_task.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_led_chan}",
                                            min_val=config.ao_vmin, max_val=config.ao_vmax)
    ao_task.ao_channels.add_ao_voltage_chan(f"{config.dev_ao}/{config.ao_cam_chan}",
                                            min_val=config.ao_vmin, max_val=config.ao_vmax)
    ao_task.timing.cfg_samp_clk_timing(rate=sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
    AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False).write_many_sample(ao2)

    # Configure camera
    try:
        core = Core()
        cam = core.get_camera_device()
        for prop, val in [("TriggerMode", "External"), ("Trigger Source", "External"),
                          ("trigger_mode", "External")]:
            try:
                core.set_property(cam, prop, val)
            except Exception:
                pass
        core.set_exposure(config.mm_set_exposure_ms)
        print("[MM] Camera set to External Trigger; exposure set.")
    except Exception as e:
        print(f"[MM] Could not set camera trigger via Core: {e}")

    # Initialize progress tracking
    if progress:
        progress.start(T, len(snap_ts))

    # DAQ thread with progress updates
    def run_daq():
        try:
            time.sleep(config.daq_start_delay_s)
            ao_task.start()
            mag_task.start()

            # Monitor progress periodically
            start_time = time.time()
            while mag_task.is_task_done() is False or ao_task.is_task_done() is False:
                elapsed = time.time() - start_time
                if progress and elapsed <= T:
                    # Find current segment and field
                    current_field = (0.0, 0.0, 0.0)
                    for (t0, t1, state) in segs:
                        if t0 <= elapsed < t1:
                            if state == "off" or state == "baseline":
                                current_field = (0.0, 0.0, 0.0)
                            else:
                                current_field = levels[state]
                            break

                    # Check if LED should be on
                    led_is_on = False
                    for ts in snap_ts:
                        if elapsed >= max(0, ts - config.led_advance_s) and elapsed < (ts - config.led_advance_s + config.led_pulse_s):
                            led_is_on = True
                            break

                    progress.update(elapsed_time=elapsed, current_field=current_field, led_on=led_is_on)
                time.sleep(0.05)  # Update at 20 Hz

            mag_task.wait_until_done(timeout=T + 5.0)
            ao_task.wait_until_done(timeout=T + 5.0)
        finally:
            for t in (ao_task, mag_task):
                try:
                    t.stop()
                except Exception:
                    pass
                try:
                    t.close()
                except Exception:
                    pass
            zero_daq_outputs(config)
            if progress:
                progress.stop()

    daq_thread = threading.Thread(target=run_daq, daemon=True)

    # MM acquisition thread
    acq_exc = []

    def run_acq():
        try:
            def ipf(image, metadata):
                idx = ipf.counter
                md = frame_meta[idx] if idx < len(frame_meta) else {"frame_index": int(idx)}
                ud = metadata.get("userData", {})
                ud.update(md)
                ud.setdefault("experiment_summary_file", "experiment_summary.json")
                metadata["userData"] = ud
                ipf.counter += 1
                # Update progress with current frame
                if progress:
                    progress.update(current_frame=ipf.counter)
                return image, metadata

            ipf.counter = 0

            events = multi_d_acquisition_events(num_time_points=len(snap_ts), time_interval_s=0)
            with Acquisition(directory=str(save_root), name=run_name,
                             show_display=config.mm_show_display, image_process_fn=ipf) as acq:
                acq.acquire(events)
        except Exception as e:
            acq_exc.append(e)

    acq_thread = threading.Thread(target=run_acq, daemon=True)

    # Start both
    acq_thread.start()
    daq_thread.start()
    deadline = time.time() + T + config.acq_timeout_margin_s
    while time.time() < deadline and (acq_thread.is_alive() or daq_thread.is_alive()):
        time.sleep(0.1)

    # Save summary with comprehensive metadata
    actual_run_dir = resolve_actual_run_dir(save_root, run_name)
    if actual_run_dir:
        # Build comprehensive metadata
        summary = build_comprehensive_metadata(config, "custom_field_sequence", M, Minv)
        summary = add_custom_specific_metadata(summary, config, levels, mapping_on)

        # Add run-specific info
        summary["run_name"] = run_name
        summary["saved_path"] = str(actual_run_dir)
        summary["timestamp"] = datetime.now().isoformat()

        # Add duration and frame info
        summary["acquisition_info"] = {
            "total_duration_s": T,
            "total_duration_min": T / 60.0,
            "total_frames": len(snap_ts),
            "waveform_samples": N,
        }

        # Add segment sequence for reference
        summary["segment_sequence"] = [
            {
                "start_s": t0,
                "end_s": t1,
                "state": "off" if state == "off" else ("baseline" if state == "baseline" else "on"),
                "field_index": state if isinstance(state, int) else None,
                "field_mT": None if (state == "off" or state == "baseline") else {
                    "Bx": levels[state][0],
                    "By": levels[state][1],
                    "Bz": levels[state][2],
                    "magnitude": float(np.linalg.norm(levels[state]))
                }
            }
            for t0, t1, state in segs
        ]

        # Add per-frame metadata
        summary["frames"] = frame_meta

        with open(actual_run_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[MM] Wrote summary: {actual_run_dir / 'experiment_summary.json'}")

        open_in_fiji(Path(config.fiji_exe), actual_run_dir)

    if acq_exc:
        raise acq_exc[0]
