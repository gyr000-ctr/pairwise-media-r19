"""Statistical voice activity detection.

The public :class:`Vad` combines:

* normalised spectral entropy (speech is spectrally more structured than
  broadband noise),
* an adaptive per-band noise power estimate (robust quantile plus a causal
  fast-jump tracker), and
* a statistical energy/SNR score whose reference distribution is estimated
  from the signal being analysed.

The constructor ``threshold`` remains as a caller-supplied normalised RMS
floor for digital silence / API compatibility.  It is not the noisy-speech
decision boundary: the entropy and SNR boundaries are recomputed from each
input.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np

__all__ = [
    "Vad",
    "VAD",
    "SpectralEntropyVad",
    "SpectralEntropyVAD",
    "StatisticalVad",
    "StatisticalVAD",
    "EnergyVad",
    "EnergyVAD",
    "VadStream",
    "AdaptiveNoiseModel",
    "vad_segments",
    "detect_vad",
    "energy_segments",
    "VadError",
    "InvalidParameterError",
    "InvalidParameterTypeError",
    "InvalidSampleRateError",
    "InvalidSampleRateTypeError",
    "InvalidThresholdError",
    "InvalidThresholdTypeError",
    "InvalidFrameParameterError",
    "InvalidFrameParameterTypeError",
    "InvalidInputError",
    "InvalidInputTypeError",
    "InvalidInputShapeError",
    "NonFiniteInputError",
    "NaNInputError",
    "InfiniteInputError",
]

_FRAME_MS = 20.0
_HOP_MS = 5.0
_N_BANDS = 32
_EPS = 1e-300


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #
class VadError(ValueError):
    """Base class for all VAD configuration and input errors."""

    code = "vad_error"


class InvalidParameterError(VadError):
    """A configuration parameter is invalid."""

    code = "invalid_parameter"

    def __init__(self, message: str, *, parameter: str | None = None,
                 value: Any = None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class InvalidParameterTypeError(TypeError, InvalidParameterError):
    """A configuration parameter has the wrong type."""

    code = "invalid_parameter_type"


class InvalidSampleRateError(InvalidParameterError):
    """The sample rate is not a positive finite number."""

    code = "invalid_sample_rate"


class InvalidSampleRateTypeError(InvalidParameterTypeError,
                                 InvalidSampleRateError):
    """The sample rate has an unsupported type."""

    code = "invalid_sample_rate_type"


class InvalidThresholdError(InvalidParameterError):
    """The public RMS threshold is invalid."""

    code = "invalid_threshold"


class InvalidThresholdTypeError(InvalidParameterTypeError,
                                InvalidThresholdError):
    """The threshold has an unsupported type."""

    code = "invalid_threshold_type"


class InvalidFrameParameterError(InvalidParameterError):
    """Frame/hop/minimum-duration configuration is invalid."""

    code = "invalid_frame_parameter"


class InvalidFrameParameterTypeError(InvalidParameterTypeError,
                                     InvalidFrameParameterError):
    """A frame configuration parameter has an unsupported type."""

    code = "invalid_frame_parameter_type"


class InvalidInputError(VadError):
    """The audio input cannot be analysed."""

    code = "invalid_input"


class InvalidInputTypeError(TypeError, InvalidInputError):
    """The input is not a one-dimensional sequence of real samples."""

    code = "invalid_input_type"


class InvalidInputShapeError(InvalidInputError):
    """The input has an unsupported shape."""

    code = "invalid_input_shape"


class NonFiniteInputError(InvalidInputError):
    """The input contains NaN or infinite samples."""

    code = "non_finite_input"


class NaNInputError(NonFiniteInputError):
    """The input contains at least one NaN sample."""

    code = "nan_input"


class InfiniteInputError(NonFiniteInputError):
    """The input contains at least one positive or negative infinity."""

    code = "infinite_input"


# --------------------------------------------------------------------------- #
# Validation and small helpers
# --------------------------------------------------------------------------- #
def _real_scalar(value: Any, name: str, error_type: type[InvalidParameterError],
                 *, positive: bool = False, strict_unit: bool = False,
                 type_error_type: type[InvalidParameterTypeError] | None = None
                 ) -> float:
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        cls = type_error_type or error_type
        raise cls(
            f"{name} 参数非法: must be a real number, got {type(value).__name__}",
            parameter=name, value=value)
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_type(
            f"{name} must be a finite real number",
            parameter=name, value=value) from exc
    if not math.isfinite(out):
        raise error_type(
            f"{name} 参数非法: must be finite, got {value!r}",
            parameter=name, value=value)
    if positive and out <= 0.0:
        raise error_type(
            f"{name} 参数非法: must be positive, got {value!r}",
            parameter=name, value=value)
    if strict_unit and not (0.0 < out < 1.0):
        raise error_type(
            f"{name} 参数非法: must be strictly between 0 and 1, got {value!r}",
            parameter=name, value=value)
    return out


def _validate_samples(samples: Any) -> np.ndarray:
    """Return a finite, one-dimensional float64 array without mutating input."""
    if isinstance(samples, (str, bytes, bytearray, memoryview)):
        raise InvalidInputTypeError(
            "samples must be a sequence of real numbers, not text/bytes")

    if isinstance(samples, Iterable) and not isinstance(
            samples, (list, tuple, np.ndarray, range)) and not hasattr(
            samples, "__len__"):
        try:
            samples = list(samples)
        except Exception as exc:  # pragma: no cover - exotic iterable
            raise InvalidInputTypeError(
                "samples must be a one-dimensional sequence of real numbers"
            ) from exc

    try:
        arr = np.asarray(samples)
    except Exception as exc:
        raise InvalidInputTypeError(
            "samples must be a one-dimensional sequence of real numbers"
        ) from exc

    if arr.dtype == object:
        # np.asarray on a generator/scalar can produce a 0-D object array.
        if arr.ndim == 0:
            raise InvalidInputTypeError(
                "samples must be a one-dimensional sequence, not a scalar")
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidInputTypeError(
                "samples must contain real numbers only") from exc
    else:
        if arr.dtype.kind not in "biuf":
            raise InvalidInputTypeError(
                "samples must contain real numbers only "
                f"(got dtype {arr.dtype})")
        try:
            arr = arr.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise InvalidInputTypeError(
                "samples must contain real numbers only") from exc

    if arr.ndim == 0:
        raise InvalidInputShapeError(
            "samples must be one-dimensional, not a scalar")
    if arr.ndim != 1:
        raise InvalidInputShapeError(
            f"samples must be one-dimensional, got {arr.ndim} dimensions")
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)

    finite = np.isfinite(arr)
    if not np.all(finite):
        bad = arr[~finite]
        if np.any(np.isnan(bad)):
            raise NaNInputError("samples contain NaN (输入包含 NaN/非有限样本)")
        raise InfiniteInputError("samples contain positive or negative infinity (输入包含 inf/非有限样本)")
    return arr


def _spectral_entropy(band_power: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy for rows of non-negative band powers."""
    total = band_power.sum(axis=1, keepdims=True)
    prob = np.divide(
        band_power, total, out=np.zeros_like(band_power), where=total > 0.0)
    entropy = -np.sum(prob * np.log(prob + _EPS), axis=1)
    n_bands = max(2, int(band_power.shape[1]))
    return np.clip(entropy / math.log(n_bands), 0.0, 1.0)


def _high_noise_stats(values: np.ndarray, quantile: float = 68.0
                      ) -> tuple[float, float]:
    """Median/MAD of the high (noise-like) component of a feature."""
    if values.size == 0:
        return 0.95, 0.02
    cut = float(np.percentile(values, quantile))
    candidate = values[values >= cut]
    if candidate.size < 3:
        candidate = values
    center = float(np.median(candidate))
    mad = float(np.median(np.abs(candidate - center))) * 1.4826
    # The floor prevents a handful of identical digital-silence frames from
    # producing a zero variance and an over-tight statistical boundary.
    scale = max(mad, 0.010)
    return center, scale


# --------------------------------------------------------------------------- #
# Causal adaptive noise PSD tracker
# --------------------------------------------------------------------------- #
class AdaptiveNoiseModel:
    """Fast noise-PSD/energy tracker with a broadband-step override.

    A uniform broadband level jump is adopted within one or two 5 ms frames,
    i.e. well within the required 50 ms adaptation budget.  Speech-like
    spectrally structured frames update the estimate only very slowly.
    """

    def __init__(self, n_bands: int, *, hop_ms: float = _HOP_MS):
        self.n_bands = int(n_bands)
        self.hop_ms = float(hop_ms)
        self.psd: np.ndarray | None = None
        self.energy_mean: float | None = None
        self.energy_scale = 0.08
        self.entropy_mean = 0.95
        self.entropy_scale = 0.02
        self.fast_left = 0
        self.last_jump_frame = -10**9
        self.frame_index = 0
        self._pending: list[np.ndarray] = []

    def update(self, band_power: np.ndarray, entropy: float,
               log_energy: float) -> tuple[np.ndarray, float, bool]:
        i = self.frame_index
        self.frame_index += 1

        if self.psd is None:
            self._pending.append(np.asarray(band_power, dtype=np.float64))
            if len(self._pending) >= 4:
                self.psd = np.median(np.stack(self._pending), axis=0)
                self.energy_mean = float(log_energy)
                self.entropy_mean = float(entropy)
            else:
                # Not enough context yet; expose the observation itself.
                self.psd = np.asarray(band_power, dtype=np.float64).copy()
                self.energy_mean = float(log_energy)
                self.entropy_mean = float(entropy)
            return self.psd.copy(), 0.0, False

        ratio = band_power / np.maximum(self.psd, 1e-20)
        log_ratio = np.log(np.maximum(ratio, 1e-20))
        # Mean statistics are ample for a uniform broadband step and avoid two
        # relatively expensive per-frame NumPy medians on tiny arrays.
        level = float(np.mean(log_ratio))
        dispersion = float(np.mean(np.abs(log_ratio - level))) + 1e-6
        entropy_noise_like = entropy >= self.entropy_mean - 3.0 * self.entropy_scale

        jump = (
            (level > math.log(2.6) or level < -math.log(2.6))
            and dispersion < 0.55
            and entropy_noise_like
        )
        if jump:
            # Snap, then average quickly for a few frames to reduce periodogram
            # variance.  This is the sub-50 ms adaptation path.
            self.psd = 0.45 * self.psd + 0.55 * band_power
            self.energy_mean = 0.4 * self.energy_mean + 0.6 * log_energy
            self.energy_scale = 0.08
            self.fast_left = max(self.fast_left, 8)
            self.last_jump_frame = i
            alpha_e = 0.5
            self.entropy_mean = 0.5 * self.entropy_mean + 0.5 * entropy
        else:
            energy_z = abs(log_energy - self.energy_mean) / max(
                self.energy_scale, 0.05)
            noise_like = (
                entropy >= self.entropy_mean - 2.5 * self.entropy_scale
                and energy_z < 4.0
            )
            if self.fast_left > 0:
                alpha = 0.45
                self.fast_left -= 1
            elif noise_like:
                alpha = 0.18
            else:
                alpha = 0.015
            self.psd = (1.0 - alpha) * self.psd + alpha * band_power
            if noise_like or self.fast_left > 0:
                a = 0.12 if noise_like else 0.4
                new_scale = max(
                    0.05,
                    1.4826 * abs(log_energy - self.energy_mean))
                self.energy_mean = (1.0 - a) * self.energy_mean + a * log_energy
                self.energy_scale = (
                    0.9 * self.energy_scale + 0.1 * new_scale)
                self.entropy_mean = (
                    0.9 * self.entropy_mean + 0.1 * entropy)

        z_energy = (log_energy - self.energy_mean) / max(
            self.energy_scale, 0.05)
        return self.psd.copy(), float(z_energy), jump


# --------------------------------------------------------------------------- #
# Main detector
# --------------------------------------------------------------------------- #
class Vad:
    """Spectral-entropy statistical VAD.

    Parameters
    ----------
    rate:
        Sample rate in Hz.
    threshold:
        Normalised RMS activity floor in ``(0, 1)``.  This preserves the
        public constructor contract and removes digital low-level noise; it is
        not used as the adaptive speech/noise boundary.
    """

    def __init__(self, rate: Any = 16000, threshold: Any = 0.02, *,
                 frame_ms: float = _FRAME_MS, hop_ms: float = _HOP_MS,
                 min_segment_ms: float = 18.0):
        # Accept integer-valued floating sample rates as a convenience.
        rate_value = _real_scalar(
            rate, "sample_rate", InvalidSampleRateError, positive=True,
            type_error_type=InvalidSampleRateTypeError)
        if isinstance(rate, float) and not math.isclose(
                rate_value, round(rate_value), rel_tol=0.0, abs_tol=1e-9):
            # Non-integer rates are mathematically usable; retain the exact
            # value rather than rejecting an unusual but legal configuration.
            self.rate: int | float = rate_value
        else:
            self.rate = int(round(rate_value))
        self.sample_rate = self.rate

        self.threshold = _real_scalar(
            threshold, "threshold", InvalidThresholdError, strict_unit=True,
            type_error_type=InvalidThresholdTypeError)

        self.frame_ms = _real_scalar(
            frame_ms, "frame_ms", InvalidFrameParameterError, positive=True,
            type_error_type=InvalidFrameParameterTypeError)
        self.hop_ms = _real_scalar(
            hop_ms, "hop_ms", InvalidFrameParameterError, positive=True,
            type_error_type=InvalidFrameParameterTypeError)
        self.min_segment_ms = _real_scalar(
            min_segment_ms, "min_segment_ms", InvalidFrameParameterError,
            positive=True,
            type_error_type=InvalidFrameParameterTypeError)
        if self.hop_ms > self.frame_ms:
            raise InvalidFrameParameterError(
                "hop_ms must not be greater than frame_ms",
                parameter="hop_ms", value=hop_ms)

        rate_f = float(self.rate)
        self.win = max(16, int(round(rate_f * self.frame_ms / 1000.0)))
        self.hop = max(1, int(round(rate_f * self.hop_ms / 1000.0)))
        self.min_run = max(1, int(round(
            self.min_segment_ms / max(self.hop_ms, 1e-9))))

        # Diagnostics from the most recent complete-buffer analysis.  They are
        # advisory only; all classification state is local to segments().
        self.entropy_threshold_: float | None = None
        self.whitened_entropy_threshold_: float | None = None
        self.energy_threshold_: float | None = None
        self.noise_psd_: np.ndarray | None = None
        self.adaptation_ms_: float = self.hop_ms

    # ------------------------------------------------------------------ #
    def segments(self, samples: Any) -> list[tuple[int, int]]:
        """Return ordered, non-overlapping half-open ``(start, end)`` pairs."""
        x = _validate_samples(samples)
        n = int(x.size)
        if n == 0:
            return []

        rms, log_energy, band_power, n_frames = self._features(x)
        entropy = _spectral_entropy(band_power)

        # Robust clip-wide noise PSD.  The 22nd percentile is below the mean
        # periodogram but above exact zeros; entropy is scale-normalised, so
        # the de-biasing constant is unnecessary for discrimination.
        noise_psd = np.percentile(band_power, 22.0, axis=0)
        kernel = np.array([0.15, 0.20, 0.30, 0.20, 0.15])
        noise_psd = np.convolve(noise_psd, kernel, mode="same")
        noise_psd = np.maximum(noise_psd, 1e-12)
        whitened = _spectral_entropy(band_power / noise_psd[None, :])

        mu_h, sd_h = _high_noise_stats(entropy)
        mu_w, sd_w = _high_noise_stats(whitened)
        z_h = (mu_h - entropy) / sd_h
        z_w = (mu_w - whitened) / sd_w

        energy_base = float(np.percentile(log_energy, 25.0))
        energy_noise = log_energy <= float(np.percentile(log_energy, 60.0))
        if int(energy_noise.sum()) >= 3:
            energy_mad = float(np.median(np.abs(
                log_energy[energy_noise]
                - float(np.median(log_energy[energy_noise]))))) * 1.4826
        else:
            energy_mad = 0.08
        energy_scale = max(energy_mad, 0.06)
        z_e = (log_energy - energy_base) / energy_scale

        tracker = AdaptiveNoiseModel(band_power.shape[1], hop_ms=self.hop_ms)
        jump_guard = np.zeros(n_frames, dtype=bool)
        tracked_psd = np.empty_like(band_power)
        z_e_adaptive = np.zeros(n_frames, dtype=np.float64)
        guard_left = 0
        for i in range(n_frames):
            psd, z_e_fast, jump = tracker.update(
                band_power[i], float(entropy[i]), float(log_energy[i]))
            tracked_psd[i] = psd
            z_e_adaptive[i] = z_e_fast
            if jump and z_h[i] < 5.5:
                guard_left = max(guard_left, 2)
            if guard_left > 0 and z_h[i] < 5.5:
                jump_guard[i] = True
                guard_left -= 1

        gate = rms >= self.threshold
        peak_share = np.max(band_power, axis=1) / np.maximum(
            band_power.sum(axis=1), _EPS)

        # The local adaptive energy score is the primary SNR term.  Keep the
        # clip-global robust score only as a secondary onset cue.
        voice = (
            (z_h > 4.5)
            | ((z_h > 2.2) & (z_e_adaptive > 4.0))
            | ((z_h > 2.8) & (z_e > 5.0))
            | (z_w > 5.5)
            | ((z_w > 2.5) & (z_e_adaptive > 5.0))
            # Strong narrow-band structure remains detectable even in a clip
            # that contains no separate noise-only reference frames.
            | ((entropy < 0.18) & (peak_share > 0.65))
        )
        # A pure DC/full-scale line has essentially zero band entropy.  This
        # very tight degeneracy check is separate from the adaptive boundary
        # and is paired with the caller RMS gate.
        voice |= entropy < 1e-5
        voice &= gate
        voice &= ~jump_guard

        decoded = self._decode(voice)
        result = self._states_to_segments(decoded, n)

        # Advisory diagnostics; reference assignment is atomic under the
        # CPython GIL and cannot corrupt another call's local arrays.
        self.entropy_threshold_ = float(mu_h - 4.5 * sd_h)
        self.whitened_entropy_threshold_ = float(mu_w - 5.5 * sd_w)
        self.energy_threshold_ = float(energy_base + 4.0 * energy_scale)
        self.noise_psd_ = tracked_psd
        return result

    # Convenience aliases for callers expecting a function-style API.
    detect = segments

    def __call__(self, samples: Any) -> list[tuple[int, int]]:
        return self.segments(samples)

    # ------------------------------------------------------------------ #
    def _features(self, x: np.ndarray):
        n = int(x.size)
        win, hop = self.win, self.hop
        n_frames = max(1, int(math.ceil(n / hop)))
        left = win // 2
        padded = np.pad(x, (left, win - left))
        frames = np.lib.stride_tricks.sliding_window_view(padded, win)
        frames = frames[::hop][:n_frames]

        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-300)
        log_energy = np.log(rms + 1e-300)

        window = np.hanning(win)
        spectrum = np.fft.rfft(frames * window, axis=1)
        power = np.maximum(
            spectrum.real * spectrum.real + spectrum.imag * spectrum.imag,
            1e-20)

        bins = int(power.shape[1])
        n_bands = min(_N_BANDS, max(2, bins - 1))
        edges = np.linspace(1, bins, n_bands + 1).astype(np.intp)
        # Defensive de-duplication for unusual tiny sample rates.
        edges = np.unique(edges)
        if edges.size < 2:
            band_power = power
        else:
            band_power = np.maximum(
                np.add.reduceat(power, edges[:-1], axis=1), 1e-20)
        return rms, log_energy, band_power, n_frames

    def _decode(self, labels: np.ndarray) -> np.ndarray:
        out = np.asarray(labels, dtype=bool).copy()
        nf = out.size
        if nf == 0:
            return out

        # Bridge single missing frames.
        if nf >= 3:
            middle = ~out[1:-1] & out[:-2] & out[2:]
            out[1:-1] |= middle

        # Drop isolated sub-minimum runs.
        i = 0
        while i < nf:
            if not out[i]:
                i += 1
                continue
            j = i + 1
            while j < nf and out[j]:
                j += 1
            if j - i < max(2, self.min_run):
                out[i:j] = False
            i = j

        # One-frame hangover smooths endpoint jitter while staying well inside
        # the 20 ms boundary budget at the default 5 ms hop.
        add = np.zeros_like(out)
        i = 0
        while i < nf:
            if not out[i]:
                i += 1
                continue
            j = i + 1
            while j < nf and out[j]:
                j += 1
            if j < nf and j - i >= 2:
                add[j] = True
            i = j
        return out | add

    def _states_to_segments(self, state: np.ndarray, n: int
                            ) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        nf = int(state.size)
        i = 0
        hop = self.hop
        while i < nf:
            if not state[i]:
                i += 1
                continue
            j = i + 1
            while j < nf and state[j]:
                j += 1
            # Frame centres are on the hop grid.  Mapping an edge to the
            # first/last active centre gives <= one hop of quantisation and
            # avoids expanding systematic boundary error with the window size.
            start = max(0, int(i * hop))
            end = min(n, int((j - 1) * hop))
            if end <= start:
                end = min(n, start + max(1, self.hop // 2))
            if end > start:
                segments.append((int(start), int(end)))
            i = j
        return segments


# --------------------------------------------------------------------------- #
# Simple energy baseline and streaming wrapper
# --------------------------------------------------------------------------- #
class EnergyVad:
    """Small RMS/hysteresis energy detector used for comparison/testing."""

    def __init__(self, rate: Any = 16000, threshold: Any = 0.02, *,
                 hop_ms: float = 10.0, factor: float = 4.0):
        self.rate = int(_real_scalar(
            rate, "sample_rate", InvalidSampleRateError, positive=True,
            type_error_type=InvalidSampleRateTypeError))
        self.threshold = _real_scalar(
            threshold, "threshold", InvalidThresholdError, strict_unit=True,
            type_error_type=InvalidThresholdTypeError)
        self.hop = max(1, int(round(self.rate * hop_ms / 1000.0)))
        self.factor = float(factor)

    def segments(self, samples: Any) -> list[tuple[int, int]]:
        x = _validate_samples(samples)
        n = x.size
        if n == 0:
            return []
        hop = self.hop
        nf = max(1, math.ceil(n / hop))
        padded = np.zeros(nf * hop, dtype=np.float64)
        padded[:n] = x
        rms = np.sqrt(np.mean(padded.reshape(nf, hop) ** 2, axis=1))
        floor = max(float(np.percentile(rms[rms > 0], 15))
                    if np.any(rms > 0) else 1e-12, 1e-12)
        on = rms > max(self.threshold, floor * self.factor)
        state = np.zeros(nf, dtype=bool)
        voice = False
        on_run = off_run = 0
        for i, hit in enumerate(on):
            if not voice:
                on_run = on_run + 1 if hit else 0
                if on_run >= 2:
                    voice = True
                    state[max(0, i - 1):i + 1] = True
                    off_run = 0
            else:
                state[i] = True
                off_run = off_run + 1 if not hit else 0
                if off_run >= 3:
                    voice = False
                    state[max(0, i - 2):i + 1] = False
                    on_run = 0
        out = []
        i = 0
        while i < nf:
            if not state[i]:
                i += 1
                continue
            j = i + 1
            while j < nf and state[j]:
                j += 1
            out.append((int(i * hop), int(min(n, j * hop))))
            i = j
        return out


class VadStream:
    """Thread-safe incremental wrapper around :class:`Vad`.

    The wrapper carries a small sample overlap so calls may be made on
    arbitrary buffer boundaries.  Returned endpoints are local to each chunk;
    ``global_offset`` gives the sample position of that chunk in the stream.
    """

    def __init__(self, vad: Vad | None = None, **kwargs: Any):
        self.vad = vad if vad is not None else Vad(**kwargs)
        self._lock = threading.RLock()
        self._carry = np.zeros(0, dtype=np.float64)
        self._global_offset = 0

    def process(self, chunk: Any) -> list[tuple[int, int]]:
        with self._lock:
            x = _validate_samples(chunk)
            # The batch detector already uses centred, overlapping frames; a
            # small carry prevents edge truncation at buffer boundaries.
            data = np.concatenate([self._carry, x]) if self._carry.size else x
            raw = self.vad.segments(data)
            carry_n = min(data.size, self.vad.win)
            self._carry = data[-carry_n:].copy() if carry_n else data[:0].copy()
            base = self._global_offset - (data.size - x.size)
            result = []
            for a, b in raw:
                ga = max(self._global_offset, base + a)
                gb = min(self._global_offset + x.size, base + b)
                if gb > ga:
                    result.append((int(ga - self._global_offset),
                                   int(gb - self._global_offset)))
            self._global_offset += x.size
            return result

    def reset(self) -> None:
        with self._lock:
            self._carry = np.zeros(0, dtype=np.float64)
            self._global_offset = 0


def vad_segments(samples: Any, rate: int = 16000,
                 threshold: float = 0.02) -> list[tuple[int, int]]:
    """Functional shorthand for ``Vad(rate, threshold).segments(samples)``."""
    return Vad(rate, threshold).segments(samples)


def energy_segments(samples: Any, rate: int = 16000,
                    threshold: float = 0.02) -> list[tuple[int, int]]:
    """Convenience energy-baseline segmentation."""
    return EnergyVad(rate, threshold).segments(samples)


# Common descriptive aliases.
VAD = Vad
SpectralEntropyVad = Vad
SpectralEntropyVAD = Vad
StatisticalVad = Vad
StatisticalVAD = Vad
EnergyVAD = EnergyVad
detect_vad = vad_segments
