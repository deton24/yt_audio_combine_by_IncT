#!/usr/bin/env python3
"""Download and combine YouTube audio formats via chunked streaming DSP.

Downloads m4a (format 140) and webm/opus (format 251) from YouTube, decodes
both to PCM via ffmpeg, and combines them through a multi-step signal processing
pipeline operating in fixed-size chunks to bound memory usage.

Usage:
    python yt_audio_combine.py [URL] [--stdout] [--keep] [--dont-keep]
                               [-p PATH] [--chunk-size N] [--format FMT]
                               [--preset NAME] [yt-dlp opts...]

Examples:
    python yt_audio_combine.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python yt_audio_combine.py URL --format mp3
    python yt_audio_combine.py URL --format flac --bit-depth 16
    python yt_audio_combine.py URL --format flac --embed-metadata --output-template "%(artist)s - %(title)s"
    python yt_audio_combine.py URL --format m4a --encoder "qaac --cvbr 256 - -o {output}"
    python yt_audio_combine.py --save-preset hq --format flac --bit-depth 24 --embed-metadata
    python yt_audio_combine.py URL --preset hq
    python yt_audio_combine.py -a urls.txt --cookies-from-browser chrome
    python yt_audio_combine.py -p F:\\music                # process pairs in custom folder
    python yt_audio_combine.py -p F:\\music\\song.m4a      # process single pair by file
    python yt_audio_combine.py                             # process existing files in streams/yt/
    python yt_audio_combine.py URL --stdout | ffplay -
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import firwin, stft as scipy_stft, istft as scipy_istft, fftconvolve
from scipy.fft import next_fast_len

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS = 2
BYTES_PER_SAMPLE = 4  # float32
FRAME_BYTES = CHANNELS * BYTES_PER_SAMPLE  # 8 bytes per stereo sample frame
DEFAULT_CHUNK = 262144  # samples per chunk (~5.9 s at 44100)

STREAMS_DIR = Path("streams/yt")
COMBINED_DIR = Path("combined/yt")
ARCHIVE_DIR = Path("archive/yt")

# Table bias mode: frequency-band-optimized bias values
TABLE_BIASES = (45, 55, 60)

# Built-in format defaults for ffmpeg-based encoding
FORMAT_DEFAULTS: dict[str, dict] = {
    "wav":  {"ext": "wav",  "codec": None},
    "mp3":  {"ext": "mp3",  "codec": "libmp3lame", "default_bitrate": "320k"},
    "flac": {"ext": "flac", "codec": "flac",       "default_bit_depth": 24},
    "opus": {"ext": "opus", "codec": "libopus",    "default_bitrate": "128k"},
    "aac":  {"ext": "m4a",  "codec": "aac",        "default_bitrate": "256k"},
    "ogg":  {"ext": "ogg",  "codec": "libvorbis",  "default_bitrate": "320k"},
}

# Metadata field mapping: info.json key -> standard tag name
METADATA_TAG_MAP = {
    "title": "title",
    "artist": "artist",
    "uploader": "artist",
    "album": "album",
    "track": "track",
    "upload_date": "date",
}


# ---------------------------------------------------------------------------
# Filter design (from linear_phase_filter.py)
# ---------------------------------------------------------------------------
def design_brickwall_fir(pass_type: str, freq: float, fs: int,
                         taps_count: int = 513) -> np.ndarray:
    """Design a brickwall linear-phase FIR (highpass or lowpass).

    Returns 1-D float32 tap array of odd length.
    """
    nyq = fs / 2.0
    norm_cutoff = freq / nyq
    if not 0.0 < norm_cutoff < 1.0:
        raise ValueError(f"cutoff {freq} Hz out of range for fs={fs}")

    numtaps = max(taps_count * 4, 2049)
    if numtaps % 2 == 0:
        numtaps += 1

    if pass_type in ("lp", "blp"):
        taps = firwin(numtaps, norm_cutoff, window="hamming", pass_zero=True)
    elif pass_type in ("hp", "bhp"):
        taps = firwin(numtaps, norm_cutoff, window="hamming", pass_zero=False)
    else:
        raise ValueError(f"Unknown pass_type: {pass_type}")
    return taps.astype(np.float32)


def design_multiband_fir(cutoffs: list[float], fs: int, pass_zero: bool,
                         taps_count: int = 513) -> np.ndarray:
    """Design a multiband linear-phase FIR filter.

    cutoffs: list of transition frequencies in Hz (must be in ascending order).
    pass_zero: if True the first band (DC) is a passband; if False it is a
               stopband.  Bands alternate pass/stop between adjacent cutoffs.

    Returns 1-D float32 tap array of odd length.
    """
    nyq = fs / 2.0
    norm_cutoffs = [f / nyq for f in cutoffs]
    for nc in norm_cutoffs:
        if not 0.0 < nc < 1.0:
            raise ValueError(f"cutoff {nc * nyq} Hz out of range for fs={fs}")

    numtaps = max(taps_count * 4, 2049)
    if numtaps % 2 == 0:
        numtaps += 1

    taps = firwin(numtaps, norm_cutoffs, window="hamming", pass_zero=pass_zero)
    return taps.astype(np.float32)


# ---------------------------------------------------------------------------
# Streaming processors
# ---------------------------------------------------------------------------
class StreamingFIR:
    """Causal FIR via overlap-save.  Output is delayed by (L-1)//2 samples
    relative to scipy fftconvolve(mode='same').  Caller compensates with
    StreamingDelay on unfiltered paths."""

    def __init__(self, taps: np.ndarray):
        self.h = taps.astype(np.float32)
        self.L = len(taps)
        self._H: np.ndarray | None = None
        self._fft_n: int = 0
        self._tail: np.ndarray | None = None  # (L-1, C)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (M, C) float32 -> (M, C) float32."""
        M, C = chunk.shape
        if self._tail is None:
            self._tail = np.zeros((self.L - 1, C), dtype=np.float32)
            N = M + self.L - 1
            self._fft_n = next_fast_len(N)
            self._H = np.fft.rfft(self.h, self._fft_n)

        out = np.empty((M, C), dtype=np.float32)
        for ch in range(C):
            block = np.concatenate([self._tail[:, ch], chunk[:, ch]])
            Y = np.fft.rfft(block, self._fft_n)
            Y *= self._H
            y = np.fft.irfft(Y, self._fft_n)
            out[:, ch] = y[self.L - 1: self.L - 1 + M]

        if M >= self.L - 1:
            self._tail = chunk[-(self.L - 1):].copy()
        else:
            self._tail = np.concatenate(
                [self._tail[M:], chunk], axis=0
            ).copy()
        return out


class StreamingDelay:
    """Delays a signal by exactly D samples (zero-filled at start)."""

    def __init__(self, D: int):
        self.D = D
        self._buf: np.ndarray | None = None  # (D, C)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        M, C = chunk.shape
        if self._buf is None:
            self._buf = np.zeros((self.D, C), dtype=np.float32)
        combined = np.concatenate([self._buf, chunk], axis=0)
        out = combined[:M].copy()
        self._buf = combined[M: M + self.D].copy()
        return out


class StreamingSTFTMasker:
    """Streaming STFT frequency masker (replaces mask_extraction.py dual-file mode).

    Keeps source bins where mask magnitude <= threshold_db; zeroes the rest.
    Uses scipy stft/istft per chunk for correct WOLA synthesis (zero intrinsic
    delay -- boundary handling via prepended tail from previous chunk).

    NOTE: uses scaling='psd' so that STFT magnitudes stay close to raw FFT
    levels.  The default 'spectrum' scaling divides by sum(window) ≈ 512,
    pushing magnitudes ~54 dB lower and causing threshold-edge bins to
    flicker above/below the cutoff, producing audible periodic modulation.
    """

    def __init__(self, nperseg: int = 1024, overlap_factor: int = 8,
                 threshold_db: float = -118.0):
        self.nperseg = nperseg
        self.noverlap = int(nperseg * (1.0 - 1.0 / overlap_factor))
        self.hop = nperseg - self.noverlap
        self.threshold_db = threshold_db
        # Tail length: noverlap samples from previous chunk for continuity
        self._tail_len = self.noverlap
        self._src_tail: np.ndarray | None = None
        self._msk_tail: np.ndarray | None = None

    def process(self, src: np.ndarray, msk: np.ndarray) -> np.ndarray:
        """src, msk: (M, C). Returns (M, C)."""
        M, C = src.shape
        if self._src_tail is None:
            self._src_tail = np.zeros((self._tail_len, C), dtype=np.float32)
            self._msk_tail = np.zeros((self._tail_len, C), dtype=np.float32)

        # Prepend tail for boundary continuity, append nperseg zeros for
        # end-of-chunk protection
        pad_end = self.nperseg
        src_ext = np.concatenate([self._src_tail, src,
                                  np.zeros((pad_end, C), dtype=np.float32)])
        msk_ext = np.concatenate([self._msk_tail, msk,
                                  np.zeros((pad_end, C), dtype=np.float32)])

        out = np.empty((M, C), dtype=np.float32)
        for ch in range(C):
            _, _, Zs = scipy_stft(src_ext[:, ch], fs=SAMPLE_RATE,
                                  nperseg=self.nperseg, noverlap=self.noverlap,
                                  scaling='psd')
            _, _, Zm = scipy_stft(msk_ext[:, ch], fs=SAMPLE_RATE,
                                  nperseg=self.nperseg, noverlap=self.noverlap,
                                  scaling='psd')
            mask = (20.0 * np.log10(
                np.maximum(np.abs(Zm), 1e-20))) <= self.threshold_db
            Zo = np.where(mask, Zs, 0.0j)
            _, y = scipy_istft(Zo, fs=SAMPLE_RATE,
                               nperseg=self.nperseg, noverlap=self.noverlap,
                               scaling='psd')
            # Extract the M samples corresponding to the current chunk
            out[:, ch] = y[self._tail_len:self._tail_len + M]

        self._src_tail = src[-self._tail_len:].copy()
        self._msk_tail = msk[-self._tail_len:].copy()
        return out


class StreamingSTFTEnsemble:
    """Streaming per-bin absmax selection in STFT domain (replaces ensemble.py max_fft).

    For two input signals, selects the complex STFT bin with larger absolute
    magnitude, then reconstructs via scipy istft.  Zero intrinsic delay --
    boundary handling via prepended tail from previous chunk.
    """

    def __init__(self, nfft: int = 2048, hop: int = 1024):
        self.nfft = nfft
        self.hop = hop
        self.noverlap = nfft - hop
        self._tail_len = self.noverlap
        self._a_tail: np.ndarray | None = None
        self._b_tail: np.ndarray | None = None

    def process(self, sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
        """sig_a, sig_b: (M, C). Returns (M, C)."""
        M, C = sig_a.shape
        if self._a_tail is None:
            self._a_tail = np.zeros((self._tail_len, C), dtype=np.float32)
            self._b_tail = np.zeros((self._tail_len, C), dtype=np.float32)

        pad_end = self.nfft
        a_ext = np.concatenate([self._a_tail, sig_a,
                                np.zeros((pad_end, C), dtype=np.float32)])
        b_ext = np.concatenate([self._b_tail, sig_b,
                                np.zeros((pad_end, C), dtype=np.float32)])

        out = np.empty((M, C), dtype=np.float32)
        for ch in range(C):
            _, _, Za = scipy_stft(a_ext[:, ch], fs=SAMPLE_RATE,
                                  nperseg=self.nfft, noverlap=self.noverlap,
                                  scaling='psd')
            _, _, Zb = scipy_stft(b_ext[:, ch], fs=SAMPLE_RATE,
                                  nperseg=self.nfft, noverlap=self.noverlap,
                                  scaling='psd')
            sel_b = np.abs(Zb) > np.abs(Za)
            Zo = np.where(sel_b, Zb, Za)
            _, y = scipy_istft(Zo, fs=SAMPLE_RATE,
                               nperseg=self.nfft, noverlap=self.noverlap,
                               scaling='psd')
            out[:, ch] = y[self._tail_len:self._tail_len + M]

        self._a_tail = sig_a[-self._tail_len:].copy()
        self._b_tail = sig_b[-self._tail_len:].copy()
        return out


class MultibandDecomposer:
    """3-filter multiband decomposition for table bias mode.

    Designs three multiband FIR filters (one per bias value 45/55/60) whose
    passbands match the bias-to-frequency-range table.  Perfect reconstruction
    is guaranteed: filter_45 is computed as the complement of filter_55 + filter_60,
    so filter_45(S) + filter_55(S) + filter_60(S) = delay(S) for any signal S.

    Total delay: D1 samples (single FIR group delay ≈ 1026).
    """

    # Bias 55 passbands: 1195-1370, 1607-1723, 8955-9591, 11020-12008, 13083-13780
    CUTOFFS_55 = [1195, 1370, 1607, 1723, 8955, 9591, 11020, 12008, 13083, 13780]
    # Bias 60 passbands: 1370-1607, 1723-8955, 9591-11020, 12008-13083
    CUTOFFS_60 = [1370, 1607, 1723, 8955, 9591, 11020, 12008, 13083]

    def __init__(self) -> None:
        taps_55 = design_multiband_fir(self.CUTOFFS_55, SAMPLE_RATE,
                                       pass_zero=False)
        taps_60 = design_multiband_fir(self.CUTOFFS_60, SAMPLE_RATE,
                                       pass_zero=False)

        # Complement: filter_45 = identity_delay - filter_55 - filter_60
        numtaps = len(taps_55)
        delay_kernel = np.zeros(numtaps, dtype=np.float32)
        delay_kernel[(numtaps - 1) // 2] = 1.0
        taps_45 = delay_kernel - taps_55 - taps_60

        self.firs = {
            45: StreamingFIR(taps_45),
            55: StreamingFIR(taps_55),
            60: StreamingFIR(taps_60),
        }
        self.D1 = (numtaps - 1) // 2  # 1026
        self.total_delay = self.D1

    def process(self, bias: int, chunk: np.ndarray) -> np.ndarray:
        """Filter chunk through the multiband FIR for the given bias value."""
        return self.firs[bias].process(chunk)


class PacketWeightProvider:
    """Maps VBR packet sizes to per-sample blend weights.

    For each chunk, returns (w_m4a, w_opus) arrays shaped (N, 1) that broadcast
    with (N, 2) stereo signals.  Uses np.searchsorted for O(M log P) lookup.

    Weight formula (bias=50 → no bias):
        bias_ratio = bias / (100 - bias)
        eff_opus = opus_size * bias_ratio
        w_opus = eff_opus / (eff_opus + m4a_size)
        w_m4a  = 1 - w_opus
    """

    def __init__(self,
                 m4a_packets: list[tuple[float, int]],
                 opus_packets: list[tuple[float, int]],
                 m4a_encoder_delay: int,
                 alignment_offset: int,
                 bias: int = 50):
        fs = SAMPLE_RATE

        # Build m4a sample starts: aligned_sample = round(pts*fs) + delay - skip
        m4a_skip = max(alignment_offset, 0)
        starts_m4a = []
        sizes_m4a = []
        for pts, sz in m4a_packets:
            s = round(pts * fs) + m4a_encoder_delay - m4a_skip
            starts_m4a.append(s)
            sizes_m4a.append(sz)

        # Build opus sample starts: normalize so first packet → sample 0
        opus_skip = max(-alignment_offset, 0)
        opus_base = round(opus_packets[0][0] * fs) if opus_packets else 0
        starts_opus = []
        sizes_opus = []
        for pts, sz in opus_packets:
            s = round(pts * fs) - opus_base - opus_skip
            starts_opus.append(s)
            sizes_opus.append(sz)

        self._m4a_starts = np.array(starts_m4a, dtype=np.int64)
        self._m4a_sizes = np.array(sizes_m4a, dtype=np.float32)
        self._opus_starts = np.array(starts_opus, dtype=np.int64)
        self._opus_sizes = np.array(sizes_opus, dtype=np.float32)

        # Bias mode
        if bias >= 100:
            self._mode = "all_opus"
        elif bias <= 0:
            self._mode = "all_m4a"
        else:
            self._mode = "dynamic"
            self._bias_ratio = np.float32(bias / (100 - bias))

    def get_weights(self, start_sample: int, num_samples: int,
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (w_m4a, w_opus) as (num_samples, 1) float32 arrays."""
        ones = np.ones((num_samples, 1), dtype=np.float32)
        zeros = np.zeros((num_samples, 1), dtype=np.float32)

        if self._mode == "all_opus":
            return zeros, ones
        if self._mode == "all_m4a":
            return ones, zeros

        indices = np.arange(start_sample, start_sample + num_samples,
                            dtype=np.int64)

        m4a_pkt = np.searchsorted(self._m4a_starts, indices, side="right") - 1
        opus_pkt = np.searchsorted(self._opus_starts, indices, side="right") - 1
        np.clip(m4a_pkt, 0, len(self._m4a_sizes) - 1, out=m4a_pkt)
        np.clip(opus_pkt, 0, len(self._opus_sizes) - 1, out=opus_pkt)

        m4a_sz = self._m4a_sizes[m4a_pkt]
        opus_sz = self._opus_sizes[opus_pkt]

        eff_opus = opus_sz * self._bias_ratio
        total = eff_opus + m4a_sz
        np.maximum(total, np.float32(1e-10), out=total)

        w_opus = (eff_opus / total).astype(np.float32).reshape(-1, 1)
        w_m4a = np.float32(1.0) - w_opus
        return w_m4a, w_opus


# ---------------------------------------------------------------------------
# M/S encode / decode
# ---------------------------------------------------------------------------
def ms_encode(stereo: np.ndarray) -> np.ndarray:
    """(M,2) L/R -> (M,2) Mid/Side.  M=(L+R)*0.5, S=(L-R)*0.5."""
    L = stereo[:, 0]
    R = stereo[:, 1]
    return np.column_stack(((L + R) * 0.5, (L - R) * 0.5))


def ms_decode(ms: np.ndarray) -> np.ndarray:
    """(M,2) Mid/Side -> (M,2) L/R.  L=M+S, R=M-S."""
    M = ms[:, 0]
    S = ms[:, 1]
    return np.column_stack((M + S, M - S))


# ---------------------------------------------------------------------------
# Pipeline state + per-chunk processing
# ---------------------------------------------------------------------------
class PipelineState:
    """Holds all stateful streaming processors for the combination pipeline."""

    def __init__(self, *, ensemble_mask: bool = False):
        fs = SAMPLE_RATE

        # ---- filter taps ----
        bhp850 = design_brickwall_fir("bhp", 850, fs)
        bhp15600 = design_brickwall_fir("bhp", 15600, fs)
        blp15750 = design_brickwall_fir("blp", 15750, fs)

        D1 = (len(bhp850) - 1) // 2  # 1026

        # ---- STFT processors (zero intrinsic delay) ----
        self.stft_masker = StreamingSTFTMasker(
            nperseg=1024, overlap_factor=8, threshold_db=-118.0)
        self.stft_ensemble = StreamingSTFTEnsemble(nfft=1024, hop=512)

        # ---- optional: ensemble for mask mixing (step 6) ----
        self.ensemble_mask = ensemble_mask
        if ensemble_mask:
            self.stft_ensemble_mask = StreamingSTFTEnsemble(nfft=1024, hop=512)

        # ---- delay lines ----
        # FIR delay compensation (Group A → Group B)
        self.delay_m4a = StreamingDelay(D1)
        self.delay_opus = StreamingDelay(D1)
        # Weight delay: Group A weights → Group B (for step 6 mask mixing)
        self.delay_weights = StreamingDelay(D1)
        # BHP crossover delay compensation (Group B → Group C)
        self.delay_avg2p = StreamingDelay(D1)
        # BLP crossover: bring opus_iblp from Group B to Group D
        self.delay_opus_iblp = StreamingDelay(2 * D1)

        # ---- FIR filters ----
        self.fir_bhp_avg = StreamingFIR(bhp850)
        self.fir_bhp_m4a = StreamingFIR(bhp850)

        self.fir_bhp_ens = StreamingFIR(bhp15600)
        self.fir_bhp_avg2p = StreamingFIR(bhp15600)

        self.fir_blp_avg3 = StreamingFIR(blp15750)
        self.fir_blp_opus = StreamingFIR(blp15750)

        # ---- gain ----
        self.gain_neg1db = np.float32(10.0 ** (-1.0 / 20.0))

        # ---- pipeline latency in samples (for flush) ----
        self.total_delay = 3 * D1  # 3078
        self.D1 = D1


class _SharedTableState:
    """Bias-independent stateful processors for table mode (run once per chunk)."""

    def __init__(self) -> None:
        fs = SAMPLE_RATE
        bhp850 = design_brickwall_fir("bhp", 850, fs)
        blp15750 = design_brickwall_fir("blp", 15750, fs)
        D1 = (len(bhp850) - 1) // 2

        self.delay_m4a = StreamingDelay(D1)
        self.delay_opus = StreamingDelay(D1)
        self.fir_bhp_m4a = StreamingFIR(bhp850)
        self.stft_masker = StreamingSTFTMasker(
            nperseg=1024, overlap_factor=8, threshold_db=-118.0)
        self.fir_blp_opus = StreamingFIR(blp15750)
        self.delay_opus_iblp = StreamingDelay(2 * D1)

        self.D1 = D1
        self.total_delay = 3 * D1  # same pipeline delay structure


class _PerBiasTableState:
    """Bias-dependent stateful processors for table mode (one per bias value)."""

    def __init__(self, *, ensemble_mask: bool = False) -> None:
        fs = SAMPLE_RATE
        bhp850 = design_brickwall_fir("bhp", 850, fs)
        bhp15600 = design_brickwall_fir("bhp", 15600, fs)
        blp15750 = design_brickwall_fir("blp", 15750, fs)
        D1 = (len(bhp850) - 1) // 2

        self.fir_bhp_avg = StreamingFIR(bhp850)
        self.delay_weights = StreamingDelay(D1)

        self.stft_ensemble = StreamingSTFTEnsemble(nfft=1024, hop=512)
        self.ensemble_mask = ensemble_mask
        if ensemble_mask:
            self.stft_ensemble_mask = StreamingSTFTEnsemble(nfft=1024, hop=512)

        self.fir_bhp_ens = StreamingFIR(bhp15600)
        self.fir_bhp_avg2p = StreamingFIR(bhp15600)
        self.delay_avg2p = StreamingDelay(D1)
        self.fir_blp_avg3 = StreamingFIR(blp15750)

        self.gain_neg1db = np.float32(10.0 ** (-1.0 / 20.0))


def process_chunk(m4a_raw: np.ndarray, opus_raw: np.ndarray,
                  st: PipelineState, *,
                  w_m4a: np.ndarray | None = None,
                  w_opus: np.ndarray | None = None) -> np.ndarray:
    """Run one chunk through the full combination pipeline.

    m4a_raw, opus_raw: (M, 2) float32 stereo.
    Returns (M, 2) float32 stereo output.

    Delay groups (all signals within a group are time-aligned):
      A  = 0              raw inputs, averaged
      B  = D1             after FIR bhp / delay; mask & ensemble inputs
      C  = 2*D1           after bhp crossover
      D  = 3*D1           after blp crossover (opus hi + averaged3 lo)
    """
    # Step 1 – M/S encode
    m4a_ms = ms_encode(m4a_raw)
    opus_ms = ms_encode(opus_raw)

    # Delay unfiltered copies to Group B (D1)
    m4a_ms_d = st.delay_m4a.process(m4a_ms)
    opus_ms_d = st.delay_opus.process(opus_ms)

    # Step 2 – average (Group A), static or dynamic weights
    if w_m4a is not None:
        averaged = m4a_ms * w_m4a + opus_ms * w_opus
    else:
        averaged = (m4a_ms + opus_ms) * np.float32(0.5)

    # Step 3 – bhp 850 Hz on averaged → Group B
    averaged_bhp = st.fir_bhp_avg.process(averaged)

    # Step 4 – bhp 850 Hz on m4a → Group B; complementary split
    m4a_bhp = st.fir_bhp_m4a.process(m4a_ms)
    m4a_ibhp = m4a_ms_d - m4a_bhp
    averaged2 = m4a_ibhp + averaged_bhp

    # Step 5 – mask extraction (zero-delay STFT, stays in Group B)
    opus_patch = st.stft_masker.process(opus_ms_d, m4a_ms_d)

    # Step 6 – patch mask into average (Group B); multiplier = m4a weight
    if st.ensemble_mask:
        averaged2_p = st.stft_ensemble_mask.process(averaged2, opus_patch)
        if w_m4a is not None:
            st.delay_weights.process(w_m4a)  # keep delay state in sync
    elif w_m4a is not None:
        w_m4a_B = st.delay_weights.process(w_m4a)
        averaged2_p = (opus_patch * w_m4a_B) + averaged2
    else:
        averaged2_p = (opus_patch * np.float32(0.5)) + averaged2

    # Step 7 – ensemble max_fft (zero-delay STFT, stays in Group B)
    ens_result = st.stft_ensemble.process(averaged2_p, m4a_ms_d)

    # Step 8 – apply -1 dB gain
    ens_result = ens_result * st.gain_neg1db

    # Step 9 – bhp 15600 on ensemble result (B → C, adds D1)
    ens_bhp = st.fir_bhp_ens.process(ens_result)

    # Step 10 – bhp 15600 on averaged2_p (B → C); complementary crossover
    avg2p_bhp = st.fir_bhp_avg2p.process(averaged2_p)
    averaged2_p_d = st.delay_avg2p.process(averaged2_p)
    averaged2_p_ibhp = averaged2_p_d - avg2p_bhp
    averaged3 = averaged2_p_ibhp + ens_bhp

    # Step 11 – blp crossover: opus hi-freq + averaged3 lo-freq (C → D)
    averaged3_blp = st.fir_blp_avg3.process(averaged3)       # → 3*D1
    opus_blp = st.fir_blp_opus.process(opus_ms)              # → D1
    opus_iblp = opus_ms_d - opus_blp                         # → D1
    opus_iblp_d = st.delay_opus_iblp.process(opus_iblp)      # → 3*D1
    averaged3_p = opus_iblp_d + averaged3_blp                 # → 3*D1

    # Step 12 – M/S decode
    return ms_decode(averaged3_p)


def _precompute_table_shared(
        m4a_raw: np.ndarray, opus_raw: np.ndarray,
        sst: _SharedTableState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Compute bias-independent pipeline work once for table mode.

    Returns (m4a_ms, opus_ms, m4a_ms_d, m4a_ibhp, opus_patch, opus_iblp_d).
    """
    m4a_ms = ms_encode(m4a_raw)
    opus_ms = ms_encode(opus_raw)

    m4a_ms_d = sst.delay_m4a.process(m4a_ms)
    opus_ms_d = sst.delay_opus.process(opus_ms)

    m4a_bhp = sst.fir_bhp_m4a.process(m4a_ms)
    m4a_ibhp = m4a_ms_d - m4a_bhp

    opus_patch = sst.stft_masker.process(opus_ms_d, m4a_ms_d)

    opus_blp = sst.fir_blp_opus.process(opus_ms)
    opus_iblp = opus_ms_d - opus_blp
    opus_iblp_d = sst.delay_opus_iblp.process(opus_iblp)

    return (m4a_ms, opus_ms, m4a_ms_d, m4a_ibhp, opus_patch, opus_iblp_d)


def _process_chunk_per_bias(
        shared: tuple, pst: _PerBiasTableState, *,
        w_m4a: np.ndarray | None = None,
        w_opus: np.ndarray | None = None,
) -> np.ndarray:
    """Run bias-dependent pipeline steps using precomputed shared data.

    shared: tuple from _precompute_table_shared.
    Returns (M, 2) float32 stereo output.
    """
    m4a_ms, opus_ms, m4a_ms_d, m4a_ibhp, opus_patch, opus_iblp_d = shared

    # Step 2 – weighted average (Group A)
    if w_m4a is not None:
        averaged = m4a_ms * w_m4a + opus_ms * w_opus
    else:
        averaged = (m4a_ms + opus_ms) * np.float32(0.5)

    # Step 3 – bhp 850 Hz on averaged → Group B
    averaged_bhp = pst.fir_bhp_avg.process(averaged)

    # Step 4 – combine with shared m4a_ibhp
    averaged2 = m4a_ibhp + averaged_bhp

    # Step 6 – patch mask into average (Group B)
    if pst.ensemble_mask:
        averaged2_p = pst.stft_ensemble_mask.process(averaged2, opus_patch)
        if w_m4a is not None:
            pst.delay_weights.process(w_m4a)  # keep delay state in sync
    elif w_m4a is not None:
        w_m4a_B = pst.delay_weights.process(w_m4a)
        averaged2_p = (opus_patch * w_m4a_B) + averaged2
    else:
        averaged2_p = (opus_patch * np.float32(0.5)) + averaged2

    # Step 7 – ensemble max_fft (Group B)
    ens_result = pst.stft_ensemble.process(averaged2_p, m4a_ms_d)

    # Step 8 – apply -1 dB gain
    ens_result = ens_result * pst.gain_neg1db

    # Step 9 – bhp 15600 on ensemble result (B → C)
    ens_bhp = pst.fir_bhp_ens.process(ens_result)

    # Step 10 – bhp 15600 on averaged2_p (B → C); complementary crossover
    avg2p_bhp = pst.fir_bhp_avg2p.process(averaged2_p)
    averaged2_p_d = pst.delay_avg2p.process(averaged2_p)
    averaged2_p_ibhp = averaged2_p_d - avg2p_bhp
    averaged3 = averaged2_p_ibhp + ens_bhp

    # Step 11 – blp crossover with shared opus_iblp_d (C → D)
    averaged3_blp = pst.fir_blp_avg3.process(averaged3)
    averaged3_p = opus_iblp_d + averaged3_blp

    # Step 12 – M/S decode
    return ms_decode(averaged3_p)


# ---------------------------------------------------------------------------
# WAV writer
# ---------------------------------------------------------------------------
class WavWriter:
    """Writes 32-bit float WAV in chunks. Patches header sizes on finalize."""

    def __init__(self, path: str | Path | None, *, stdout: bool = False):
        self.is_stdout = stdout
        if stdout:
            self._f = sys.stdout.buffer
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, "wb")
        self.data_bytes = 0
        self._write_header(0xFFFFFFFF if stdout else 0)

    def _write_header(self, data_size: int) -> None:
        block_align = CHANNELS * BYTES_PER_SAMPLE
        byte_rate = SAMPLE_RATE * block_align
        riff_size = 0xFFFFFFFF if data_size == 0xFFFFFFFF else 36 + data_size
        hdr = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", riff_size, b"WAVE",
            b"fmt ", 16,
            3,               # WAVE_FORMAT_IEEE_FLOAT
            CHANNELS, SAMPLE_RATE, byte_rate,
            block_align, BYTES_PER_SAMPLE * 8,
            b"data", data_size,
        )
        self._f.write(hdr)

    def write(self, chunk: np.ndarray) -> None:
        """Write (M, 2) float32 chunk."""
        raw = chunk.astype(np.float32).tobytes()
        self._f.write(raw)
        self.data_bytes += len(raw)

    def finalize(self) -> None:
        if not self.is_stdout:
            self._f.seek(4)
            self._f.write(struct.pack("<I", 36 + self.data_bytes))
            self._f.seek(40)
            self._f.write(struct.pack("<I", self.data_bytes))
            self._f.close()
        else:
            self._f.flush()


class EncoderWriter:
    """Pipes WAV-formatted PCM to an external encoder subprocess."""

    def __init__(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.data_bytes = 0
        self._write_wav_header()

    def _write_wav_header(self) -> None:
        block_align = CHANNELS * BYTES_PER_SAMPLE
        byte_rate = SAMPLE_RATE * block_align
        hdr = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 0xFFFFFFFF, b"WAVE",
            b"fmt ", 16,
            3,               # WAVE_FORMAT_IEEE_FLOAT
            CHANNELS, SAMPLE_RATE, byte_rate,
            block_align, BYTES_PER_SAMPLE * 8,
            b"data", 0xFFFFFFFF,
        )
        self._proc.stdin.write(hdr)

    def write(self, chunk: np.ndarray) -> None:
        """Write (M, 2) float32 chunk to encoder stdin."""
        raw = chunk.astype(np.float32).tobytes()
        try:
            self._proc.stdin.write(raw)
        except BrokenPipeError:
            stderr = self._proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"Encoder died: {stderr.strip()}")
        self.data_bytes += len(raw)

    def finalize(self) -> None:
        self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            stderr = self._proc.stderr.read().decode(errors="replace")
            print(f"[encoder] Warning: exited with code {rc}",
                  file=sys.stderr)
            if stderr:
                print(f"[encoder] {stderr.strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Encoder command builder
# ---------------------------------------------------------------------------
def _build_encoder_cmd(output_path: Path, *,
                       format_name: str,
                       bitrate: str | None = None,
                       bit_depth: int | None = None,
                       custom_encoder: str | None = None,
                       metadata: dict[str, str] | None = None,
                       thumbnail_path: Path | None = None,
                       extra_encoder_args: list[str] | None = None,
                       ) -> list[str]:
    """Build the encoder command list for a given format configuration."""
    out = str(output_path)

    # Custom encoder: user provides the full command with {output} placeholder
    if custom_encoder is not None:
        if "{output}" not in custom_encoder:
            raise ValueError("--encoder command must contain {output} placeholder")
        return shlex.split(custom_encoder.replace("{output}", out))

    # Built-in ffmpeg encoding
    fmt = FORMAT_DEFAULTS[format_name]
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    # Thumbnail input (must come before WAV stdin input)
    if thumbnail_path is not None:
        cmd.extend(["-i", str(thumbnail_path)])

    # WAV from stdin
    cmd.extend(["-f", "wav", "-i", "-"])

    # Stream mapping when thumbnail is present
    if thumbnail_path is not None:
        cmd.extend(["-map", "1:a", "-map", "0:v",
                    "-c:v", "copy", "-disposition:v", "attached_pic"])

    # Audio codec
    cmd.extend(["-c:a", fmt["codec"]])

    # Bitrate (lossy codecs)
    br = bitrate or fmt.get("default_bitrate")
    if br is not None:
        cmd.extend(["-b:a", br])

    # Bit depth (lossless codecs like flac)
    bd = bit_depth or fmt.get("default_bit_depth")
    if bd is not None and format_name in ("flac",):
        cmd.extend(["-sample_fmt", f"s{bd}"])

    # Metadata tags
    if metadata:
        for tag, val in metadata.items():
            cmd.extend(["-metadata", f"{tag}={val}"])

    # Extra user args
    if extra_encoder_args:
        cmd.extend(extra_encoder_args)

    cmd.extend(["-y", out])
    return cmd


# ---------------------------------------------------------------------------
# Preset system
# ---------------------------------------------------------------------------
def _get_config_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "ytaudiocomb"
    return Path(os.environ.get("XDG_CONFIG_HOME",
                str(Path.home() / ".config"))) / "ytaudiocomb"


def _load_presets() -> dict:
    p = _get_config_dir() / "presets.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_preset(name: str, config: dict) -> None:
    d = _get_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / "presets.json"
    presets = _load_presets()
    presets[name] = config
    with open(p, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def _list_presets() -> None:
    presets = _load_presets()
    if not presets:
        print("No saved presets.", file=sys.stderr)
        return
    print("Saved presets:", file=sys.stderr)
    for name, cfg in presets.items():
        parts = [f"format={cfg.get('format', '?')}"]
        if "bitrate" in cfg:
            parts.append(f"bitrate={cfg['bitrate']}")
        if "bit_depth" in cfg:
            parts.append(f"bit_depth={cfg['bit_depth']}")
        if "encoder" in cfg:
            parts.append("encoder=<custom>")
        if cfg.get("embed_metadata"):
            parts.append("embed_metadata")
        if cfg.get("embed_thumbnail"):
            parts.append("embed_thumbnail")
        if "output_template" in cfg:
            parts.append(f"template={cfg['output_template']}")
        print(f"  {name}: {', '.join(parts)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Metadata + thumbnail helpers
# ---------------------------------------------------------------------------
def _extract_video_id(stem: str) -> str | None:
    """Extract video ID from a filename stem like 'Title [dQw4w9WgXcQ]'."""
    m = re.search(r"\[([a-zA-Z0-9_-]+)\]$", stem)
    return m.group(1) if m else None


def _find_info_json(source_path: Path) -> dict | None:
    """Find and parse the .info.json for a source file."""
    video_id = _extract_video_id(source_path.stem)
    if video_id is None:
        return None
    for d in (source_path.parent, source_path.parent.parent):
        for p in d.glob(f"*[[]{ video_id }[]].info.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _find_thumbnail(source_dir: Path, video_id: str) -> Path | None:
    """Find a thumbnail image for the given video ID."""
    for ext in ("webp", "jpg", "png"):
        for p in source_dir.glob(f"*[[]{ video_id }[]].{ext}"):
            return p
    return None


def _extract_metadata_tags(info: dict) -> dict[str, str]:
    """Extract standard audio tags from yt-dlp info.json."""
    tags: dict[str, str] = {}
    seen_tags: set[str] = set()
    for info_key, tag_name in METADATA_TAG_MAP.items():
        if tag_name in seen_tags:
            continue
        val = info.get(info_key)
        if val is not None and str(val).strip():
            val = str(val).strip()
            if info_key == "upload_date" and len(val) == 8:
                val = f"{val[:4]}-{val[4:6]}-{val[6:8]}"
            tags[tag_name] = val
            seen_tags.add(tag_name)
    return tags


def _sanitize_filename(s: str) -> str:
    """Remove filesystem-unsafe characters."""
    s = re.sub(r'[/\\:*?"<>|]', "", s)
    s = re.sub(r"\s+", " ", s).strip().strip(".")
    return s


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------
def _resolve_output_path(m4a_path: Path, combined_dir: Path, ext: str,
                         output_template: str | None = None,
                         info: dict | None = None) -> Path:
    """Build the output file path, optionally using a metadata template."""
    if output_template is not None and info is not None:
        def _repl(m: re.Match) -> str:
            field = m.group(1)
            val = info.get(field, "")
            return _sanitize_filename(str(val)) if val else ""
        stem = re.sub(r"%\(([^)]+)\)s", _repl, output_template).strip()
        if not stem:
            stem = m4a_path.stem
    else:
        stem = m4a_path.stem
    return combined_dir / f"{stem}.{ext}"


# ---------------------------------------------------------------------------
# FFmpeg subprocess helpers
# ---------------------------------------------------------------------------
def _start_ffmpeg_m4a(path: Path, *, quiet: bool = False) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-c:a", "pcm_f32le", "-vn",
            "-f", "f32le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def _start_ffmpeg_webm(path: Path, *, quiet: bool = False) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-c:a", "pcm_f32le", "-vn",
            "-af", "aresample=44100:resampler=soxr:cutoff=1",
            "-f", "f32le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def _read_chunk(pipe, chunk_size: int) -> np.ndarray | None:
    """Read one chunk of interleaved stereo float32 from an ffmpeg pipe.

    Returns (M, 2) float32 or None on EOF.
    """
    nbytes = chunk_size * FRAME_BYTES
    raw = pipe.stdout.read(nbytes)
    if not raw:
        return None
    samples = np.frombuffer(raw, dtype=np.float32).copy()
    n = len(samples) // CHANNELS
    if n == 0:
        return None
    return samples[: n * CHANNELS].reshape(n, CHANNELS)


def _probe_duration(path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _read_ffprobe_packets(path: Path) -> list[tuple[float, int]]:
    """Read (pts_time, packet_size) for every audio packet via ffprobe.

    Extra fields (e.g. side data on first m4a line) are silently ignored.
    Returns empty list on failure.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_packets", "-select_streams", "a:0",
             "-show_entries", "packet=pts_time,size", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    packets: list[tuple[float, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            packets.append((float(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return packets


def _read_m4a_ffprobe(path: Path) -> tuple[list[tuple[float, int]], int]:
    """Read m4a packets and detect encoder delay from first-packet metadata.

    Returns (packets, encoder_delay_samples).
    encoder_delay is the 'Skip Samples' value if present (e.g. 1600), else 0.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_packets", "-select_streams", "a:0",
             "-show_entries", "packet=pts_time,size", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return [], 0

    encoder_delay = 0
    packets: list[tuple[float, int]] = []
    for i, line in enumerate(result.stdout.splitlines()):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        # First line may have side data: pts,size,Skip Samples,1600,...
        if i == 0 and len(parts) >= 4 and parts[2].strip() == "Skip Samples":
            try:
                encoder_delay = int(parts[3])
            except ValueError:
                pass
        try:
            packets.append((float(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return packets, encoder_delay


class ProgressBar:
    """Simple stderr progress bar for the decoding stage."""

    def __init__(self, total_samples: int | None):
        self._total = total_samples
        self._written = 0
        self._start = time.monotonic()
        self._last_update = 0.0

    def update(self, samples: int) -> None:
        self._written += samples
        now = time.monotonic()
        if now - self._last_update < 0.15:
            return
        self._last_update = now
        elapsed = now - self._start
        rate = self._written / elapsed if elapsed > 0 else 0
        written_s = self._written / SAMPLE_RATE

        if self._total and self._total > 0:
            pct = min(self._written / self._total * 100, 100)
            total_s = self._total / SAMPLE_RATE
            bar_w = 30
            filled = int(bar_w * pct / 100)
            bar = "█" * filled + "░" * (bar_w - filled)
            eta = (self._total - self._written) / rate if rate > 0 else 0
            print(f"\r  [{bar}] {pct:5.1f}%  {written_s:.1f}s/{total_s:.1f}s  "
                  f"ETA {eta:.0f}s   ", end="", file=sys.stderr)
        else:
            print(f"\r  {written_s:.1f}s decoded   ", end="", file=sys.stderr)

    def finish(self) -> None:
        elapsed = time.monotonic() - self._start
        written_s = self._written / SAMPLE_RATE
        print(f"\r  {written_s:.1f}s decoded in {elapsed:.1f}s"
              + " " * 30, file=sys.stderr)


def detect_alignment_offset(m4a_path: Path, webm_path: Path) -> int:
    """Detect m4a-to-webm sample offset via cross-correlation.

    Decodes ~15 s from both files, finds the first non-silent region, and
    cross-correlates a 5 s window to measure the lag.

    Returns positive int → skip that many samples from m4a.
    Returns negative int → skip abs() samples from webm.
    Returns 0 → sources are already aligned.

    Typical values: 0 (YouTube Music origin) or 1600 (regular YouTube).
    """
    PROBE_SECONDS = 15.0
    WINDOW_SECONDS = 5.0
    MAX_OFFSET = 3200
    SILENCE_THRESH = 0.005
    BLK = 1024

    probe_n = int(PROBE_SECONDS * SAMPLE_RATE)

    proc_m = _start_ffmpeg_m4a(m4a_path, quiet=True)
    proc_w = _start_ffmpeg_webm(webm_path, quiet=True)
    m4a = _read_chunk(proc_m, probe_n)
    webm = _read_chunk(proc_w, probe_n)
    # Terminate early -- we only needed a short probe, not the full file
    proc_m.terminate()
    proc_w.terminate()
    proc_m.stdout.close()
    proc_w.stdout.close()
    proc_m.wait()
    proc_w.wait()

    if m4a is None or webm is None:
        return 0

    n = min(len(m4a), len(webm))
    m_mono = (m4a[:n, 0] + m4a[:n, 1]).astype(np.float64)
    w_mono = (webm[:n, 0] + webm[:n, 1]).astype(np.float64)

    # Find first non-silent block in either signal
    start = 0
    for i in range(0, n - BLK, BLK):
        if max(np.sqrt(np.mean(m_mono[i:i + BLK] ** 2)),
               np.sqrt(np.mean(w_mono[i:i + BLK] ** 2))) > SILENCE_THRESH:
            start = max(0, i - BLK)
            break

    win_n = min(int(WINDOW_SECONDS * SAMPLE_RATE), n - start)
    if win_n < SAMPLE_RATE:
        return 0

    ref = w_mono[start:start + win_n]
    test = m_mono[start:start + win_n]
    ref -= np.mean(ref)
    test -= np.mean(test)

    # FFT cross-correlation
    corr = fftconvolve(ref, test[::-1], mode="full")
    mid = len(test) - 1  # index of zero lag

    lo = mid - MAX_OFFSET
    hi = mid + MAX_OFFSET + 1
    search = corr[lo:hi]
    peak_idx = int(np.argmax(np.abs(search)))
    # peak_idx within search → lag = peak_idx - MAX_OFFSET
    # positive lag means test (m4a) is behind ref (webm) → need to advance m4a
    # offset to skip from m4a = -lag
    offset = MAX_OFFSET - peak_idx
    return offset


def detect_offset_ffprobe(encoder_delay: int) -> int:
    """Detect m4a alignment offset from ffprobe encoder delay metadata.

    If encoder_delay > 0 (first packet had 'Skip Samples' side data),
    the container properly marks the AAC priming samples → offset = 0.

    If encoder_delay == 0 (no 'Skip Samples'), the AAC priming samples
    are not marked and ffmpeg outputs them → offset = 1600.
    """
    return 0 if encoder_delay > 0 else 1600


# ---------------------------------------------------------------------------
# yt-dlp download + file matching
# ---------------------------------------------------------------------------
def download_streams(url: str | None, extra_args: list[str],
                     streams_dir: Path = STREAMS_DIR,
                     inject_args: list[str] | None = None) -> None:
    """Download format 140 (m4a) and 251 (webm) via yt-dlp.

    If url is None, extra_args must contain -a/--batch-file pointing to a
    file of URLs.  inject_args are prepended (e.g. --write-info-json).
    """
    # Check if user already supplied -o in extra_args
    has_user_output = any(
        a == "-o" or a.startswith("-o") and len(a) > 2
        for a in extra_args
    )
    streams_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "--output-na-placeholder", "",
    ]
    if not has_user_output:
        cmd.extend([
            "-o", str(streams_dir / "%(playlist_index)02d%(playlist_index& - |)s%(title)s [%(id)s].%(ext)s"),
        ])
    cmd.extend(["-f", "251,140"])
    if inject_args:
        cmd.extend(inject_args)
    if extra_args:
        cmd.extend(extra_args)
    if url is not None:
        cmd.append(url)

    print("[yt-dlp] Downloading formats 140 + 251...", file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[yt-dlp] Warning: download exited with code {rc}",
              file=sys.stderr)


def find_matches(directory: Path) -> list[tuple[Path, Path]]:
    """Return list of (m4a_path, webm_path) pairs sharing the same stem."""
    m4a_files = {p.stem: p for p in directory.glob("*.m4a")}
    webm_files = {p.stem: p for p in directory.glob("*.webm")}
    pairs = []
    for stem in sorted(set(m4a_files) & set(webm_files)):
        pairs.append((m4a_files[stem], webm_files[stem]))
    return pairs


# ---------------------------------------------------------------------------
# Core processing loop for one matched pair
# ---------------------------------------------------------------------------
def process_pair(m4a_path: Path, webm_path: Path, output_path: Path | None,
                 *, stdout: bool = False, chunk_size: int = DEFAULT_CHUNK,
                 offset_mode: str = "auto",
                 forced_offset: int | None = None,
                 static_avg: bool = False,
                 bias: str | int = "table",
                 ensemble_mask: bool = False,
                 encoder_cmd: list[str] | None = None) -> None:
    """Decode, combine, and write one m4a+webm pair."""
    is_table = bias == "table"
    stem = m4a_path.stem
    if output_path is None and not stdout:
        COMBINED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = COMBINED_DIR / f"{stem}.wav"

    # --- Phase 1: ffprobe packet data ---
    print(f"[process] {stem}: reading packet data...", file=sys.stderr)
    m4a_packets, encoder_delay = _read_m4a_ffprobe(m4a_path)
    opus_packets = [] if static_avg else _read_ffprobe_packets(webm_path)

    # --- Phase 2: offset detection ---
    if forced_offset is not None:
        offset = forced_offset
        print(f"[process] {stem}: using forced offset {offset}", file=sys.stderr)
    elif offset_mode == "xcorr":
        print(f"[process] {stem}: detecting alignment (xcorr)...",
              file=sys.stderr)
        offset = detect_alignment_offset(m4a_path, webm_path)
    else:  # "auto" = ffprobe
        offset = detect_offset_ffprobe(encoder_delay)
        print(f"[process] {stem}: ffprobe offset={offset} "
              f"(encoder_delay={encoder_delay})", file=sys.stderr)

    if offset > 0:
        print(f"[process] {stem}: m4a leads by {offset} samples, skipping",
              file=sys.stderr)
    elif offset < 0:
        print(f"[process] {stem}: webm leads by {-offset} samples, skipping",
              file=sys.stderr)

    # --- Phase 3: weight provider(s) ---
    if is_table:
        if static_avg or not m4a_packets or not opus_packets:
            weight_providers = {b: None for b in TABLE_BIASES}
            if not static_avg and (not m4a_packets or not opus_packets):
                print(f"[process] {stem}: packet data unavailable, using "
                      "static averaging", file=sys.stderr)
        else:
            weight_providers = {}
            for b in TABLE_BIASES:
                weight_providers[b] = PacketWeightProvider(
                    m4a_packets, opus_packets,
                    m4a_encoder_delay=encoder_delay,
                    alignment_offset=offset,
                    bias=b,
                )
            print(f"[process] {stem}: table mode, dynamic weights "
                  f"(biases={TABLE_BIASES}, "
                  f"{len(m4a_packets)} m4a + {len(opus_packets)} opus "
                  f"packets)", file=sys.stderr)
    else:
        if static_avg or not m4a_packets or not opus_packets:
            weight_provider = None
            if not static_avg and (not m4a_packets or not opus_packets):
                print(f"[process] {stem}: packet data unavailable, using "
                      "static averaging", file=sys.stderr)
        else:
            weight_provider = PacketWeightProvider(
                m4a_packets, opus_packets,
                m4a_encoder_delay=encoder_delay,
                alignment_offset=offset,
                bias=bias,
            )
            print(f"[process] {stem}: dynamic weights (bias={bias}, "
                  f"{len(m4a_packets)} m4a + {len(opus_packets)} opus "
                  f"packets)", file=sys.stderr)

    # Probe duration for progress bar
    dur = _probe_duration(m4a_path)
    total_samples = int(dur * SAMPLE_RATE) if dur else None

    print(f"[process] {stem}: decoding...", file=sys.stderr)
    proc_m4a = _start_ffmpeg_m4a(m4a_path)
    proc_webm = _start_ffmpeg_webm(webm_path)

    # Discard leading samples from the source that is ahead
    if offset > 0:
        _read_chunk(proc_m4a, offset)
    elif offset < 0:
        _read_chunk(proc_webm, -offset)

    if stdout:
        writer = WavWriter(None, stdout=True)
    elif encoder_cmd is not None:
        writer = EncoderWriter(encoder_cmd)
    else:
        writer = WavWriter(output_path)
    if is_table:
        all_static = all(wp is None for wp in weight_providers.values())
        if all_static:
            # All pipelines identical under static averaging; run one
            state = PipelineState(ensemble_mask=ensemble_mask)
            total_delay = state.total_delay  # 3*D1
            weight_provider = None
            is_table = False
            print(f"[process] {stem}: table + static-avg -> single pipeline",
                  file=sys.stderr)
        else:
            shared_st = _SharedTableState()
            per_bias_states = {b: _PerBiasTableState(
                                      ensemble_mask=ensemble_mask)
                               for b in TABLE_BIASES}
            decomposer = MultibandDecomposer()
            total_delay = shared_st.total_delay + decomposer.total_delay
    else:
        state = PipelineState(ensemble_mask=ensemble_mask)
        total_delay = state.total_delay
    progress = ProgressBar(total_samples)

    total_written = 0
    input_pos = 0  # tracks aligned input sample position for weight lookup
    skip_remaining = total_delay  # skip pipeline latency at start
    end_pad = 0  # padding added to last chunk (reduces flush keep)
    try:
        while True:
            m4a_chunk = _read_chunk(proc_m4a, chunk_size)
            opus_chunk = _read_chunk(proc_webm, chunk_size)

            if m4a_chunk is None and opus_chunk is None:
                break

            # Handle one stream ending before the other (trim to shorter)
            if m4a_chunk is None:
                break
            if opus_chunk is None:
                break

            min_len = min(m4a_chunk.shape[0], opus_chunk.shape[0])
            m4a_chunk = m4a_chunk[:min_len]
            opus_chunk = opus_chunk[:min_len]

            # Pad last (short) chunk to multiple of 1024 for STFT alignment.
            # Do NOT trim the output -- due to the pipeline delay, the padded
            # output positions still process real (earlier) input samples.
            # Instead, reduce the flush keep amount by the same padding.
            remainder = min_len % 1024
            if remainder != 0:
                pad = 1024 - remainder
                m4a_chunk = np.pad(m4a_chunk, ((0, pad), (0, 0)))
                opus_chunk = np.pad(opus_chunk, ((0, pad), (0, 0)))
                end_pad = pad

            # Get per-sample weights and process chunk
            if is_table:
                shared = _precompute_table_shared(m4a_chunk, opus_chunk,
                                                  shared_st)
                out = np.zeros((m4a_chunk.shape[0], CHANNELS),
                               dtype=np.float32)
                for b in TABLE_BIASES:
                    wp = weight_providers[b]
                    if wp is not None:
                        wm, wo = wp.get_weights(input_pos,
                                                m4a_chunk.shape[0])
                    else:
                        wm, wo = None, None
                    pipe_out = _process_chunk_per_bias(shared,
                                                      per_bias_states[b],
                                                      w_m4a=wm, w_opus=wo)
                    out += decomposer.process(b, pipe_out)
            else:
                if weight_provider is not None:
                    w_m4a, w_opus = weight_provider.get_weights(
                        input_pos, m4a_chunk.shape[0])
                else:
                    w_m4a, w_opus = None, None

                out = process_chunk(m4a_chunk, opus_chunk, state,
                                    w_m4a=w_m4a, w_opus=w_opus)

            input_pos += min_len  # advance by real samples, not padding

            # Discard leading samples produced while pipeline delay lines fill
            if skip_remaining > 0:
                if out.shape[0] <= skip_remaining:
                    skip_remaining -= out.shape[0]
                    continue
                out = out[skip_remaining:]
                skip_remaining = 0

            writer.write(out)
            total_written += out.shape[0]
            progress.update(out.shape[0])

        # Flush pipeline: push zeros to drain delay lines and filter tails.
        # Subtract end_pad because those zeros already entered the pipeline
        # during the last chunk's padding.
        flush_keep = total_delay - end_pad
        if flush_keep > 0:
            flush_len = flush_keep + 1024  # extra for STFT OLA tails
            flush_len = ((flush_len + 1023) // 1024) * 1024
            zeros = np.zeros((flush_len, CHANNELS), dtype=np.float32)

            if is_table:
                shared = _precompute_table_shared(zeros, zeros, shared_st)
                flush_out = np.zeros((flush_len, CHANNELS),
                                     dtype=np.float32)
                for b in TABLE_BIASES:
                    pipe_out = _process_chunk_per_bias(shared,
                                                      per_bias_states[b])
                    flush_out += decomposer.process(b, pipe_out)
            else:
                flush_out = process_chunk(zeros, zeros, state)

            keep = min(flush_out.shape[0], flush_keep)
            writer.write(flush_out[:keep])
            total_written += keep

        progress.finish()

    finally:
        proc_m4a.stdout.close()
        proc_webm.stdout.close()
        proc_m4a.wait()
        proc_webm.wait()

    writer.finalize()
    duration = total_written / SAMPLE_RATE
    dest = "stdout" if stdout else str(output_path)
    print(f"[process] {stem}: wrote {duration:.1f}s to {dest}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and combine YouTube audio formats",
        add_help=True,
    )
    parser.add_argument(
        "url", nargs="?", default=None,
        help="YouTube URL (youtube.com or youtu.be). "
             "Omit to process existing files in streams/yt/.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Write output WAV to stdout instead of file.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK,
        help=f"Samples per processing chunk (default {DEFAULT_CHUNK}). "
             "Must be divisible by 1024.",
    )
    parser.add_argument(
        "--dont-keep", action="store_true",
        help="Delete source files after processing (default: move to archive).",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep source files in place (don't move to archive). "
             "Implied by --stdout.",
    )
    parser.add_argument(
        "-p", "--path", default=None,
        help="Path to a folder or file (.m4a/.webm). "
             "Folder: use as streams directory. "
             "File: find matching pair in same directory. "
             "Output goes to <dir>/combined_yt/, archive to <dir>/archive_yt/.",
    )
    parser.add_argument(
        "--offset", default="auto",
        help="M4a sample offset: 'auto' (default) detects via ffprobe packet "
             "times, 'xcorr' detects via cross-correlation (slower), "
             "or an integer (e.g. 0 or 1600) to force a specific skip.",
    )
    parser.add_argument(
        "--static-avg", action="store_true",
        help="Use static 0.5/0.5 averaging instead of dynamic VBR-based weights.",
    )
    parser.add_argument(
        "--bias", default="table",
        help="Bias mode: 'table' (default) uses frequency-band-optimized "
             "processing with biases 45/55/60 across 11 bands. "
             "Or an integer 0-100 for single-bias mode. "
             "50 = no bias. >50 = prefer opus. <50 = prefer m4a.",
    )
    parser.add_argument(
        "--simple-mix-mask", action="store_true",
        help="Use simple additive mixing for mask mixing (step 6) instead of "
             "the default max_fft ensemble.",
    )
    # Output format
    parser.add_argument(
        "--format", default="wav",
        help="Output format: wav (default), mp3, flac, opus, aac, ogg. "
             "Or any extension when used with --encoder.",
    )
    parser.add_argument(
        "--encoder", default=None,
        help="Custom encoder command (reads WAV from stdin). "
             "Use {output} for the output file path. "
             "Example: 'qaac --cvbr 256 - -o {output}'",
    )
    parser.add_argument(
        "--bitrate", default=None,
        help="Bitrate for lossy formats (e.g. 320k, 128k). "
             "Defaults: mp3=320k, opus=128k, aac=256k.",
    )
    parser.add_argument(
        "--bit-depth", type=int, default=None,
        help="Bit depth for lossless: 16, 24, 32. "
             "Defaults: wav=32 (float), flac=24.",
    )
    parser.add_argument(
        "--encoder-args", default=None,
        help="Extra arguments appended to the ffmpeg encoder command.",
    )
    # Presets
    parser.add_argument(
        "--preset", default=None,
        help="Load a saved format preset.",
    )
    parser.add_argument(
        "--save-preset", default=None, metavar="NAME",
        help="Save current format+encoder settings as a preset, then exit.",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List saved presets and exit.",
    )
    # Output naming
    parser.add_argument(
        "--output-template", default=None, metavar="TPL",
        help="Output filename template using yt-dlp info.json fields. "
             "Example: '%%(artist)s - %%(title)s'. "
             "Extension added automatically from --format.",
    )
    # Metadata
    parser.add_argument(
        "--embed-metadata", action="store_true",
        help="Embed metadata from .info.json into output files.",
    )
    parser.add_argument(
        "--embed-thumbnail", action="store_true",
        help="Embed thumbnail into output files. "
             "Requires --write-thumbnail in yt-dlp args.",
    )
    args, extra = parser.parse_known_args()

    # --- Early-exit commands ---
    if args.list_presets:
        _list_presets()
        return 0

    # --- Resolve output configuration (preset -> CLI overrides) ---
    format_name = "wav"
    bitrate = None
    bit_depth = None
    custom_encoder = None
    output_template = None
    embed_metadata = False
    embed_thumbnail = False
    extra_enc_args: list[str] | None = None

    if args.preset:
        presets = _load_presets()
        if args.preset not in presets:
            print(f"Error: preset '{args.preset}' not found.", file=sys.stderr)
            _list_presets()
            return 1
        pr = presets[args.preset]
        format_name = pr.get("format", format_name)
        bitrate = pr.get("bitrate", bitrate)
        bit_depth = pr.get("bit_depth", bit_depth)
        custom_encoder = pr.get("encoder", custom_encoder)
        output_template = pr.get("output_template", output_template)
        embed_metadata = pr.get("embed_metadata", embed_metadata)
        embed_thumbnail = pr.get("embed_thumbnail", embed_thumbnail)

    # CLI flags override preset
    if args.format != "wav":
        format_name = args.format
    elif args.preset is None:
        format_name = args.format  # still "wav"
    if args.bitrate is not None:
        bitrate = args.bitrate
    if args.bit_depth is not None:
        bit_depth = args.bit_depth
    if args.encoder is not None:
        custom_encoder = args.encoder
    if args.output_template is not None:
        output_template = args.output_template
    if args.embed_metadata:
        embed_metadata = True
    if args.embed_thumbnail:
        embed_thumbnail = True
    if args.encoder_args is not None:
        extra_enc_args = shlex.split(args.encoder_args)

    # Handle --save-preset
    if args.save_preset:
        config: dict = {"format": format_name}
        if bitrate:
            config["bitrate"] = bitrate
        if bit_depth:
            config["bit_depth"] = bit_depth
        if custom_encoder:
            config["encoder"] = custom_encoder
        if output_template:
            config["output_template"] = output_template
        if embed_metadata:
            config["embed_metadata"] = True
        if embed_thumbnail:
            config["embed_thumbnail"] = True
        _save_preset(args.save_preset, config)
        print(f"Preset '{args.save_preset}' saved.", file=sys.stderr)
        return 0

    # Validate format
    if custom_encoder is None and format_name not in FORMAT_DEFAULTS:
        print(f"Error: unknown format '{format_name}'. "
              f"Known: {', '.join(FORMAT_DEFAULTS)}. "
              f"Use --encoder for custom formats.", file=sys.stderr)
        return 1

    if args.stdout and format_name != "wav":
        print("Error: --stdout only supports wav format. "
              "Pipe to an encoder manually: ... | ffmpeg -f wav -i - ...",
              file=sys.stderr)
        return 1

    # Determine output extension
    if custom_encoder:
        out_ext = format_name  # user's --format IS the extension
    else:
        out_ext = FORMAT_DEFAULTS[format_name]["ext"]

    if args.chunk_size % 1024 != 0:
        print("Error: --chunk-size must be divisible by 1024.", file=sys.stderr)
        return 1

    if args.keep and args.dont_keep:
        print("Error: --keep and --dont-keep are mutually exclusive.",
              file=sys.stderr)
        return 1

    # Parse --offset: "auto", "xcorr", or an integer
    if args.offset in ("auto", "xcorr"):
        offset_mode = args.offset
        forced_offset = None
    else:
        try:
            forced_offset = int(args.offset)
            offset_mode = "forced"
        except ValueError:
            print(f"Error: --offset must be 'auto', 'xcorr', or an integer, "
                  f"got '{args.offset}'.", file=sys.stderr)
            return 1

    if args.bias == "table":
        parsed_bias: str | int = "table"
    else:
        try:
            parsed_bias = int(args.bias)
        except ValueError:
            print(f"Error: --bias must be 'table' or an integer 0-100, "
                  f"got '{args.bias}'.", file=sys.stderr)
            return 1
        if not 0 <= parsed_bias <= 100:
            print("Error: --bias integer must be between 0 and 100.",
                  file=sys.stderr)
            return 1
    args.bias = parsed_bias

    # Detect user-supplied -o in extra yt-dlp args
    user_output_dir: Path | None = None
    for i, a in enumerate(extra):
        if a == "-o" and i + 1 < len(extra):
            user_output_dir = Path(extra[i + 1]).parent
            break
        elif a.startswith("-o") and len(a) > 2:  # -oTEMPLATE (no space)
            user_output_dir = Path(a[2:]).parent
            break

    # Resolve directories based on --path
    has_batch = any(a in ("-a", "--batch-file") for a in extra)
    file_mode_pair: tuple[Path, Path] | None = None

    if args.path is not None:
        p = Path(args.path)
        if p.is_file():
            # File mode: find matching pair
            if args.url is not None or has_batch:
                print("Error: cannot combine --path <file> with a URL or -a.",
                      file=sys.stderr)
                return 1
            ext = p.suffix.lower()
            if ext == ".m4a":
                other = p.with_suffix(".webm")
            elif ext == ".webm":
                other = p.with_suffix(".m4a")
            else:
                print(f"Error: --path file must be .m4a or .webm, "
                      f"got '{ext}'.", file=sys.stderr)
                return 1
            if not other.exists():
                print(f"Error: matching file not found: {other}",
                      file=sys.stderr)
                return 1
            m4a = p if ext == ".m4a" else other
            webm = other if ext == ".m4a" else p
            file_mode_pair = (m4a, webm)
            base_dir = p.parent
        elif p.is_dir() or (args.url is not None or has_batch):
            # Folder mode (create if downloading)
            base_dir = p
        else:
            print(f"Error: --path does not exist: {p}", file=sys.stderr)
            return 1
        streams_dir = base_dir
        combined_dir = base_dir / "combined_yt"
        archive_dir = base_dir / "archive_yt"
    else:
        if user_output_dir is not None:
            base_dir = user_output_dir
            streams_dir = base_dir
            combined_dir = base_dir / "combined_yt"
            archive_dir = base_dir / "archive_yt"
        else:
            streams_dir = STREAMS_DIR
            combined_dir = COMBINED_DIR
            archive_dir = ARCHIVE_DIR

    # Download if URL provided or -a/--batch-file in extra args
    if args.url is not None or has_batch:
        inject = []
        if (embed_metadata or output_template) \
                and "--write-info-json" not in extra:
            inject.append("--write-info-json")
        if embed_thumbnail and "--write-thumbnail" not in extra:
            inject.append("--write-thumbnail")
        download_streams(args.url, extra, streams_dir,
                         inject_args=inject or None)

    # Find matching pairs
    if file_mode_pair is not None:
        pairs = [file_mode_pair]
    else:
        streams_dir.mkdir(parents=True, exist_ok=True)
        pairs = find_matches(streams_dir)

    if not pairs:
        if args.url is None and not has_batch:
            print(f"No matched m4a/webm pairs in {streams_dir} and no URL "
                  "given. Nothing to do.", file=sys.stderr)
            return 1
        else:
            print("No matched m4a/webm pairs found after download. "
                  "One of the format downloads may have failed.",
                  file=sys.stderr)
            return 1

    # Determine keep behavior: --dont-keep > --keep/--stdout > archive
    keep = args.keep or (args.stdout and not args.dont_keep)

    # Process each pair
    for m4a_path, webm_path in pairs:
        # Read metadata if needed
        info = None
        if output_template or embed_metadata or embed_thumbnail:
            info = _find_info_json(m4a_path)
            if info is None and output_template:
                print(f"[metadata] Warning: no .info.json found for "
                      f"{m4a_path.stem}, using source filename",
                      file=sys.stderr)

        # Resolve output path
        if args.stdout:
            output_path = None
        else:
            combined_dir.mkdir(parents=True, exist_ok=True)
            output_path = _resolve_output_path(
                m4a_path, combined_dir, out_ext,
                output_template=output_template, info=info)

        # Build encoder command (None for wav)
        enc_cmd = None
        if format_name != "wav" and not args.stdout:
            metadata_tags = None
            if embed_metadata and info:
                metadata_tags = _extract_metadata_tags(info)
            thumb = None
            if embed_thumbnail:
                video_id = _extract_video_id(m4a_path.stem)
                if video_id:
                    thumb = _find_thumbnail(m4a_path.parent, video_id)
                if thumb is None:
                    print("[metadata] Warning: no thumbnail found, "
                          "skipping embed.", file=sys.stderr)
            enc_cmd = _build_encoder_cmd(
                output_path, format_name=format_name,
                bitrate=bitrate, bit_depth=bit_depth,
                custom_encoder=custom_encoder,
                metadata=metadata_tags,
                thumbnail_path=thumb,
                extra_encoder_args=extra_enc_args)

        process_pair(
            m4a_path, webm_path,
            output_path=output_path,
            stdout=args.stdout,
            chunk_size=args.chunk_size,
            offset_mode=offset_mode,
            forced_offset=forced_offset,
            static_avg=args.static_avg,
            bias=args.bias,
            ensemble_mask=not args.simple_mix_mask,
            encoder_cmd=enc_cmd,
        )

        if args.dont_keep:
            for src in (m4a_path, webm_path):
                src.unlink()
                print(f"[cleanup] deleted {src.name}", file=sys.stderr)
        elif not keep:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for src in (m4a_path, webm_path):
                dest = archive_dir / src.name
                shutil.move(str(src), str(dest))
                print(f"[archive] {src.name} -> {dest}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
