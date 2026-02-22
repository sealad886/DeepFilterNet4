use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Direct Form II transposed IIR biquad filter — same algorithm as
/// scipy.signal.lfilter for second-order sections, but avoids Python
/// function-call and intermediate-allocation overhead.
fn biquad_filter_inner(audio: &[f32], b: &[f32; 3], a: &[f32; 3]) -> Vec<f32> {
    let n = audio.len();
    let mut out = Vec::with_capacity(n);
    let mut w1: f32 = 0.0;
    let mut w2: f32 = 0.0;

    for &x in audio.iter() {
        let y = b[0] * x + w1;
        w1 = b[1] * x - a[1] * y + w2;
        w2 = b[2] * x - a[2] * y;
        out.push(y);
    }
    out
}

/// SNR-based audio mixing with anti-clip normalisation.
///
/// Returns `(clean_out, noise_scaled, noisy)`.
fn mix_audio_inner(
    clean: &[f32],
    noise: &[f32],
    snr_db: f32,
    gain_db: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = clean.len();
    let gain = 10.0_f32.powf(gain_db / 20.0);

    // Apply gain to clean
    let mut clean_out: Vec<f32> = clean.iter().map(|&s| s * gain).collect();

    // Tile noise to match clean length
    let mut tiled_noise = Vec::with_capacity(n);
    if noise.is_empty() {
        tiled_noise.resize(n, 0.0);
    } else {
        let mut idx = 0;
        while tiled_noise.len() < n {
            tiled_noise.push(noise[idx % noise.len()]);
            idx += 1;
        }
    }

    // Compute powers
    let clean_power: f32 =
        clean_out.iter().map(|&s| s * s).sum::<f32>() / (n as f32).max(1.0) + 1e-10;
    let noise_power: f32 =
        tiled_noise.iter().map(|&s| s * s).sum::<f32>() / (n as f32).max(1.0) + 1e-10;
    let target_noise_power = clean_power / 10.0_f32.powf(snr_db / 10.0);
    let mix_factor = (target_noise_power / noise_power).sqrt();

    // Scale noise
    let mut noise_scaled: Vec<f32> = tiled_noise.iter().map(|&s| s * mix_factor).collect();

    // Mix
    let mut noisy: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        noisy.push(clean_out[i] + noise_scaled[i]);
    }

    // Anti-clip normalisation
    let mut max_val: f32 = 0.0;
    for i in 0..n {
        max_val = max_val.max(clean_out[i].abs()).max(noise_scaled[i].abs()).max(noisy[i].abs());
    }
    if max_val > 1.0 - 1e-10 {
        let scale = 1.0 / (max_val + 1e-10);
        for i in 0..n {
            clean_out[i] *= scale;
            noise_scaled[i] *= scale;
            noisy[i] *= scale;
        }
    }

    (clean_out, noise_scaled, noisy)
}

/// Combine multiple noise signals with per-source gain, tiling and random offset.
fn combine_noises_inner(
    noises: &[&[f32]],
    target_len: usize,
    gains_db: &[f32],
    offsets: &[usize],
) -> Vec<f32> {
    let mut combined = vec![0.0_f32; target_len];

    for (idx, &noise) in noises.iter().enumerate() {
        let gain = 10.0_f32.powf(gains_db[idx] / 20.0);

        if noise.is_empty() {
            continue;
        }

        // Tile noise to be at least target_len + offset long
        let offset = offsets[idx];
        let needed = target_len + offset;

        // Fill combined buffer with tiled + offset noise * gain
        for i in 0..target_len {
            let src_idx = (i + offset) % noise.len();
            // If noise was shorter than needed, modulo handles tiling implicitly
            combined[i] += noise[src_idx] * gain;
        }

        // If noise was longer than target_len, the modulo truncation above
        // naturally handles the offset + tile semantics equivalent to:
        //   tile → offset-slice → truncate → accumulate
        let _ = needed; // suppress unused warning
    }

    combined
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[pymodule]
fn libdfaugment(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    /// Apply a biquad (second-order IIR) filter to an audio buffer.
    ///
    /// Parameters
    /// ----------
    /// audio : np.ndarray[f32]
    ///     1-D audio samples.
    /// b : np.ndarray[f32]
    ///     Numerator coefficients (length 3).
    /// a : np.ndarray[f32]
    ///     Denominator coefficients (length 3).
    ///
    /// Returns
    /// -------
    /// np.ndarray[f32]
    ///     Filtered audio.
    #[pyfn(m)]
    #[pyo3(name = "biquad_filter")]
    fn py_biquad_filter<'py>(
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
        b: PyReadonlyArray1<'py, f32>,
        a: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let audio_slice = audio.as_slice()?;
        let b_slice = b.as_slice()?;
        let a_slice = a.as_slice()?;

        if b_slice.len() != 3 || a_slice.len() != 3 {
            return Err(PyValueError::new_err("b and a must each have length 3"));
        }

        let b_arr: [f32; 3] = [b_slice[0], b_slice[1], b_slice[2]];
        let a_arr: [f32; 3] = [a_slice[0], a_slice[1], a_slice[2]];

        let result = biquad_filter_inner(audio_slice, &b_arr, &a_arr);
        Ok(result.into_pyarray_bound(py))
    }

    /// Mix clean speech with noise at a target SNR.
    ///
    /// Parameters
    /// ----------
    /// clean : np.ndarray[f32]
    ///     Clean speech signal.
    /// noise : np.ndarray[f32]
    ///     Noise signal (tiled if shorter than *clean*).
    /// snr_db : float
    ///     Target signal-to-noise ratio in dB.
    /// gain_db : float
    ///     Gain applied to the clean signal in dB.
    ///
    /// Returns
    /// -------
    /// tuple[np.ndarray[f32], np.ndarray[f32], np.ndarray[f32]]
    ///     ``(clean_out, noise_scaled, noisy_mixture)``
    #[pyfn(m)]
    #[pyo3(name = "mix_audio")]
    fn py_mix_audio<'py>(
        py: Python<'py>,
        clean: PyReadonlyArray1<'py, f32>,
        noise: PyReadonlyArray1<'py, f32>,
        snr_db: f32,
        gain_db: f32,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let clean_slice = clean.as_slice()?;
        let noise_slice = noise.as_slice()?;

        let (c, ns, noisy) = mix_audio_inner(clean_slice, noise_slice, snr_db, gain_db);

        Ok((
            c.into_pyarray_bound(py),
            ns.into_pyarray_bound(py),
            noisy.into_pyarray_bound(py),
        ))
    }

    /// Combine multiple noise signals into a single buffer.
    ///
    /// Parameters
    /// ----------
    /// noises : list[np.ndarray[f32]]
    ///     List of 1-D noise arrays.
    /// target_len : int
    ///     Desired output length in samples.
    /// gains_db : list[float]
    ///     Per-source gain in dB.
    /// offsets : list[int]
    ///     Per-source sample offset (for random start position).
    ///
    /// Returns
    /// -------
    /// np.ndarray[f32]
    ///     Combined noise buffer of length *target_len*.
    #[pyfn(m)]
    #[pyo3(name = "combine_noises")]
    fn py_combine_noises<'py>(
        py: Python<'py>,
        noises: Vec<PyReadonlyArray1<'py, f32>>,
        target_len: usize,
        gains_db: Vec<f32>,
        offsets: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if noises.len() != gains_db.len() || noises.len() != offsets.len() {
            return Err(PyValueError::new_err(
                "noises, gains_db, and offsets must have the same length",
            ));
        }

        let slices: Vec<&[f32]> =
            noises.iter().map(|arr| arr.as_slice()).collect::<Result<_, _>>()?;

        let result = combine_noises_inner(&slices, target_len, &gains_db, &offsets);
        Ok(result.into_pyarray_bound(py))
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_passthrough() {
        // Identity filter: b=[1,0,0], a=[1,0,0]
        let audio = vec![1.0, 0.5, -0.3, 0.8, -1.0];
        let b = [1.0, 0.0, 0.0];
        let a = [1.0, 0.0, 0.0];
        let out = biquad_filter_inner(&audio, &b, &a);
        for (o, &expected) in out.iter().zip(audio.iter()) {
            assert!((o - expected).abs() < 1e-7, "expected {expected}, got {o}");
        }
    }

    #[test]
    fn test_biquad_dc_gain() {
        // DC gain of a filter = sum(b) / sum(a)
        let b = [0.5, 0.3, 0.2]; // sum = 1.0
        let a = [1.0, -0.1, 0.1]; // sum = 1.0
                                  // Constant input should converge to input * (sum_b / sum_a) = 1.0
        let audio = vec![1.0; 1000];
        let out = biquad_filter_inner(&audio, &b, &a);
        assert!(
            (out[999] - 1.0).abs() < 1e-3,
            "DC gain mismatch: {}",
            out[999]
        );
    }

    #[test]
    fn test_mix_audio_zero_snr() {
        let clean = vec![0.5; 100];
        let noise = vec![0.5; 100];
        let (c, ns, noisy) = mix_audio_inner(&clean, &noise, 0.0, 0.0);
        // At 0 dB SNR, clean and noise should have equal power
        let c_pow: f32 = c.iter().map(|&s| s * s).sum::<f32>() / c.len() as f32;
        let n_pow: f32 = ns.iter().map(|&s| s * s).sum::<f32>() / ns.len() as f32;
        let ratio_db = 10.0 * (c_pow / n_pow).log10();
        assert!(
            ratio_db.abs() < 1.0,
            "Expected ~0 dB SNR, got {ratio_db:.2} dB"
        );
        // Noisy should be sum
        for i in 0..noisy.len() {
            assert!((noisy[i] - (c[i] + ns[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mix_audio_anticlip() {
        let clean = vec![0.9; 100];
        let noise = vec![0.9; 100];
        let (_c, _ns, noisy) = mix_audio_inner(&clean, &noise, 0.0, 6.0);
        let max_abs = noisy.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0, "Anti-clip failed: max={max_abs}");
    }

    #[test]
    fn test_mix_audio_noise_tiling() {
        // Short noise should be tiled to match clean length
        let clean = vec![0.5; 200];
        let noise = vec![0.3, -0.3];
        let (_c, ns, _noisy) = mix_audio_inner(&clean, &noise, 10.0, 0.0);
        assert_eq!(ns.len(), 200);
    }

    #[test]
    fn test_combine_noises_basic() {
        let n1 = vec![1.0; 100];
        let n2 = vec![0.5; 100];
        let gains = vec![0.0, 0.0]; // unity gain
        let offsets = vec![0, 0];
        let combined = combine_noises_inner(&[&n1, &n2], 100, &gains, &offsets);
        for &v in combined.iter() {
            assert!((v - 1.5).abs() < 1e-6, "expected 1.5, got {v}");
        }
    }

    #[test]
    fn test_combine_noises_with_offset() {
        // Noise [1,2,3,4] with offset 2 and target_len 4 → [3,4,1,2]
        let noise = vec![1.0, 2.0, 3.0, 4.0];
        let gains = vec![0.0];
        let offsets = vec![2];
        let combined = combine_noises_inner(&[&noise], 4, &gains, &offsets);
        assert!((combined[0] - 3.0).abs() < 1e-6);
        assert!((combined[1] - 4.0).abs() < 1e-6);
        assert!((combined[2] - 1.0).abs() < 1e-6);
        assert!((combined[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_combine_noises_tiling() {
        // Short noise [1,2] tiled to target_len 5 → [1,2,1,2,1]
        let noise = vec![1.0, 2.0];
        let gains = vec![0.0];
        let offsets = vec![0];
        let combined = combine_noises_inner(&[&noise], 5, &gains, &offsets);
        assert_eq!(combined.len(), 5);
        assert!((combined[0] - 1.0).abs() < 1e-6);
        assert!((combined[1] - 2.0).abs() < 1e-6);
        assert!((combined[2] - 1.0).abs() < 1e-6);
        assert!((combined[3] - 2.0).abs() < 1e-6);
        assert!((combined[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_combine_noises_gain() {
        // +6 dB ≈ factor of 2
        let noise = vec![0.5; 100];
        let gains = vec![6.0206]; // exactly 20*log10(2)
        let offsets = vec![0];
        let combined = combine_noises_inner(&[&noise], 100, &gains, &offsets);
        for &v in combined.iter() {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0 (0.5 * 2.0), got {v}");
        }
    }

    #[test]
    fn test_combine_noises_empty() {
        let combined = combine_noises_inner(&[], 100, &[], &[]);
        assert_eq!(combined.len(), 100);
        for &v in combined.iter() {
            assert!((v).abs() < 1e-10);
        }
    }
}
