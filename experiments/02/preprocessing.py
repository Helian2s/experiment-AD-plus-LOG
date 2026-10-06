"""Fixed preprocessing and paired diagnostics for the third experiment.

The brightness control matches mean BGR code-value intensity, not perceptual
luminance. It does not isolate all spatial, contrast, or saturation effects of
LoG modulation; detection metrics, not map correlation, test usefulness.
"""

from numbers import Integral

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace


CONDITIONS = ("clean", "gaussian_noise", "blur", "low_light", "haze", "multifactor")
METHODS = ("baseline", "ad_only", "ad_log", "ad_brightness_control")
AD_KAPPA = 30
AD_NITER = 5
AD_GAMMA = 0.15
LOG_SIGMA = 1.5
LOG_BETA = 0.3
LOW_LIGHT_GAMMA = 2.0
GAUSSIAN_NOISE_SIGMA = 25.0
BLUR_KERNEL = (5, 5)
HAZE_BETA = 1.0
HAZE_A = 0.9


def _validate_image(image, allow_gray=False):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Image must be a uint8 NumPy array")
    is_gray = allow_gray and image.ndim == 2
    is_bgr = image.ndim == 3 and image.shape[2] == 3
    if not (is_gray or is_bgr) or image.size == 0:
        raise ValueError("Image must be nonempty BGR (or grayscale for map operations)")


def _positive_finite(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def apply_gamma(image, gamma=LOW_LIGHT_GAMMA):
    """Apply x**gamma to normalized code values; gamma > 1 darkens them."""
    _validate_image(image)
    _positive_finite(gamma, "gamma")
    table = ((np.arange(256, dtype=np.float64) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def add_gaussian_noise(image, rng, sigma=GAUSSIAN_NOISE_SIGMA):
    _validate_image(image)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_haze(image, beta=HAZE_BETA, A=HAZE_A):
    """Retain the historical synthetic left-to-right depth ramp, not real fog."""
    _validate_image(image)
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(A) or not 0 <= A <= 1:
        raise ValueError("Haze beta must be nonnegative and A must be in [0, 1]")
    depth = np.linspace(0.1, 1.0, image.shape[1])[None, :, None]
    transmission = np.exp(-beta * depth)
    hazy = image.astype(np.float32) / 255.0 * transmission + A * (1 - transmission)
    return np.clip(hazy * 255, 0, 255).astype(np.uint8)


def degrade_image(image, condition, rng):
    """Generate one shared degraded image for every method in a paired comparison."""
    _validate_image(image)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    if condition == "clean":
        return image.copy()
    if condition == "gaussian_noise":
        return add_gaussian_noise(image, rng)
    if condition == "blur":
        return cv2.GaussianBlur(image, BLUR_KERNEL, 0)
    if condition == "low_light":
        return apply_gamma(image)
    if condition == "haze":
        return add_haze(image)
    if condition == "multifactor":
        return add_haze(apply_gamma(add_gaussian_noise(image, rng)))
    raise ValueError(f"Unknown degradation condition: {condition}")


def anisotropic_diffusion(image, kappa=AD_KAPPA, niter=AD_NITER, gamma=AD_GAMMA):
    """Four-neighbor Perona-Malik diffusion with explicit zero-flux boundaries."""
    _validate_image(image)
    _positive_finite(kappa, "kappa")
    if isinstance(niter, bool) or not isinstance(niter, Integral) or niter < 0:
        raise ValueError("niter must be a nonnegative integer")
    if not np.isfinite(gamma) or not 0 < gamma <= 0.25:
        raise ValueError("gamma must satisfy 0 < gamma <= 0.25")
    image_float = image.astype(np.float64)
    for _ in range(niter):
        vertical = image_float[1:, :, :] - image_float[:-1, :, :]
        horizontal = image_float[:, 1:, :] - image_float[:, :-1, :]
        vertical *= np.exp(-(vertical / kappa) ** 2)
        horizontal *= np.exp(-(horizontal / kappa) ** 2)
        # Each interior edge contributes opposite fluxes to its two neighbors;
        # no exterior edge or connection between opposite borders exists.
        update = np.zeros_like(image_float)
        update[:-1, :, :] += vertical
        update[1:, :, :] -= vertical
        update[:, :-1, :] += horizontal
        update[:, 1:, :] -= horizontal
        image_float += gamma * update
    return np.clip(image_float, 0, 255).astype(np.uint8)


def _gray(image):
    _validate_image(image, allow_gray=True)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return image.astype(np.float64)


def compute_log_map(image, sigma=LOG_SIGMA):
    """Absolute LoG, min-max normalized to [0, 1]; constant maps become zero."""
    _positive_finite(sigma, "sigma")
    values = np.abs(gaussian_laplace(_gray(image), sigma=sigma, mode="reflect"))
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum <= 1e-10:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def compute_gradient_map(image):
    gray = _gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return gx ** 2 + gy ** 2


def apply_log_modulation(img_ad, log_map, beta=LOG_BETA):
    _validate_image(img_ad)
    if log_map.shape != img_ad.shape[:2] or not np.all(np.isfinite(log_map)):
        raise ValueError("LoG map must be finite and match the image dimensions")
    if np.any(log_map < 0) or np.any(log_map > 1):
        raise ValueError("LoG map must be normalized to [0, 1]")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    modulated = img_ad.astype(np.float64) * (1 + beta * log_map[:, :, None])
    return np.clip(modulated, 0, 255).astype(np.uint8)


def compute_correlations(img_ad, log_map):
    """Return descriptive Pearson correlation, never NaN for degenerate maps."""
    if log_map.shape != img_ad.shape[:2]:
        raise ValueError("LoG map dimensions do not match the image")
    gradient = compute_gradient_map(img_ad).ravel()
    values = np.asarray(log_map, dtype=np.float64).ravel()
    status = "ok"
    rho = None
    if values.size < 2:
        status = "insufficient_pixels"
    elif not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradient)):
        status = "non_finite_map"
    elif np.ptp(values) <= 1e-10:
        status = "constant_log_map"
    elif np.ptp(gradient) <= 1e-10:
        status = "constant_gradient_map"
    else:
        candidate = float(np.corrcoef(values, gradient)[0, 1])
        if np.isfinite(candidate):
            rho = float(np.clip(candidate, -1, 1))
        else:
            status = "non_finite_correlation"
    return {"rho_log_g": rho, "correlation_status": status}


def brightness_matched_control(img_ad, target_image, max_gain=1 + LOG_BETA):
    """Match the target's mean uint8 intensity with one clipped scalar gain.

    Integer quantization can prevent an exact match. Bisection finds the closest
    of the adjacent attainable means and records the residual, rather than
    changing individual pixels to force equality. This controls only the mean.
    """
    _validate_image(img_ad)
    _validate_image(target_image)
    if img_ad.shape != target_image.shape:
        raise ValueError("Brightness target dimensions must match the image")
    if not np.isfinite(max_gain) or max_gain < 1:
        raise ValueError("max_gain must be finite and at least 1")
    histogram = np.bincount(img_ad.ravel(), minlength=256)
    levels = np.arange(256, dtype=np.float64)
    target = float(np.mean(target_image))

    def mean_at(gain):
        table = np.clip(levels * gain, 0, 255).astype(np.uint8)
        return float(np.dot(histogram, table) / img_ad.size)

    lower, upper = 1.0, float(max_gain)
    low_mean, high_mean = mean_at(lower), mean_at(upper)
    if target < low_mean - 1e-10 or target > high_mean + 1e-10:
        raise ValueError("Target brightness is outside the attainable gain interval")
    best_gain, achieved = lower, low_mean
    if abs(high_mean - target) < abs(achieved - target):
        best_gain, achieved = upper, high_mean
    for _ in range(64):
        middle = (lower + upper) / 2
        if middle == lower or middle == upper or achieved == target:
            break
        mean = mean_at(middle)
        if abs(mean - target) < abs(achieved - target):
            best_gain, achieved = middle, mean
        if mean < target:
            lower = middle
        else:
            upper = middle
    table = np.clip(levels * best_gain, 0, 255).astype(np.uint8)
    matched = cv2.LUT(img_ad, table)
    diagnostics = {
        "gain": float(best_gain),
        "target_mean_intensity": target,
        "achieved_mean_intensity": float(np.mean(matched)),
        "mean_intensity_error": float(np.mean(matched)) - target,
        "absolute_mean_intensity_error": abs(float(np.mean(matched)) - target),
        "status": "exact" if achieved == target else "nearest_quantized_mean",
    }
    return matched, diagnostics


def prepare_methods(degraded):
    """Reuse one AD/LoG map; saturation is the fraction of channel values at 255."""
    _validate_image(degraded)
    img_ad = anisotropic_diffusion(degraded)
    log_map = compute_log_map(img_ad)
    img_adlog = apply_log_modulation(img_ad, log_map)
    brightness_control, brightness_diagnostics = brightness_matched_control(img_ad, img_adlog)
    images = {
        "baseline": degraded.copy(),
        "ad_only": img_ad,
        "ad_log": img_adlog,
        "ad_brightness_control": brightness_control,
    }
    diagnostics = compute_correlations(img_ad, log_map)
    diagnostics["brightness_control"] = brightness_diagnostics
    diagnostics["method_intensity"] = {
        name: {
            "mean_intensity": float(np.mean(image)),
            "saturated_fraction": float(np.mean(image == 255)),
        }
        for name, image in images.items()
    }
    return images, diagnostics
