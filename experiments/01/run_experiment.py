"""
Експеримент: порівняння AD, AD+LoG і baseline для виявлення об'єктів
Датасет: COCO val2017 (кількість зображень задається через --images)
Детектор: YOLO11n (заморожений)

Reconstructed legacy code, not an original execution snapshot. See README.md.
Known scientific limitations are intentionally retained.
"""

import os
import argparse
from datetime import datetime, timezone
import re
import shutil
import uuid
import cv2
import numpy as np
import json
import torch
from scipy import stats
from scipy.ndimage import gaussian_laplace
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Налаштування ─────────────────────────────────────────────────────────────
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
COCO_IMG_DIR   = os.path.join(PROJECT_ROOT, 'shared', 'coco', 'images', 'val2017')
COCO_ANN_FILE  = os.path.join(PROJECT_ROOT, 'shared', 'coco', 'annotations', 'instances_val2017.json')
MODEL_PATH     = os.path.join(PROJECT_ROOT, 'shared', 'yolo11n.pt')
RESULTS_DIR    = os.path.join(EXPERIMENT_DIR, 'runs')
N_IMAGES       = 5000       # кількість зображень для експерименту
RANDOM_SEED    = 42

# Параметри анізотропної дифузії
AD_KAPPA       = 30        # контрастний поріг
AD_NITER       = 5         # кількість ітерацій
AD_GAMMA       = 0.15      # часовий крок

# Параметри LoG
LOG_SIGMA      = 1.5       # масштаб Гаусіана для LoG
LOG_BETA       = 0.3       # коефіцієнт модуляції

# Умови деградації
DEGRADATIONS = {
    'clean':          lambda img: img.copy(),
    'gaussian_noise': lambda img: add_gaussian_noise(img, sigma=25),
    'blur':           lambda img: cv2.GaussianBlur(img, (5,5), 0),
    'low_light':      lambda img: apply_gamma(img, gamma=0.5),
    'haze':           lambda img: add_haze(img, beta=1.0, A=0.9),
    'multifactor':    lambda img: add_haze(
                          apply_gamma(
                              add_gaussian_noise(img, sigma=25), 0.5),
                          beta=1.0, A=0.9),
}

# ── Функції деградації ────────────────────────────────────────────────────────

def add_gaussian_noise(img, sigma=25):
    """Додати гаусів шум."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_gamma(img, gamma=0.5):
    """Гамма–корекція для імітації зниженої освітленості."""
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)

def add_haze(img, beta=1.0, A=0.9):
    """Модель туману Кошмідера: I = J*t + A*(1–t), t = exp(–beta*d)."""
    img_f = img.astype(np.float32) / 255.0
    # Проста апроксимація: рівномірна карта глибини
    h, w  = img_f.shape[:2]
    d     = np.linspace(0.1, 1.0, w)
    d     = np.tile(d, (h, 1))
    if len(img_f.shape) == 3:
        d = d[:, :, np.newaxis]
    t     = np.exp(-beta * d)
    hazy  = img_f * t + A * (1 - t)
    return np.clip(hazy * 255, 0, 255).astype(np.uint8)

# ── Анізотропна дифузія (Перона–Малік) ───────────────────────────────────────

def anisotropic_diffusion(img, kappa=30, niter=5, gamma=0.15):
    """
    Анізотропна дифузія Перони–Маліка.
    img   : BGR uint8
    kappa : контрастний поріг
    niter : кількість ітерацій
    gamma : часовий крок (0 < gamma <= 0.25 для стійкості)
    """
    # Конвертувати у float і обробляти покомпонентно
    img_f = img.astype(np.float64)
    result = np.zeros_like(img_f)

    for c in range(img_f.shape[2]):
        u = img_f[:, :, c].copy()
        for _ in range(niter):
            # Градієнти у 4 напрямках
            dN = np.roll(u,  1, axis=0) - u
            dS = np.roll(u, -1, axis=0) - u
            dE = np.roll(u, -1, axis=1) - u
            dW = np.roll(u,  1, axis=1) - u
            # Функція провідності (Перона–Малік варіант 1)
            cN = np.exp(-(dN/kappa)**2)
            cS = np.exp(-(dS/kappa)**2)
            cE = np.exp(-(dE/kappa)**2)
            cW = np.exp(-(dW/kappa)**2)
            # Оновлення
            u += gamma * (cN*dN + cS*dS + cE*dE + cW*dW)
        result[:, :, c] = u

    return np.clip(result, 0, 255).astype(np.uint8)

# ── Лапласіан Гаусіана (LoG) ──────────────────────────────────────────────────

def compute_log_map(img, sigma=1.5):
    """
    Обчислити карту Лапласіана Гаусіана.
    Повертає абсолютне значення LoG, нормоване до [0,1].
    """
    # Перетворити у відтінки сірого
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = img.astype(np.float64)

    # gaussian_laplace з scipy: ∇²(G_σ * u)
    log_map = gaussian_laplace(gray, sigma=sigma)

    # Взяти абсолютне значення (нас цікавить магнітуда)
    log_map = np.abs(log_map)

    # Нормувати до [0, 1]
    log_min, log_max = log_map.min(), log_map.max()
    if log_max - log_min > 1e-10:
        log_map = (log_map - log_min) / (log_max - log_min)

    return log_map

def compute_gradient_map(img):
    """Карта квадрату градієнта G = ||∇u||²."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = img.astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx**2 + gy**2

def apply_log_modulation(img_ad, log_map, beta=0.3):
    """
    Модуляція: û = u_AD * (1 + β * LoG_map)
    log_map: нормована карта LoG [0,1]
    """
    img_f = img_ad.astype(np.float64)
    # Розширити log_map на 3 канали
    if len(img_f.shape) == 3:
        mod = log_map[:, :, np.newaxis]
    else:
        mod = log_map
    result = img_f * (1.0 + beta * mod)
    return np.clip(result, 0, 255).astype(np.uint8)

# ── YOLO детекція ─────────────────────────────────────────────────────────────

def detect_image(model, img):
    """
    Запустити YOLO11n на зображенні.
    Повертає список словників у форматі COCO.
    """
    results = model(img, verbose=False)
    detections = []
    for r in results:
        boxes  = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            # COCO формат: [x, y, width, height]
            detections.append({
                'bbox':        [x1, y1, x2 - x1, y2 - y1],
                'score':       conf,
                'category_id': cls + 1,  # YOLO: 0–indexed, COCO: 1–indexed
            })
    return detections

# ── Кореляційна діагностика ───────────────────────────────────────────────────

def compute_correlations(img_ad, log_map):
    """
    Обчислити кореляції між картами:
    – ρ(LoG, G)
    Повертає словник з кореляціями.
    """
    G = compute_gradient_map(img_ad).flatten()
    L = log_map.flatten()

    rho_log_g, _ = stats.pearsonr(L, G)

    return {'rho_log_g': rho_log_g}

# ── Головна функція ───────────────────────────────────────────────────────────

def configure_run(argv=None):
    """Select execution settings without changing the legacy algorithms."""
    global N_IMAGES, RESULTS_DIR
    parser = argparse.ArgumentParser(description="Run the reconstructed legacy protocol")
    parser.add_argument('--images', type=int, default=N_IMAGES)
    parser.add_argument('--run-name', help="New directory name under this version's runs/")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.images <= 5000:
        parser.error('--images must be between 1 and 5000')
    name = arguments.run_name
    if name is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        name = f"images_{arguments.images}_{stamp}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        parser.error('--run-name must contain only ASCII letters, digits, underscores or hyphens')
    directory = os.path.join(EXPERIMENT_DIR, 'runs', name)
    if os.path.lexists(directory):
        raise FileExistsError(f"Refusing to reuse an existing run directory: {directory}")
    N_IMAGES = arguments.images
    RESULTS_DIR = directory


def main():
    out_path = os.path.join(RESULTS_DIR, 'results.json')
    if os.path.lexists(out_path):
        raise FileExistsError(
            f"Refusing to overwrite {out_path}. Select a new --run-name."
        )
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Local model weights not found: {MODEL_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=False)
    configuration = {
        'experiment': '01',
        'provenance': 'New execution of reconstructed legacy code; not a historical rerun snapshot',
        'params': {key: globals()[key] for key in (
            'AD_KAPPA', 'AD_NITER', 'AD_GAMMA', 'LOG_SIGMA', 'LOG_BETA',
            'N_IMAGES', 'RANDOM_SEED')},
    }
    with open(os.path.join(RESULTS_DIR, 'run_config.json'), 'x', encoding='utf-8') as stream:
        json.dump(configuration, stream, indent=2)
    snapshot = os.path.join(RESULTS_DIR, 'source_snapshot')
    os.mkdir(snapshot)
    shutil.copyfile(__file__, os.path.join(snapshot, 'run_experiment.py'))

    print("=" * 60)
    print("Експеримент: AD + LoG для виявлення об'єктів")
    print("=" * 60)

    # Завантажити список зображень
    np.random.seed(RANDOM_SEED)
    all_images = sorted(os.listdir(COCO_IMG_DIR))
    selected   = np.random.choice(all_images, N_IMAGES, replace=False)
    img_ids    = [int(os.path.splitext(f)[0]) for f in selected]

    print(f"Обрано {N_IMAGES} зображень з {len(all_images)} доступних.")

    # Завантажити модель YOLO (заморожена — ваги не змінюються)
    print("Завантаження YOLO11n...")
    model = YOLO(MODEL_PATH)
    model.fuse()  # оптимізація для inference

    # Завантажити анотації COCO
    coco_gt = COCO(COCO_ANN_FILE)

    # Словники для збереження результатів
    all_results   = {}   # {умова: {метод: [детекції]}}
    all_corr      = {}   # {умова: [rho_log_g per image]}

    for deg_name, deg_func in DEGRADATIONS.items():
        print(f"\nУмова деградації: {deg_name}")
        all_results[deg_name] = {
            'baseline': [],
            'ad_only':  [],
            'ad_log':   [],
        }
        corr_list = []

        for fname in tqdm(selected, desc=deg_name):
            img_path = os.path.join(COCO_IMG_DIR, fname)
            img_id   = int(os.path.splitext(fname)[0])

            # Завантажити і деградувати зображення
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                continue
            img_deg  = deg_func(img_orig)

            # ── Метод 1: Baseline (без обробки) ──────────────────────────
            dets = detect_image(model, img_deg)
            for d in dets:
                d['image_id'] = img_id
            all_results[deg_name]['baseline'].extend(dets)

            # ── Метод 2: AD only ──────────────────────────────────────────
            img_ad   = anisotropic_diffusion(
                img_deg, kappa=AD_KAPPA,
                niter=AD_NITER, gamma=AD_GAMMA)
            dets_ad  = detect_image(model, img_ad)
            for d in dets_ad:
                d['image_id'] = img_id
            all_results[deg_name]['ad_only'].extend(dets_ad)

            # ── Метод 3: AD + LoG ─────────────────────────────────────────
            log_map  = compute_log_map(img_ad, sigma=LOG_SIGMA)
            img_adlog = apply_log_modulation(img_ad, log_map, beta=LOG_BETA)
            dets_adlog = detect_image(model, img_adlog)
            for d in dets_adlog:
                d['image_id'] = img_id
            all_results[deg_name]['ad_log'].extend(dets_adlog)

            # ── Кореляційна діагностика ───────────────────────────────────
            corr = compute_correlations(img_ad, log_map)
            corr_list.append(corr['rho_log_g'])

        all_corr[deg_name] = corr_list

    # ── Обчислення mAP через pycocotools ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Результати (mAP@[0.5:0.95])")
    print("=" * 60)

    map_results = {}

    for deg_name in DEGRADATIONS:
        map_results[deg_name] = {}
        for method in ['baseline', 'ad_only', 'ad_log']:
            dets = all_results[deg_name][method]
            if len(dets) == 0:
                map_results[deg_name][method] = 0.0
                continue
            try:
                coco_dt  = coco_gt.loadRes(dets)
                evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
                evaluator.params.imgIds = img_ids
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                map_val  = float(evaluator.stats[0])  # mAP@[0.5:0.95]
                map50    = float(evaluator.stats[1])  # mAP@0.5
            except Exception as e:
                print(f"  Помилка оцінювання {deg_name}/{method}: {e}")
                map_val, map50 = 0.0, 0.0
            map_results[deg_name][method] = {
                'mAP_50_95': map_val,
                'mAP_50':    map50,
            }

    # ── Вивести таблицю результатів ───────────────────────────────────────────
    print("\n{'='*60}")
    print(f"{'Умова':<18} {'Baseline':>10} {'AD only':>10} {'AD+LoG':>10} {'Δ(AD vs BL)':>12} {'Δ(LoG vs AD)':>13}")
    print("–" * 75)

    for deg_name in DEGRADATIONS:
        bl  = map_results[deg_name]['baseline']['mAP_50_95']
        ad  = map_results[deg_name]['ad_only']['mAP_50_95']
        adl = map_results[deg_name]['ad_log']['mAP_50_95']
        print(f"{deg_name:<18} {bl:>10.4f} {ad:>10.4f} {adl:>10.4f} "
              f"{ad-bl:>+12.4f} {adl-ad:>+13.4f}")

    # ── Вивести кореляційну діагностику ──────────────────────────────────────
    print("\n{'='*60}")
    print("Кореляційна діагностика ρ(LoG, G)")
    print("–" * 40)
    print(f"{'Умова':<18} {'mean ρ(LoG,G)':>14} {'std':>8}")
    print("–" * 40)

    corr_summary = {}
    for deg_name in DEGRADATIONS:
        vals = all_corr[deg_name]
        mean_rho = np.mean(vals)
        std_rho  = np.std(vals)
        corr_summary[deg_name] = {
            'mean': mean_rho, 'std': std_rho
        }
        print(f"{deg_name:<18} {mean_rho:>14.4f} {std_rho:>8.4f}")

    # ── Зберегти результати у JSON ────────────────────────────────────────────
    output = {
        'map_results':  map_results,
        'corr_summary': corr_summary,
        'params': {
            'AD_KAPPA':   AD_KAPPA,
            'AD_NITER':   AD_NITER,
            'AD_GAMMA':   AD_GAMMA,
            'LOG_SIGMA':  LOG_SIGMA,
            'LOG_BETA':   LOG_BETA,
            'N_IMAGES':   N_IMAGES,
            'RANDOM_SEED': RANDOM_SEED,
        }
    }

    with open(out_path, 'x', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nРезультати збережено у: {out_path}")
    print("Експеримент завершено.")

if __name__ == '__main__':
    configure_run()
    main()
