# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:50:38 2025

@author: Rehman
"""

# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation of Handcrafted Features for Underwater Image Quality Assessment
Updated on: April 2025
Author: Rehman
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr, kendalltau

# -------------------------- Feature Extractors ------------------------------

def extract_color_features(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean_lab = np.mean(lab, axis=(0, 1))
    std_lab = np.std(lab, axis=(0, 1))
    return np.concatenate([mean_lab, std_lab])

def extract_texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.array([np.var(lap), np.mean(lap), np.std(lap)])

def extract_structure_features(img, patch_size=32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    patches = [gray[y:y+patch_size, x:x+patch_size]
               for y in range(0, h - patch_size, patch_size)
               for x in range(0, w - patch_size, patch_size)]
    ssim_values = []
    for i in range(len(patches)-1):
        a, b = patches[i].flatten(), patches[i+1].flatten()
        if len(a) == len(b):
            ssim_values.append(np.corrcoef(a, b)[0, 1])
    return np.nan_to_num([np.mean(ssim_values), np.std(ssim_values)])

def extract_frequency_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray)/255.0)
    dct_high = dct[20:, 20:]
    return np.array([np.mean(dct_high), np.std(dct_high)])

def extract_hdaf_features(img):
    # Haze, blur, and color cast based simple descriptors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    saturation = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1])
    return np.array([blur, brightness, saturation])

def extract_fpff_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropy = -np.sum((p := np.histogram(gray, bins=256)[0] / gray.size) * np.log2(p + 1e-8))
    contrast = gray.std()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    return np.array([entropy, contrast, edge_density])

def extract_colorfulness(img):
    rg = np.abs(img[:, :, 0] - img[:, :, 1])
    yb = np.abs(0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2])
    return np.array([np.mean(rg), np.mean(yb), np.std(rg), np.std(yb)])

def extract_spatial_info(img):
    return np.array([np.var(img[:, :, 0]), np.var(img[:, :, 1]), np.var(img[:, :, 2])])

def extract_turbidity(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.array([np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])])

def extract_chromatic_aberration(img):
    b, g, r = cv2.split(img)
    return np.array([np.std(r - g), np.std(g - b), np.std(b - r)])

#################################
import cv2
import numpy as np

def resize_half(img):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), sigmaX=1.2)

# Color Features (LAB) at multiple scales
def extract_color_features_multiscale(img):
    feats = []
    for scale_img in [img, resize_half(img), gaussian_blur(img)]:
        lab = cv2.cvtColor(scale_img, cv2.COLOR_BGR2LAB)
        mean_lab = np.mean(lab, axis=(0, 1))
        std_lab = np.std(lab, axis=(0, 1))
        feats.append(np.concatenate([mean_lab, std_lab]))
    return np.concatenate(feats)

# Texture Features (Laplacian)
def extract_texture_features_multiscale(img):
    feats = []
    for scale_img in [img, resize_half(img), gaussian_blur(img)]:
        gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        feats.append(np.array([np.var(lap), np.mean(lap), np.std(lap)]))
    return np.concatenate(feats)

# Structure Features via patch similarity
def extract_structure_features_multiscale(img, patch_size=32):
    feats = []
    for scale_img in [img, resize_half(img), gaussian_blur(img)]:
        gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        patches = [gray[y:y+patch_size, x:x+patch_size]
                   for y in range(0, h - patch_size, patch_size)
                   for x in range(0, w - patch_size, patch_size)]
        ssim_vals = []
        for i in range(len(patches)-1):
            a = patches[i].flatten()
            b = patches[i+1].flatten()
            if len(a) == len(b):
                corr = np.corrcoef(a, b)[0, 1]
                ssim_vals.append(corr)
        ssim_vals = np.nan_to_num(ssim_vals)
        feats.append(np.array([np.mean(ssim_vals), np.std(ssim_vals)]))
    return np.concatenate(feats)

# Frequency Features (DCT)
def extract_frequency_features_multiscale(img):
    feats = []
    for scale_img in [img, resize_half(img), gaussian_blur(img)]:
        gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray) / 255.0)
        dct_high = dct[20:, 20:]
        feats.append(np.array([np.mean(dct_high), np.std(dct_high)]))
    return np.concatenate(feats)

# Replace this in your existing code
def extract_features_multiscale(img):
    color_feat = extract_color_features_multiscale(img)
    texture_feat = extract_texture_features_multiscale(img)
    structure_feat = extract_structure_features_multiscale(img)
    frequency_feat = extract_frequency_features_multiscale(img)
    return {
        'Color': color_feat,
        'Texture': texture_feat,
        'Structure': structure_feat,
        'Frequency': frequency_feat
    }



##############################


#########################
def extract_hdaf_features_multiscale(img):
    feats = []
    for scale_img in [img, resize_half(img), gaussian_blur(img)]:
        gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        saturation = np.std(cv2.cvtColor(scale_img, cv2.COLOR_BGR2HSV)[:, :, 1])
        feats.append(np.array([blur, brightness, saturation]))
    return np.concatenate(feats)

def extract_fpff_features_multiscale(img):
    feats = []
    for scale_img in [img, resize_half(img), gaussian_blur(img)]:
        gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
        entropy = -np.sum((p := np.histogram(gray, bins=256)[0] / gray.size) * np.log2(p + 1e-8))
        contrast = gray.std()
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        feats.append(np.array([entropy, contrast, edge_density]))
    return np.concatenate(feats)




############################
def light_attenuation_profile(img):
    """Estimate vertical attenuation across image height."""
    h, _, _ = img.shape
    thirds = [img[:h//3], img[h//3:2*h//3], img[2*h//3:]]
    mean_vals = [np.mean(np.mean(t, axis=0), axis=0) for t in thirds]
    attenuation = np.diff(np.array(mean_vals), axis=0)
    return attenuation.flatten()  # 2 rows of RGB differences
def specular_suppression_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 240)
    return np.array([bright_pixels / gray.size])
def edge_energy_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_energy = np.sum(edges)
    total_energy = np.sum(gray)
    return np.array([edge_energy / (total_energy + 1e-5)])
from scipy.spatial import distance

def channel_divergence(img):
    r_hist, _ = np.histogram(img[:, :, 2], bins=256, range=(0, 256), density=True)
    g_hist, _ = np.histogram(img[:, :, 1], bins=256, range=(0, 256), density=True)
    b_hist, _ = np.histogram(img[:, :, 0], bins=256, range=(0, 256), density=True)
    r_g = distance.jensenshannon(r_hist, g_hist)
    r_b = distance.jensenshannon(r_hist, b_hist)
    g_b = distance.jensenshannon(g_hist, b_hist)
    return np.array([r_g, r_b, g_b])
def non_uniform_illumination(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    illum_diff = gray.astype(np.float32) - blur.astype(np.float32)
    return np.array([np.std(illum_diff)])
def extract_pswod_features(img):
    wavelengths = [450, 500, 550, 600, 650, 700]
    features = []
    for wl in wavelengths:
        filtered = simulate_spectral_band(img, wl)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        # Statistical profile per wavelength
        mean_intensity = np.mean(gray)
        contrast = np.std(gray)
        entropy = -np.sum((p := np.histogram(gray, bins=256)[0] / gray.size) * np.log2(p + 1e-8))
        features.extend([mean_intensity, contrast, entropy])
    return np.array(features)
def simulate_spectral_band(img, wavelength_nm):
    """Simulate attenuation of a specific wavelength based on physics."""
    # Constants for absorption in water (approx. values)
    k = {
        450: 0.03,  # blue
        500: 0.07,  # green
        550: 0.15,  # yellow-green
        600: 0.4,   # orange
        650: 0.8,   # red
        700: 1.2    # deep red
    }
    factor = np.exp(-k.get(wavelength_nm, 0.5))  # simulate attenuation
    img_filtered = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    return img_filtered

def estimate_backscatter(img):
    """Estimate backscatter by measuring haze-like intensity in bright areas."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mask = v_channel > np.percentile(v_channel, 95)
    if np.sum(mask) == 0:
        return np.array([0.0])
    return np.array([np.mean(v_channel[mask]) / 255.0])

def chromatic_depth_cue(img):
    """Measure red attenuation compared to green and blue."""
    b, g, r = cv2.split(img.astype(np.float32))
    epsilon = 1e-5
    r_g = np.mean(r) / (np.mean(g) + epsilon)
    r_b = np.mean(r) / (np.mean(b) + epsilon)
    return np.array([r_g, r_b])

def veiling_light_gradient(img):
    """Gradient of brightness as a proxy for veiling light (blur)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.array([np.mean(magnitude), np.std(magnitude)])

def local_contrast_entropy(img, patch_size=32):
    """Entropy across patches to model texture and detail."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    entropy_list = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            hist = cv2.calcHist([patch], [0], None, [256], [0,256]).flatten()
            p = hist / (np.sum(hist) + 1e-8)
            entropy = -np.sum(p * np.log2(p + 1e-8))
            entropy_list.append(entropy)
    return np.array([np.mean(entropy_list), np.std(entropy_list)])

def perceptual_fog_index(img):
    """Estimate perceptual visibility loss."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    chroma = np.std(hsv[:, :, 1])
    fog_index = brightness / (chroma + 1e-5)
    return np.array([fog_index])


# -------------------------- Data Handling ------------------------------

def load_mos(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.lower()
    df.rename(columns={'name': 'image'}, inplace=True)
    return df

def extract_all_features(img):
    return {
        'color': extract_color_features(img),
        'texture': extract_texture_features(img),
        'structure': extract_structure_features(img),
        'frequency': extract_frequency_features(img),
        'hdaf': extract_hdaf_features(img),
        'fpff': extract_fpff_features(img),
        'colorfulness': extract_colorfulness(img),
        'spatial': extract_spatial_info(img),
        'turbidity': extract_turbidity(img),
        'chromatic': extract_chromatic_aberration(img),
        'Multi': extract_features_multiscale(img)
    }
def extract_all_features(img):
    return {
        'color': extract_color_features(img),
        'texture': extract_texture_features(img),
        'structure': extract_structure_features(img),
        'frequency': extract_frequency_features(img),
        'hdaf': extract_hdaf_features(img),
        'fpff': extract_fpff_features(img),
        'colorfulness': extract_colorfulness(img),
        'spatial': extract_spatial_info(img),
        'turbidity': extract_turbidity(img),
        'chromatic': extract_chromatic_aberration(img),
        'multi_color': extract_color_features_multiscale(img),
        'multi_texture': extract_texture_features_multiscale(img),
        'multi_structure': extract_structure_features_multiscale(img),
        'multi_frequency': extract_frequency_features_multiscale(img),
        'multi_hdaf': extract_hdaf_features_multiscale(img),
        'multi_fpff': extract_fpff_features_multiscale(img),
        # NEW novel features
        'backscatter': estimate_backscatter(img),
        'chromatic_depth': chromatic_depth_cue(img),
        'veiling_gradient': veiling_light_gradient(img),
        'local_entropy': local_contrast_entropy(img),
        'fog_index': perceptual_fog_index(img),
        
        'attenuation': light_attenuation_profile(img),
        'specular': specular_suppression_ratio(img),
        'edge_energy': edge_energy_ratio(img),
        'divergence': channel_divergence(img),
        'illumination': non_uniform_illumination(img),
        'pswod': extract_pswod_features(img)
    }



def extract_dataset_features(folder_path, mos_df):
    features_all, labels_all = [], []
    for _, row in tqdm(mos_df.iterrows(), total=len(mos_df)):
        filename = row['image']
        if not filename.lower().endswith('.png'):
            filename += '.png'
        
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipped: {img_path}")
            continue
        feat = extract_all_features(img)
        features_all.append(feat)
        labels_all.append(row['score'])
    return features_all, np.array(labels_all)

# -------------------------- Evaluation ------------------------------

def evaluate_feature_combinations(features_all, labels_all, group_names):
    results = []
    for i in range(1, len(group_names) + 1):
        for combo in combinations(group_names, i):
            selected_features = np.array([
                np.concatenate([f[g] for g in combo]) for f in features_all
            ])
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features, labels_all, test_size=0.2, random_state=42)
            model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            srcc = spearmanr(y_test, preds).correlation
            plcc = pearsonr(y_test, preds)[0]
            results.append((combo, rmse, srcc, plcc))     
            print(f"Tested: {combo} ➤ RMSE: {rmse:.3f}, SRCC: {srcc:.3f}, PLCC: {plcc:.3f}")
    return sorted(results, key=lambda x: -x[2])

# -------------------------- Main ------------------------------
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_kde_curves(mos_true, mos_pred, save_path="ablation_results/kde_mos_comparison.png"):
    """
    Compare distributions of true and predicted MOS using KDE plots.
    """
    plt.figure(figsize=(8, 5))
    
    sns.kdeplot(mos_true, label="True MOS", color="blue", linewidth=2)
    sns.kdeplot(mos_pred, label="Predicted MOS", color="orange", linewidth=2)
    
    plt.xlabel("MOS Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("KDE: True vs. Predicted MOS", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Saved KDE plot → {save_path}")

def plot_mos_distribution(mos_values, save_path="ablation_results/mos_distribution.png"):
    """
    Plots the distribution of MOS values as a histogram with mean and median lines.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(mos_values, bins=20, kde=True, color="skyblue", edgecolor='black')
    
    mean_val = np.mean(mos_values)
    median_val = np.median(mos_values)
    
    # Draw vertical lines
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median = {median_val:.2f}')
    
    plt.xlabel("MOS Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of MOS Values", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved MOS distribution plot → {save_path}")
    
    
def main():
    image_folder = 'UWIQA'
    mos_file = os.path.join('MOS Values', 'IQA Value.xlsx')
    df = load_mos(mos_file)
    # Convert image names to strings and pad with zeros to match the format '0001', '0002', etc.
    df['image'] = df['image'].astype(str).str.zfill(4)
    

    print("🔍 Extracting all features...")
    features_all, labels_all = extract_dataset_features(image_folder, df)

    plot_mos_distribution(labels_all)

    combo = ['color', 'multi_texture', 'multi_frequency', 'edge_energy', 'local_entropy', 'multi_fpff', 'illumination', 'specular', 'veiling_gradient']

    print(f"\n🧪 Evaluating with all features: {combo}")

    selected_features = np.array([
        np.concatenate([f[g] for g in combo]) for f in features_all
    ])

    # Store metrics for 1000 runs
    rmse_list, srcc_list, plcc_list,krocc_list = [], [], [],[]

    print("\n⏳ Starting 1000 random train-test splits...")
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(
            selected_features, labels_all, test_size=0.2, random_state=None
        )

        model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        srcc = spearmanr(y_test, preds).correlation
        plcc = pearsonr(y_test, preds)[0]
        krocc=kendalltau(y_test, preds).correlation


        rmse_list.append(rmse)
        srcc_list.append(srcc)
        plcc_list.append(plcc)
        krocc_list.append(krocc)


        if (i + 1) % 100 == 0:
            print(f"  ✔ Run {i+1}/1000")

    compare_kde_curves(y_test, preds)

    # Compute and display median values
    median_rmse = np.median(rmse_list)
    median_srcc = np.median(srcc_list)
    median_plcc = np.median(plcc_list)
    median_krocc = np.median(krocc_list)


    print("\n📊 Final Median Results After 1000 Runs:")
    print(f"   ➤ RMSE: {median_rmse:.3f}")
    print(f"   ➤ SRCC: {median_srcc:.3f}")
    print(f"   ➤ PLCC: {median_plcc:.3f}")
    print(f"   ➤ KROCC: {median_krocc:.3f}")


if __name__ == "__main__":
    main()
