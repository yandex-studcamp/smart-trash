from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_PATCHCORE_SCORE_THRESHOLD = 0.5


@dataclass(slots=True)
class CropInfo:
    bbox: tuple[int, int, int, int]
    source: str
    heatmap_quantile: float | None
    coarse_positive_ratio: float
    mask_positive_ratio: float
    components_kept: int


def normalize_visual_map(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return None
    array = np.squeeze(array)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros(array.shape, dtype=np.uint8)
    finite_values = array[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if max_value - min_value < 1e-12:
        normalized = np.zeros(array.shape, dtype=np.float32)
    else:
        normalized = (array - min_value) / (max_value - min_value)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def heatmap_rgb(normalized_map: np.ndarray) -> np.ndarray:
    values = normalized_map.astype(np.float32) / 255.0
    red = values
    green = np.clip(1.0 - np.abs(values - 0.5) * 2.0, 0.0, 1.0)
    blue = np.clip(1.0 - values, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1) * 255.0


def get_attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def get_prediction_score(prediction: Any) -> float | None:
    value = get_attr(prediction, ("score", "pred_score", "image_score"), None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def get_prediction_threshold(prediction: Any, fallback: float) -> float:
    value = get_attr(prediction, ("score_threshold", "threshold", "image_threshold"), None)
    if value is None:
        return float(fallback)
    try:
        return float(value)
    except Exception:
        return float(fallback)


def get_prediction_label(prediction: Any) -> int | str | None:
    value = get_attr(prediction, ("label", "pred_label", "class_label"), None)
    if value is None:
        return None
    try:
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return int(value)
    except Exception:
        pass
    return str(value)


def item_present_from_prediction(prediction: Any, fallback_threshold: float = DEFAULT_PATCHCORE_SCORE_THRESHOLD) -> bool:
    is_anomaly = get_attr(prediction, ("is_anomaly", "pred_is_anomaly"), None)
    if is_anomaly is not None:
        if isinstance(is_anomaly, str):
            return is_anomaly.lower() in {"true", "1", "anomaly", "item", "yes"}
        return bool(is_anomaly)

    label = get_prediction_label(prediction)
    if isinstance(label, str):
        label_l = label.lower()
        if label_l in {"anomaly", "item", "true", "1"}:
            return True
        if label_l in {"normal", "no_item", "no item", "false", "0"}:
            return False
    elif label is not None:
        return int(label) == 1

    score = get_prediction_score(prediction)
    threshold = get_prediction_threshold(prediction, fallback_threshold)
    if score is None:
        return False
    return float(score) >= float(threshold)


def resize_binary_mask(mask: np.ndarray | None, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)

    array = np.squeeze(np.asarray(mask))
    if array.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        max_value = float(finite.max(initial=0.0))
        threshold = 0.5 if max_value <= 1.0 else 127.0
        array = (finite > threshold).astype(np.uint8) * 255
    else:
        array = (array > 0).astype(np.uint8) * 255

    return cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)


def resize_heatmap(anomaly_map: np.ndarray | None, image_size: tuple[int, int]) -> np.ndarray | None:
    normalized = normalize_visual_map(anomaly_map)
    if normalized is None:
        return None
    return np.asarray(
        Image.fromarray(normalized, mode="L").resize(image_size, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def choose_heatmap_quantile(coarse_positive_ratio: float) -> float:
    if coarse_positive_ratio >= 0.85:
        return 70.0
    if coarse_positive_ratio >= 0.65:
        return 80.0
    return 90.0


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    kernel_size = max(3, int(round(min(width, height) * 0.018)) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return cleaned


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return np.zeros(mask.shape[:2], dtype=np.uint8)

    kept = np.zeros(mask.shape[:2], dtype=np.uint8)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == label] = 255
    return kept


def expanded_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    ratio: float = 0.35,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    pad = int(round(max(x2 - x1, y2 - y1) * ratio))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def align_background_to_image(
    image_array: np.ndarray,
    background_array: np.ndarray,
    ignore_mask: np.ndarray,
) -> np.ndarray:
    image_f = image_array.astype(np.float32)
    bg_f = background_array.astype(np.float32)
    valid = ignore_mask == 0

    if int(valid.sum()) < 500:
        return background_array

    aligned = bg_f.copy()
    for channel in range(3):
        source = bg_f[:, :, channel][valid]
        target = image_f[:, :, channel][valid]

        source_p10, source_p90 = np.percentile(source, [10, 90])
        target_p10, target_p90 = np.percentile(target, [10, 90])
        source_range = max(1.0, float(source_p90 - source_p10))
        target_range = max(1.0, float(target_p90 - target_p10))

        gain = float(np.clip(target_range / source_range, 0.75, 1.25))
        bias = float(np.median(target) - gain * np.median(source))
        aligned[:, :, channel] = bg_f[:, :, channel] * gain + bias

    return np.clip(aligned, 0, 255).astype(np.uint8)


def background_difference_mask(
    image: Image.Image,
    background: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    threshold_min: float,
    threshold_max: float,
    align_background: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if background.size != image.size:
        background = background.resize(image.size, Image.Resampling.BILINEAR)

    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    background_array = np.asarray(background.convert("RGB"), dtype=np.uint8)

    if align_background:
        ignore_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        rx1, ry1, rx2, ry2 = expanded_bbox(bbox, image.size, ratio=0.45)
        ignore_mask[ry1:ry2, rx1:rx2] = 255
        background_array = align_background_to_image(image_array, background_array, ignore_mask)

    image_blur = cv2.GaussianBlur(image_array, (5, 5), 0)
    background_blur = cv2.GaussianBlur(background_array, (5, 5), 0)

    rgb_delta = np.max(
        np.abs(image_blur.astype(np.int16) - background_blur.astype(np.int16)),
        axis=2,
    ).astype(np.float32)

    image_lab = cv2.cvtColor(image_blur, cv2.COLOR_RGB2LAB).astype(np.int16)
    background_lab = cv2.cvtColor(background_blur, cv2.COLOR_RGB2LAB).astype(np.int16)
    light_delta = np.abs(image_lab[:, :, 0] - background_lab[:, :, 0]).astype(np.float32)
    chroma_delta = np.sqrt(
        (image_lab[:, :, 1] - background_lab[:, :, 1]).astype(np.float32) ** 2
        + (image_lab[:, :, 2] - background_lab[:, :, 2]).astype(np.float32) ** 2
    )

    diff = np.maximum(rgb_delta, np.maximum(chroma_delta * 1.35, light_delta * 0.45))
    diff = cv2.GaussianBlur(np.clip(diff, 0, 255).astype(np.uint8), (5, 5), 0)

    rx1, ry1, rx2, ry2 = expanded_bbox(bbox, image.size, ratio=0.35)
    roi_diff = diff[ry1:ry2, rx1:rx2]
    if roi_diff.size == 0:
        threshold = float(threshold_min)
    else:
        otsu_threshold, _ = cv2.threshold(roi_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(float(threshold_min), min(float(otsu_threshold), float(threshold_max)))

    mask = (diff >= threshold).astype(np.uint8) * 255
    mask = clean_binary_mask(mask)

    x1, y1, x2, y2 = bbox
    crop_area = max(1, (x2 - x1) * (y2 - y1))
    min_area = max(18, int(crop_area * 0.0025))
    mask = filter_small_components(mask, min_area=min_area)
    return mask, diff, threshold


def edge_difference_mask(
    image: Image.Image,
    background: Image.Image,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    if background.size != image.size:
        background = background.resize(image.size, Image.Resampling.BILINEAR)

    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bg_array = np.asarray(background.convert("RGB"), dtype=np.uint8)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)

    edges = cv2.Canny(gray, 35, 95)
    bg_edges = cv2.Canny(bg_gray, 35, 95)
    bg_edges_dilated = cv2.dilate(bg_edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
    object_edges = np.where((edges > 0) & (bg_edges_dilated == 0), 255, 0).astype(np.uint8)

    full = np.zeros_like(object_edges)
    x1, y1, x2, y2 = bbox
    full[y1:y2, x1:x2] = object_edges[y1:y2, x1:x2]

    kernel_size = max(3, int(round(min(x2 - x1, y2 - y1) * 0.025)) | 1)
    full = cv2.dilate(full, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    return full


def keep_components_touching_anchor(mask: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    labels_count, labels, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return np.zeros(mask.shape[:2], dtype=np.uint8)

    anchor = anchor_mask > 0
    if not np.any(anchor):
        return mask

    kept = np.zeros(mask.shape[:2], dtype=np.uint8)
    for label in range(1, labels_count):
        component = labels == label
        if np.any(component & anchor):
            kept[component] = 255
    return kept


def fill_external_contours(mask: np.ndarray, *, use_hull: bool = False, min_area: int = 16) -> np.ndarray:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape[:2], dtype=np.uint8)
    if not contours:
        return filled

    drawable = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        drawable.append(cv2.convexHull(contour) if use_hull else contour)
    if drawable:
        cv2.drawContours(filled, drawable, contourIdx=-1, color=255, thickness=cv2.FILLED)
    return filled


def make_support_mask(
    *,
    seed: np.ndarray,
    edge_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    anchor_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    full_candidate = np.where((seed > 0) | (edge_mask > 0), 255, 0).astype(np.uint8)

    if np.any(anchor_mask):
        anchored = keep_components_touching_anchor(full_candidate, anchor_mask)
        if np.any(anchored):
            full_candidate = anchored

    crop = full_candidate[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros_like(seed)

    crop_h, crop_w = crop.shape[:2]
    if mode == "strict":
        open_ratio = 0.012
        close_ratio = 0.055
        dilate_ratio = 0.018
        use_hull = False
        max_coverage = 0.88
    else:
        open_ratio = 0.006
        close_ratio = 0.11
        dilate_ratio = 0.045
        use_hull = True
        max_coverage = 0.95

    open_k = max(3, int(round(min(crop_w, crop_h) * open_ratio)) | 1)
    close_k = max(5, int(round(min(crop_w, crop_h) * close_ratio)) | 1)
    dilate_k = max(3, int(round(min(crop_w, crop_h) * dilate_ratio)) | 1)

    crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, np.ones((open_k, open_k), dtype=np.uint8), iterations=1)
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((close_k, close_k), dtype=np.uint8), iterations=1)
    crop = cv2.dilate(crop, np.ones((dilate_k, dilate_k), dtype=np.uint8), iterations=1)
    crop = fill_external_contours(crop, use_hull=use_hull, min_area=max(16, int(crop_w * crop_h * 0.001)))

    object_mask = np.zeros_like(seed)
    object_mask[y1:y2, x1:x2] = crop

    crop_area = max(1, crop_w * crop_h)
    coverage = float((crop > 0).sum()) / float(crop_area)
    if coverage > max_coverage:
        strict_seed = seed[y1:y2, x1:x2]
        seed_coverage = float((strict_seed > 0).sum()) / float(crop_area)
        if seed_coverage >= 0.006:
            object_mask = np.zeros_like(seed)
            object_mask[y1:y2, x1:x2] = strict_seed

    return object_mask.astype(np.uint8)


def mask_coverage_inside_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    crop_area = max(1, (x2 - x1) * (y2 - y1))
    return float((mask[y1:y2, x1:x2] > 0).sum()) / float(crop_area)


def build_clean_object_mask(
    *,
    image: Image.Image,
    background: Image.Image | None,
    crop_info: CropInfo,
    spotter_mask: np.ndarray,
    clean_mode: str,
    threshold_min: float,
    threshold_max: float,
    anchor_dilate_ratio: float,
    align_background: bool,
    use_edge_support: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
    height, width = spotter_mask.shape[:2]
    x1, y1, x2, y2 = crop_info.bbox

    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 255

    spotter_inside = np.where((spotter_mask > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)
    anchor_kernel = max(7, int(round(max(x2 - x1, y2 - y1) * anchor_dilate_ratio)) | 1)
    spotter_anchor = cv2.dilate(spotter_inside, np.ones((anchor_kernel, anchor_kernel), dtype=np.uint8), iterations=1)
    spotter_anchor = np.where((spotter_anchor > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)

    if background is None:
        fallback = spotter_anchor if np.any(spotter_anchor) else spotter_inside
        if not np.any(fallback):
            fallback = bbox_mask
        return fallback.astype(np.uint8), None, None, None

    foreground_mask, diff_map, diff_threshold = background_difference_mask(
        image,
        background,
        crop_info.bbox,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        align_background=align_background,
    )

    seed = np.where((foreground_mask > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)
    if np.any(spotter_anchor):
        anchored_seed = keep_components_touching_anchor(seed, spotter_anchor)
        if np.any(anchored_seed):
            seed = anchored_seed

    if use_edge_support:
        edge_mask = edge_difference_mask(image, background, crop_info.bbox)
        edge_mask = np.where((edge_mask > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)
    else:
        edge_mask = np.zeros_like(seed)

    object_mask = make_support_mask(
        seed=seed,
        edge_mask=edge_mask,
        bbox=crop_info.bbox,
        anchor_mask=spotter_anchor,
        mode=clean_mode,
    )

    coverage = mask_coverage_inside_bbox(object_mask, crop_info.bbox)
    seed_coverage = mask_coverage_inside_bbox(seed, crop_info.bbox)

    if coverage < 0.006:
        if np.any(seed) and seed_coverage >= 0.002:
            object_mask = seed
        elif np.any(spotter_anchor):
            object_mask = spotter_anchor
        elif np.any(spotter_inside):
            object_mask = spotter_inside
        else:
            object_mask = bbox_mask
    elif coverage > (0.88 if clean_mode == "strict" else 0.95):
        if np.any(seed) and seed_coverage >= 0.006:
            object_mask = seed

    object_mask = clean_binary_mask(object_mask)
    object_mask = np.where((object_mask > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)
    return object_mask, foreground_mask, diff_map, diff_threshold


def make_clean_crop(
    *,
    image: Image.Image,
    background: Image.Image | None,
    bbox: tuple[int, int, int, int],
    object_mask: np.ndarray,
    fill_mode: str,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    crop = image_array[y1:y2, x1:x2]
    mask = object_mask[y1:y2, x1:x2]

    if fill_mode == "black":
        fill = np.zeros_like(crop, dtype=np.uint8)
    elif fill_mode == "white":
        fill = np.full_like(crop, 255, dtype=np.uint8)
    elif fill_mode == "median" and background is not None:
        bg = np.asarray(background.resize(image.size, Image.Resampling.BILINEAR).convert("RGB"), dtype=np.uint8)
        fill = bg[y1:y2, x1:x2]
    else:
        fill = np.full_like(crop, 128, dtype=np.uint8)

    soft_mask = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    alpha = np.expand_dims(np.clip(soft_mask, 0.0, 1.0), axis=2)
    cleaned = crop.astype(np.float32) * alpha + fill.astype(np.float32) * (1.0 - alpha)
    return Image.fromarray(np.clip(cleaned, 0, 255).astype(np.uint8), mode="RGB")


def bbox_from_mask(
    mask: np.ndarray,
    *,
    padding_ratio: float,
    min_padding: int,
    component_keep_ratio: float = 0.15,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray, int]:
    height, width = mask.shape[:2]
    positive = (mask > 0).astype(np.uint8)
    if int(positive.sum()) == 0:
        return None, np.zeros((height, width), dtype=np.uint8), 0

    cleaned = clean_binary_mask(positive * 255)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return None, cleaned, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    min_area = max(16, int(width * height * 0.001), int(largest * component_keep_ratio))
    keep_labels = [index + 1 for index, area in enumerate(areas) if int(area) >= min_area]
    if not keep_labels:
        keep_labels = [int(areas.argmax()) + 1]

    kept = np.isin(labels, keep_labels)
    ys, xs = np.where(kept)
    if xs.size == 0 or ys.size == 0:
        return None, cleaned, 0

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    pad = max(min_padding, int(round(max(x2 - x1, y2 - y1) * padding_ratio)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    return (x1, y1, x2, y2), kept.astype(np.uint8) * 255, len(keep_labels)


def crop_from_spotter_maps(
    *,
    image: Image.Image,
    prediction: Any,
    padding_ratio: float,
    min_padding: int,
) -> tuple[CropInfo, np.ndarray, np.ndarray | None]:
    width, height = image.size
    pred_mask = get_attr(prediction, ("pred_mask", "mask", "anomaly_mask"), None)
    anomaly_map = get_attr(prediction, ("anomaly_map", "heatmap", "map"), None)

    coarse_mask = resize_binary_mask(pred_mask, image.size)
    coarse_positive_ratio = float((coarse_mask > 0).mean())

    heatmap = resize_heatmap(anomaly_map, image.size)
    if heatmap is not None:
        quantile = choose_heatmap_quantile(coarse_positive_ratio)
        threshold = float(np.percentile(heatmap, quantile))
        candidate_mask = (heatmap >= threshold).astype(np.uint8) * 255
        if 0.0 < coarse_positive_ratio < 0.85:
            candidate_mask = np.where(coarse_mask > 0, candidate_mask, 0).astype(np.uint8)

        bbox, final_mask, components = bbox_from_mask(
            candidate_mask,
            padding_ratio=padding_ratio,
            min_padding=min_padding,
        )
        if bbox is not None:
            crop_info = CropInfo(
                bbox=bbox,
                source="anomaly_map",
                heatmap_quantile=quantile,
                coarse_positive_ratio=coarse_positive_ratio,
                mask_positive_ratio=float((final_mask > 0).mean()),
                components_kept=components,
            )
            return crop_info, final_mask, heatmap

    bbox, final_mask, components = bbox_from_mask(
        coarse_mask,
        padding_ratio=padding_ratio,
        min_padding=min_padding,
        component_keep_ratio=0.12,
    )
    if bbox is None:
        bbox = (0, 0, width, height)
        final_mask = np.ones((height, width), dtype=np.uint8) * 255
        source = "full_image_fallback"
    else:
        source = "pred_mask_fallback"

    crop_info = CropInfo(
        bbox=bbox,
        source=source,
        heatmap_quantile=None,
        coarse_positive_ratio=coarse_positive_ratio,
        mask_positive_ratio=float((final_mask > 0).mean()),
        components_kept=components,
    )
    return crop_info, final_mask, heatmap
