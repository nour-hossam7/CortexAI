"""Tumor segmentation analysis: measurements, bounding boxes, localization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch
from scipy import ndimage

# Internal label convention: 0=BG, 1=NCR, 2=Edema, 3=ET


@dataclass
class BoundingBox3D:
    lesion_id: int
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    width: int
    height: int
    depth: int
    center_z: float
    center_y: float
    center_x: float
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    largest_diameter_mm: float
    equivalent_diameter_mm: float


@dataclass
class TumorStatistics:
    patient_id: str
    spacing_mm: tuple[float, float, float]
    wt_voxels: int
    tc_voxels: int
    et_voxels: int
    wt_volume_cm3: float
    tc_volume_cm3: float
    et_volume_cm3: float
    wt_area_mm2: float
    largest_diameter_mm: float
    equivalent_diameter_mm: float
    tumor_percentage: float
    brain_percentage: float
    bbox_width_mm: float
    bbox_height_mm: float
    bbox_depth_mm: float
    tumor_center: tuple[float, float, float]
    tumor_laterality: str
    tumor_lobe: str
    largest_slice_axis: str
    largest_slice_index: int
    largest_component_voxels: int
    num_lesions: int
    bboxes: list[BoundingBox3D]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bboxes"] = [asdict(b) for b in self.bboxes]
        return d

    def table_rows(self) -> list[tuple[str, str]]:
        return [
            ("Whole Tumor Volume (cm³)", f"{self.wt_volume_cm3:.2f}"),
            ("Tumor Core Volume (cm³)", f"{self.tc_volume_cm3:.2f}"),
            ("Enhancing Tumor Volume (cm³)", f"{self.et_volume_cm3:.2f}"),
            ("WT Voxels", f"{self.wt_voxels:,}"),
            ("TC Voxels", f"{self.tc_voxels:,}"),
            ("ET Voxels", f"{self.et_voxels:,}"),
            ("Tumor Area (max slice, mm²)", f"{self.wt_area_mm2:.1f}"),
            ("Largest Diameter (mm)", f"{self.largest_diameter_mm:.1f}"),
            ("Equivalent Diameter (mm)", f"{self.equivalent_diameter_mm:.1f}"),
            ("Tumor Percentage (%)", f"{self.tumor_percentage:.2f}"),
            ("Brain Occupancy (%)", f"{self.brain_percentage:.2f}"),
            ("Bounding Box (W×H×D mm)", f"{self.bbox_width_mm:.1f} × {self.bbox_height_mm:.1f} × {self.bbox_depth_mm:.1f}"),
            ("Tumor Center (z, y, x)", f"({self.tumor_center[0]:.1f}, {self.tumor_center[1]:.1f}, {self.tumor_center[2]:.1f})"),
            ("Tumor Laterality", self.tumor_laterality),
            ("Tumor Lobe (approx.)", self.tumor_lobe),
            ("Largest Slice", f"{self.largest_slice_axis} #{self.largest_slice_index}"),
            ("Largest Connected Component (vox)", f"{self.largest_component_voxels:,}"),
            ("Lesion Count", str(self.num_lesions)),
        ]


def _mask(pred: torch.Tensor | np.ndarray, labels: set[int]) -> np.ndarray:
    arr = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    out = np.zeros_like(arr, dtype=bool)
    for label in labels:
        out |= arr == label
    return out


def voxel_volume_mm3(spacing: tuple[float, float, float]) -> float:
    return float(spacing[0] * spacing[1] * spacing[2])


def compute_tumor_statistics(
    prediction: torch.Tensor | np.ndarray,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    patient_id: str = "Unknown",
    brain_mask: np.ndarray | None = None,
) -> TumorStatistics:
    pred = prediction.detach().cpu().numpy() if isinstance(prediction, torch.Tensor) else np.asarray(prediction)
    vox_mm3 = voxel_volume_mm3(spacing_mm)

    wt = _mask(pred, {1, 2, 3})
    tc = _mask(pred, {1, 3})
    et = _mask(pred, {3})

    wt_vox = int(wt.sum())
    tc_vox = int(tc.sum())
    et_vox = int(et.sum())

    bboxes = detect_bounding_boxes(wt, spacing_mm)
    num_lesions = len(bboxes)

    if brain_mask is None:
        brain_mask = pred >= 0
    brain_vox = max(int(brain_mask.sum()), 1)

    largest_comp_vox = bboxes[0].voxel_count if bboxes else wt_vox
    primary = bboxes[0] if bboxes else None

    wt_vol_cm3 = wt_vox * vox_mm3 / 1000.0
    tc_vol_cm3 = tc_vox * vox_mm3 / 1000.0
    et_vol_cm3 = et_vox * vox_mm3 / 1000.0

    largest_diam = primary.largest_diameter_mm if primary else 0.0
    eq_diam = primary.equivalent_diameter_mm if primary else 0.0

    axis_name, slice_idx, area_mm2 = _largest_slice(wt, spacing_mm)
    center = _wt_centroid(wt)
    laterality = _estimate_laterality(center, pred.shape, spacing_mm)
    lobe = _estimate_lobe(center, pred.shape)

    if primary:
        bw = primary.width * spacing_mm[2]
        bh = primary.height * spacing_mm[1]
        bd = primary.depth * spacing_mm[0]
        bbox_center = (primary.center_z, primary.center_y, primary.center_x)
    else:
        bw = bh = bd = 0.0
        bbox_center = center

    return TumorStatistics(
        patient_id=patient_id,
        spacing_mm=spacing_mm,
        wt_voxels=wt_vox,
        tc_voxels=tc_vox,
        et_voxels=et_vox,
        wt_volume_cm3=wt_vol_cm3,
        tc_volume_cm3=tc_vol_cm3,
        et_volume_cm3=et_vol_cm3,
        wt_area_mm2=area_mm2,
        largest_diameter_mm=largest_diam,
        equivalent_diameter_mm=eq_diam,
        tumor_percentage=100.0 * wt_vox / brain_vox,
        brain_percentage=100.0 * brain_vox / pred.size,
        bbox_width_mm=bw,
        bbox_height_mm=bh,
        bbox_depth_mm=bd,
        tumor_center=bbox_center,
        tumor_laterality=laterality,
        tumor_lobe=lobe,
        largest_slice_axis=axis_name,
        largest_slice_index=slice_idx,
        largest_component_voxels=largest_comp_vox,
        num_lesions=num_lesions,
        bboxes=bboxes,
    )


def detect_bounding_boxes(
    wt_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> list[BoundingBox3D]:
    labeled, n = ndimage.label(wt_mask)
    if n == 0:
        return []

    vox_mm3 = voxel_volume_mm3(spacing_mm)
    boxes: list[BoundingBox3D] = []

    for lesion_id in range(1, n + 1):
        coords = np.argwhere(labeled == lesion_id)
        if coords.size == 0:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        voxels = int(coords.shape[0])
        vol_mm3 = voxels * vox_mm3
        eq_d = 2.0 * ((3.0 * vol_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
        cz, cy, cx = coords.mean(axis=0)
        dz = (z1 - z0 + 1) * spacing_mm[0]
        dy = (y1 - y0 + 1) * spacing_mm[1]
        dx = (x1 - x0 + 1) * spacing_mm[2]
        largest_d = max(dz, dy, dx)

        boxes.append(
            BoundingBox3D(
                lesion_id=lesion_id,
                z_min=int(z0),
                z_max=int(z1),
                y_min=int(y0),
                y_max=int(y1),
                x_min=int(x0),
                x_max=int(x1),
                width=int(x1 - x0 + 1),
                height=int(y1 - y0 + 1),
                depth=int(z1 - z0 + 1),
                center_z=float(cz),
                center_y=float(cy),
                center_x=float(cx),
                voxel_count=voxels,
                volume_mm3=vol_mm3,
                volume_cm3=vol_mm3 / 1000.0,
                largest_diameter_mm=largest_d,
                equivalent_diameter_mm=eq_d,
            )
        )

    boxes.sort(key=lambda b: b.voxel_count, reverse=True)
    return boxes


def draw_bbox_on_slice(
    rgb: np.ndarray,
    bboxes: list[BoundingBox3D],
    axis: int,
    slice_idx: int,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
) -> np.ndarray:
    out = rgb.copy()
    for box in bboxes:
        if axis == 0 and not (box.z_min <= slice_idx <= box.z_max):
            continue
        if axis == 1 and not (box.y_min <= slice_idx <= box.y_max):
            continue
        if axis == 2 and not (box.x_min <= slice_idx <= box.x_max):
            continue

        if axis == 0:
            y0, y1, x0, x1 = box.y_min, box.y_max, box.x_min, box.x_max
        elif axis == 1:
            y0, y1, x0, x1 = box.z_min, box.z_max, box.x_min, box.x_max
        else:
            y0, y1, x0, x1 = box.z_min, box.z_max, box.y_min, box.y_max

        _draw_rect(out, x0, y0, x1, y1, color, thickness)
    return out


def mask_to_rgba_overlay(
    base_gray: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    if mask_slice.shape != base_gray.shape:
        if mask_slice.T.shape == base_gray.shape:
            mask_slice = mask_slice.T
        else:
            import cv2

            mask_slice = cv2.resize(
                mask_slice.astype(np.int32),
                (base_gray.shape[1], base_gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

    base = np.stack([base_gray] * 3, axis=-1)
    overlay = base.copy()
    colors = {
        1: np.array([0.9, 0.2, 0.2]),
        2: np.array([0.2, 0.8, 0.2]),
        3: np.array([0.2, 0.4, 1.0]),
    }
    for label, color in colors.items():
        m = mask_slice == label
        if m.any():
            overlay[m] = (1 - alpha) * base[m] + alpha * color
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def _draw_rect(img, x0, y0, x1, y1, color, thickness):
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    for t in range(thickness):
        yy0 = max(0, y0 - t)
        yy1 = min(h - 1, y1 + t)
        xx0 = max(0, x0 - t)
        xx1 = min(w - 1, x1 + t)
        img[yy0, xx0:xx1 + 1] = color
        img[yy1, xx0:xx1 + 1] = color
        img[yy0:yy1 + 1, xx0] = color
        img[yy0:yy1 + 1, xx1] = color


def _wt_centroid(wt: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(wt)
    if coords.size == 0:
        d, h, w = wt.shape
        return (d / 2, h / 2, w / 2)
    c = coords.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def _largest_slice(wt: np.ndarray, spacing: tuple[float, float, float]) -> tuple[str, int, float]:
    best_area = 0.0
    best_axis = "Axial"
    best_idx = 0
    axes = [(0, "Axial", spacing[1] * spacing[2]), (1, "Coronal", spacing[0] * spacing[2]), (2, "Sagittal", spacing[0] * spacing[1])]
    for axis, name, pix_area in axes:
        for i in range(wt.shape[axis]):
            sl = np.take(wt, i, axis=axis)
            area = sl.sum() * pix_area
            if area > best_area:
                best_area = area
                best_axis = name
                best_idx = i
    return best_axis, best_idx, float(best_area)


def _estimate_laterality(center: tuple[float, float, float], shape: tuple, spacing: tuple[float, float, float]) -> str:
    _, _, cx = center
    mid = shape[2] / 2.0
    if abs(cx - mid) < shape[2] * 0.08:
        return "Midline / Bilateral"
    return "Right" if cx > mid else "Left"


def _estimate_lobe(center: tuple[float, float, float], shape: tuple) -> str:
    cz, cy, _ = center
    dz, hy, _ = shape
    anterior = cz < dz * 0.35
    posterior = cz > dz * 0.65
    superior = cy < hy * 0.35
    inferior = cy > hy * 0.65
    if anterior and not superior:
        return "Frontal (approx.)"
    if anterior and superior:
        return "Frontal-Parietal (approx.)"
    if posterior and superior:
        return "Occipital (approx.)"
    if posterior:
        return "Parietal-Occipital (approx.)"
    if inferior:
        return "Temporal (approx.)"
    return "Central / Deep (approx.)"
