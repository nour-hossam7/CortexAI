"""Analytics — training curves, evaluation plots, embeddings."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.utils.disclaimer import render_disclaimer
from src.ui.utils.paths import get_paths

setup_page("Analytics", icon="📊")

st.title("Analytics & Evaluation")
render_disclaimer()

paths = get_paths()

tab_cv, tab_fusion, tab_data = st.tabs(["Computer Vision", "Fusion Model", "Cohort Statistics"])

with tab_cv:
    st.subheader("Segmentation Training")
    hist_path = paths.cv_results / "training_history.csv"
    if hist_path.exists():
        import pandas as pd
        hist = pd.read_csv(hist_path)
        if "validation_dice" in hist.columns:
            st.line_chart(hist.set_index("epoch")[["train_loss", "validation_loss"]])
            st.line_chart(hist.set_index("epoch")[["validation_dice"]])
        else:
            st.dataframe(hist.head(20), width='stretch')
    else:
        st.info("Training history not found.")

    st.subheader("Evaluation Results")
    eval_path = paths.cv_results / "evaluation_results.csv"
    if eval_path.exists():
        import pandas as pd
        ev = pd.read_csv(eval_path)
        st.dataframe(ev, width='stretch', hide_index=True)
        if "mean_after" in ev.columns:
            st.metric("Mean Dice (post-proc)", f"{ev['mean_after'].mean():.3f}")

    for name in ["loss_curve.png", "dice_curve.png", "subregion_dice.png", "dice_distribution.png"]:
        fig = paths.cv_figures / name
        if fig.exists():
            st.image(str(fig), caption=name, width='stretch')

with tab_fusion:
    st.subheader("Fusion Model Performance")
    fusion_hist = paths.root / "models" / "fusion" / "training_history.csv"
    if fusion_hist.exists():
        import pandas as pd
        fh = pd.read_csv(fusion_hist)
        if "val_acc" in fh.columns:
            st.line_chart(fh.set_index("epoch")[["train_loss", "val_loss"]])
            st.line_chart(fh.set_index("epoch")[["train_acc", "val_acc"]])
    else:
        st.info("Fusion training history not in models/fusion/ — may exist only on Kaggle output.")

    for name in [
        "confusion_matrices.png", "training_history.png", "calibration.png",
        "confidence_distribution.png", "repr_pca_tsne_risk.png", "repr_pca_splits.png",
    ]:
        fig = paths.fusion_figures / name
        if fig.exists():
            st.image(str(fig), caption=name.replace("_", " ").title(), width='stretch')

    if not any(paths.fusion_figures.glob("*.png")) if paths.fusion_figures.exists() else True:
        st.caption("Fusion evaluation figures expected at reports/figures/fusion/")

with tab_data:
    st.subheader("Feature Metadata")
    meta_path = paths.cv_results / "feature_metadata.csv"
    if meta_path.exists():
        import pandas as pd
        meta = pd.read_csv(meta_path)
        st.dataframe(meta.describe(), width='stretch')
        if "wt_volume" in meta.columns:
            st.bar_chart(meta.set_index("patient_id")["wt_volume"].head(30))

    norm_path = paths.cv_results / "feature_norm_table.csv"
    if norm_path.exists():
        import pandas as pd
        st.dataframe(pd.read_csv(norm_path).head(20), width='stretch')
