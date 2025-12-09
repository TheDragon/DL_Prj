import json
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import skimage.transform
import streamlit as st
import torch
import torch.optim.adam as _adam
from PIL import Image
from torch.serialization import add_safe_globals

import caption as caption_module
import models

# Use a non-interactive backend for Streamlit rendering.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (deferred after backend choice)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Keep the caption module on the same device the app picked.
caption_module.device = device

# Streamlit renamed cache helper; keep compatibility with older versions.
cache_resource = st.cache_resource if hasattr(st, "cache_resource") else st.cache


def _patch_legacy_adam_state():
    """Torch 2.x is stricter when unpickling legacy optimizer states; loosen it."""
    if hasattr(_adam, "Adam"):
        def _safe_setstate(self, state):
            try:
                for key, value in state.items():
                    setattr(self, key, value)
                if not hasattr(self, "defaults"):
                    self.defaults = {}
            except Exception:
                self.defaults = getattr(self, "defaults", {})
        try:
            _adam.Adam.__setstate__ = _safe_setstate  # type: ignore[attr-defined]
        except Exception:
            # Best-effort patch; fall back to default behavior if it fails.
            pass


@cache_resource(show_spinner=False)
def load_artifacts(model_path: str, word_map_path: str):
    """Load encoder/decoder and the word maps once and cache them."""
    _patch_legacy_adam_state()
    try:
        add_safe_globals([models])
    except Exception:
        # Safe-unpickling isn't critical for local usage; ignore if unavailable.
        pass

    checkpoint = torch.load(model_path, map_location=str(device), weights_only=False)
    decoder = checkpoint["decoder"].to(device)
    decoder.eval()
    encoder = checkpoint["encoder"].to(device)
    encoder.eval()

    with open(word_map_path, "r") as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    return encoder, decoder, word_map, rev_word_map


def tokens_to_caption(seq, rev_word_map):
    """Convert a sequence of token ids into human-readable caption text."""
    words = []
    for idx in seq:
        word = rev_word_map.get(idx, "")
        if word in {"<start>", "<pad>"}:
            continue
        if word == "<end>":
            break
        words.append(word)
    return " ".join(words)


def build_attention_figure(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Create a matplotlib figure that overlays attention for each generated token.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((14 * 24, 14 * 24), Image.LANCZOS)

    words = [rev_word_map.get(ind, "") for ind in seq]
    n_cols = 5
    n_rows = int(np.ceil(len(words) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for t, word in enumerate(words):
        if t >= len(axes) or t > 50:
            break
        ax = axes[t]
        ax.text(0, 1, word, color="black", backgroundcolor="white", fontsize=10)
        ax.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), (14 * 24, 14 * 24))
        if t == 0:
            ax.imshow(alpha, alpha=0)
        else:
            ax.imshow(alpha, alpha=0.8)
        ax.set_axis_off()

    for ax in axes[len(words):]:
        ax.set_axis_off()

    plt.tight_layout()
    return fig


def run_caption(image_path, encoder, decoder, word_map, beam_size):
    """Run beam-search captioning with no_grad and return tokens plus attention maps."""
    with torch.no_grad():
        seq, alphas = caption_module.caption_image_beam_search(
            encoder, decoder, image_path, word_map, beam_size
        )
    return seq, torch.FloatTensor(alphas)


def _resolve_image_path(uploaded_file, fallback_path):
    """
    Save the uploaded file to a temp path if provided, otherwise return the fallback.
    Returns (path, temp_created).
    """
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name, True
    if fallback_path:
        return fallback_path, False
    return None, False


def main():
    st.set_page_config(
        page_title="Show, Attend and Tell",
        page_icon="🖼️",
        layout="wide",
    )
    st.title("Image Captioning with Attention")
    st.caption(f"Running on {device.type.upper()} (set by PyTorch).")

    default_model = "BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
    default_word_map = "WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
    default_image_path = ""

    col_inputs, col_options = st.columns([2, 1])
    with col_inputs:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        image_path_text = st.text_input("...or provide a local image path", value=default_image_path, placeholder="path/to/image.jpg")
    with col_options:
        beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5, step=1)
        smooth = st.checkbox("Smooth attention overlay", value=True)
        model_path = st.text_input("Model checkpoint path", value=default_model)
        word_map_path = st.text_input("Word map JSON path", value=default_word_map)

    if uploaded is not None:
        try:
            preview_image = Image.open(uploaded).convert("RGB")
            st.image(preview_image, caption="Uploaded image", use_column_width=True)
        except Exception as exc:
            st.warning(f"Could not read the uploaded file: {exc}")
    elif image_path_text and Path(image_path_text).exists():
        try:
            preview_image = Image.open(image_path_text).convert("RGB")
            st.image(preview_image, caption=f"Image from {image_path_text}", use_column_width=True)
        except Exception as exc:
            st.warning(f"Could not open {image_path_text}: {exc}")

    if st.button("Generate caption"):
        image_path, is_temp = _resolve_image_path(uploaded, image_path_text)
        if image_path is None or not Path(image_path).exists():
            st.error("Please upload an image or provide a valid path.")
            return
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: {model_path}")
            return
        if not Path(word_map_path).exists():
            st.error(f"Word map JSON not found: {word_map_path}")
            return

        try:
            with st.spinner("Loading model..."):
                encoder, decoder, word_map, rev_word_map = load_artifacts(
                    str(Path(model_path).resolve()),
                    str(Path(word_map_path).resolve()),
                )
            with st.spinner("Running beam search..."):
                seq, alphas = run_caption(image_path, encoder, decoder, word_map, beam_size)

            caption_text = tokens_to_caption(seq, rev_word_map)
            st.subheader("Caption")
            st.write(caption_text if caption_text else "(Empty caption)")

            fig = build_attention_figure(image_path, seq, alphas, rev_word_map, smooth=smooth)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Captioning failed: {exc}")
        finally:
            if is_temp and image_path and Path(image_path).exists():
                try:
                    os.remove(image_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
