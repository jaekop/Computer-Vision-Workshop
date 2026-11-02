"""Gradio drawing UI for models trained in the current notebook runtime."""
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps

from .config import IMAGE_SIZE
from .models import DoodleModel


def sketch_to_pixels(value) -> np.ndarray:
    """Convert a black-on-white editor sketch to Quick, Draw!-style white ink.

    Gradio ImageEditor/Sketchpad returns a dict containing composite/layers.
    A blank or transparent canvas becomes an all-zero doodle.
    """
    if value is None:
        return np.zeros((1, IMAGE_SIZE ** 2), dtype=np.uint8)
    if isinstance(value, dict):
        value = value.get("composite") if value.get("composite") is not None else value.get("background")
    if value is None:
        return np.zeros((1, IMAGE_SIZE ** 2), dtype=np.uint8)
    arr = np.asarray(value)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        if arr.shape[2] >= 4:
            alpha = arr[..., 3].astype(np.float32) / 255.0
            gray = gray * alpha + 255.0 * (1.0 - alpha)
    else:
        raise ValueError("Unexpected drawing image shape")
    ink = np.clip(255.0 - gray, 0, 255).astype(np.uint8)
    # Preserve aspect ratio, then resize to the 28x28 training format.
    img = Image.fromarray(ink, mode="L")
    img = ImageOps.contain(img, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    canvas.paste(img, ((IMAGE_SIZE - img.width) // 2, (IMAGE_SIZE - img.height) // 2))
    return np.asarray(canvas, dtype=np.uint8).reshape(1, -1)


def launch_playground(
    models: Mapping[str, DoodleModel], share: bool = False, debug: bool = False
):
    """Launch a Gradio app in Colab (share=True) or locally (share=False).

    A hosted Space is separate from this runtime; it cannot magically access
    the models students trained inside their own Colab sessions.
    """
    import gradio as gr

    if not models:
        raise ValueError("Train at least one model first.")
    choices = list(models)
    with gr.Blocks(title="Doodle AI Playground") as demo:
        gr.Markdown("# Draw something!\nPick a model and try to fool it.")
        with gr.Row():
            editor = gr.Sketchpad(
                type="numpy", image_mode="RGBA", height=330,
                label="Draw with a dark pen on a light canvas"
            )
            with gr.Column():
                selector = gr.Dropdown(choices=choices, value=choices[0], label="Classifier")
                predict = gr.Button("Predict doodle", variant="primary")
                result = gr.Textbox(label="Prediction")
                scores = gr.JSON(label="Ranked scores (SVM: margins, not probabilities)")
                preview = gr.Image(label="What the model sees (28 × 28)", image_mode="L")
        def run(value, model_name):
            pixels = sketch_to_pixels(value)
            model = models[model_name]
            prediction = model.class_names[int(model.predict(pixels)[0])]
            ranking = [{"class": name, "score": round(score, 4)}
                       for name, score in model.ranked_scores(pixels)[:4]]
            return prediction, ranking, pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)
        predict.click(run, inputs=[editor, selector], outputs=[result, scores, preview])
    return demo.launch(share=share, debug=debug)

