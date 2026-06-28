# app_tk.py
# Tkinter front-end for the Fruit Freshness Detector.
# Mirrors the functionality of app.py (Streamlit) as a native desktop window.
#
# Run with:  python app_tk.py

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from PIL import Image, ImageTk

from predict import predict_image

SAMPLE_DIR = "sample_images"
ALLOWED = (".jpg", ".jpeg", ".png", ".webp")


def list_sample_images():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return [f for f in sorted(os.listdir(SAMPLE_DIR)) if f.lower().endswith(ALLOWED)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fruit Freshness Detector")
        self.resizable(False, False)

        self._pil_image = None      # current PIL image
        self._tk_image = None       # kept alive to prevent GC

        self._build_ui()
        self._refresh_sample_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Title
        tk.Label(
            self,
            text="🍎 Fruit Freshness Detector",
            font=("Helvetica", 18, "bold"),
        ).grid(row=0, column=0, columnspan=2, **pad)

        tk.Label(
            self,
            text=(
                "Upload a photo of an apple, banana, or orange\n"
                "and the model will guess whether it is fresh or rotten.\n"
                "You can also try one of the sample images below."
            ),
            justify="center",
            wraplength=460,
        ).grid(row=1, column=0, columnspan=2, **pad)

        # Source selector
        self._mode = tk.StringVar(value="upload")

        mode_frame = tk.LabelFrame(self, text="Image source", padx=8, pady=4)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        tk.Radiobutton(
            mode_frame,
            text="Upload my own",
            variable=self._mode,
            value="upload",
            command=self._on_mode_change,
        ).pack(side="left", padx=6)

        tk.Radiobutton(
            mode_frame,
            text="Use a sample image",
            variable=self._mode,
            value="sample",
            command=self._on_mode_change,
        ).pack(side="left", padx=6)

        # Upload controls
        self._upload_frame = tk.Frame(self)
        self._upload_frame.grid(row=3, column=0, columnspan=2, **pad)

        self._btn_browse = tk.Button(
            self._upload_frame,
            text="Browse…",
            width=14,
            command=self._browse,
        )
        self._btn_browse.pack(side="left", padx=4)

        self._lbl_file = tk.Label(self._upload_frame, text="No file selected", fg="gray")
        self._lbl_file.pack(side="left")

        # Sample controls (hidden initially)
        self._sample_frame = tk.Frame(self)
        self._sample_frame.grid(row=3, column=0, columnspan=2, **pad)
        self._sample_frame.grid_remove()

        tk.Label(self._sample_frame, text="Pick a sample image:").pack(side="left")

        self._sample_var = tk.StringVar()
        self._sample_combo = ttk.Combobox(
            self._sample_frame,
            textvariable=self._sample_var,
            state="readonly",
            width=30,
        )
        self._sample_combo.pack(side="left", padx=6)
        self._sample_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

        # Image preview
        self._preview_label = tk.Label(self, text="Waiting for an image…", fg="gray")
        self._preview_label.grid(row=4, column=0, columnspan=2, **pad)

        # Analyse button
        self._btn_analyse = tk.Button(
            self,
            text="Analyse 🔍",
            width=16,
            state="disabled",
            command=self._analyse,
        )
        self._btn_analyse.grid(row=5, column=0, columnspan=2, pady=6)

        # Result area
        result_frame = tk.LabelFrame(self, text="Result", padx=10, pady=6)
        result_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        self._lbl_verdict = tk.Label(result_frame, text="", font=("Helvetica", 13, "bold"))
        self._lbl_verdict.grid(row=0, column=0, columnspan=2, pady=(0, 4))

        tk.Label(result_frame, text="Fruit:").grid(row=1, column=0, sticky="e", padx=4)
        self._lbl_fruit = tk.Label(result_frame, text="—", width=18, anchor="w")
        self._lbl_fruit.grid(row=1, column=1, sticky="w")

        tk.Label(result_frame, text="Verdict:").grid(row=2, column=0, sticky="e", padx=4)
        self._lbl_state = tk.Label(result_frame, text="—", width=18, anchor="w")
        self._lbl_state.grid(row=2, column=1, sticky="w")

        tk.Label(result_frame, text="Confidence:").grid(row=3, column=0, sticky="e", padx=4)
        self._confidence_bar = ttk.Progressbar(result_frame, length=180, maximum=100)
        self._confidence_bar.grid(row=3, column=1, sticky="w", pady=2)

        self._lbl_confidence = tk.Label(result_frame, text="—")
        self._lbl_confidence.grid(row=4, column=1, sticky="w")

        self._lbl_advice = tk.Label(result_frame, text="", fg="gray", font=("Helvetica", 10, "italic"))
        self._lbl_advice.grid(row=5, column=0, columnspan=2, pady=(4, 0))

        # Footer
        tk.Label(
            self,
            text="AIT102 Group Project · Model: MobileNetV2 · Front-end: Tkinter",
            fg="gray",
            font=("Helvetica", 8),
        ).grid(row=7, column=0, columnspan=2, pady=(4, 8))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_sample_list(self):
        samples = list_sample_images()
        self._sample_combo["values"] = samples
        if samples:
            self._sample_var.set(samples[0])

    def _on_mode_change(self):
        if self._mode.get() == "upload":
            self._sample_frame.grid_remove()
            self._upload_frame.grid()
        else:
            self._upload_frame.grid_remove()
            self._sample_frame.grid()
            samples = self._sample_combo["values"]
            if samples:
                self._load_image(os.path.join(SAMPLE_DIR, self._sample_var.get()))
            else:
                messagebox.showwarning("No samples", "No sample images found in 'sample_images' folder.")

    def _on_sample_selected(self, _event=None):
        path = os.path.join(SAMPLE_DIR, self._sample_var.get())
        self._load_image(path)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select a fruit image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )
        if path:
            self._lbl_file.config(text=os.path.basename(path), fg="black")
            self._load_image(path)

    def _load_image(self, path):
        try:
            img = Image.open(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return

        self._pil_image = img
        self._show_preview(img)
        self._btn_analyse.config(state="normal")
        self._clear_result()

    def _show_preview(self, img: Image.Image):
        preview = img.copy()
        preview.thumbnail((460, 300))
        self._tk_image = ImageTk.PhotoImage(preview)
        self._preview_label.config(image=self._tk_image, text="")

    def _clear_result(self):
        self._lbl_verdict.config(text="", bg=self.cget("bg"))
        self._lbl_fruit.config(text="—")
        self._lbl_state.config(text="—")
        self._confidence_bar["value"] = 0
        self._lbl_confidence.config(text="—")
        self._lbl_advice.config(text="")

    # ------------------------------------------------------------------
    # Analysis (run in a background thread so the UI stays responsive)
    # ------------------------------------------------------------------

    def _analyse(self):
        if self._pil_image is None:
            return

        self._btn_analyse.config(state="disabled", text="Analysing…")
        self._lbl_verdict.config(text="Looking at your fruit…", fg="gray")
        self.update_idletasks()

        thread = threading.Thread(target=self._run_model, daemon=True)
        thread.start()

    def _run_model(self):
        try:
            state, fruit, confidence = predict_image(self._pil_image)
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return
        self.after(0, lambda: self._show_result(state, fruit, confidence))

    def _show_result(self, state: str, fruit: str, confidence: float):
        fruit = fruit.strip().capitalize()
        confidence = round(confidence, 1)

        if state == "Fresh":
            verdict_text = f"✅ This {fruit} looks FRESH"
            verdict_color = "#2e7d32"
            advice = "Good to eat. Enjoy it!"
        else:
            verdict_text = f"🦠 This {fruit} looks ROTTEN"
            verdict_color = "#c62828"
            advice = "Better not eat this one — throw it away."

        self._lbl_verdict.config(text=verdict_text, fg=verdict_color)
        self._lbl_fruit.config(text=fruit)
        self._lbl_state.config(text=state)
        self._confidence_bar["value"] = min(confidence, 100)
        self._lbl_confidence.config(text=f"{confidence}%")
        self._lbl_advice.config(text=advice)

        self._btn_analyse.config(state="normal", text="Analyse 🔍")

    def _show_error(self, exc: Exception):
        messagebox.showerror("Prediction error", str(exc))
        self._btn_analyse.config(state="normal", text="Analyse 🔍")
        self._lbl_verdict.config(text="")


if __name__ == "__main__":
    app = App()
    app.mainloop()
