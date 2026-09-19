
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk


APP_TITLE = "Detector de áreas suspeitas de mofo"
LOCAL_SOURCE = "Webcam / câmera USB"
NETWORK_SOURCE = "Câmera Wi-Fi / IP"


def open_camera(source: Union[int, str]) -> cv2.VideoCapture:
    """Abre uma câmera local ou um fluxo de rede."""
    if isinstance(source, int) and os.name == "nt":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return capture


class CameraWorker:
    """Lê a câmera em segundo plano para não travar o Tkinter."""

    def __init__(self, source: Union[int, str]) -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.error: str | None = None
        self.connected = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            self.capture = open_camera(self.source)
            if not self.capture.isOpened():
                self.error = "Não foi possível abrir a câmera selecionada."
                return

            self.connected = True
            failures = 0

            while not self._stop_event.is_set():
                ok, frame = self.capture.read()
                if ok and frame is not None:
                    failures = 0
                    with self._lock:
                        self.frame = frame
                else:
                    failures += 1
                    if failures >= 30:
                        self.error = "A câmera parou de enviar imagens."
                        break
                    time.sleep(0.05)
        except Exception as exc:  # protege a interface de erros do backend
            self.error = f"Erro ao acessar a câmera: {exc}"
        finally:
            self.connected = False
            if self.capture is not None:
                self.capture.release()

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self._stop_event.set()
        if self.capture is not None:
            self.capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class MoldDetector:
    """Localiza manchas suspeitas combinando cor, contraste e textura."""

    @staticmethod
    def analyze(
        frame: np.ndarray,
        sensitivity: float,
        minimum_area: int,
        show_mask: bool,
    ) -> tuple[np.ndarray, int, float]:
        original_height, original_width = frame.shape[:2]

        # Processar em até 960 px mantém a interface fluida.
        scale = min(1.0, 960.0 / original_width)
        if scale < 1.0:
            work = cv2.resize(
                frame,
                (int(original_width * scale), int(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = frame.copy()

        blurred = cv2.GaussianBlur(work, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        hue, saturation, value = cv2.split(hsv)

        # Desvio-padrão local: manchas orgânicas costumam ter textura irregular.
        gray_float = gray.astype(np.float32)
        local_mean = cv2.boxFilter(gray_float, -1, (11, 11), normalize=True)
        local_mean_sq = cv2.boxFilter(
            gray_float * gray_float, -1, (11, 11), normalize=True
        )
        variance = np.maximum(local_mean_sq - local_mean * local_mean, 0)
        texture = cv2.sqrt(variance)

        level = np.clip(sensitivity / 100.0, 0.0, 1.0)
        texture_limit = 18.0 - (12.0 * level)
        dark_limit = int(68 + (50 * level))
        saturation_limit = int(62 - (34 * level))

        # Tons verdes/azulados, marrons/amarelados e manchas muito escuras.
        green = (
            (hue >= 25)
            & (hue <= 100)
            & (saturation >= saturation_limit)
            & (value >= 20)
            & (value <= 235)
        )
        brown = (
            (hue >= 4)
            & (hue <= 28)
            & (saturation >= saturation_limit + 8)
            & (value >= 20)
            & (value <= 190)
        )
        dark = value <= dark_limit
        textured = texture >= texture_limit

        # Textura ao redor ajuda a conservar o interior de manchas escuras.
        texture_mask = (textured.astype(np.uint8) * 255)
        nearby_texture = cv2.dilate(
            texture_mask, np.ones((9, 9), np.uint8), iterations=1
        ) > 0
        candidate = ((green | brown) & textured) | (dark & nearby_texture)
        mask = candidate.astype(np.uint8) * 255

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        annotated = frame.copy()
        accepted: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
        work_area = work.shape[0] * work.shape[1]
        adjusted_minimum_area = max(40, int(minimum_area * scale * scale))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < adjusted_minimum_area or area > work_area * 0.35:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < 12 or height < 12:
                continue

            roi_mask = mask[y : y + height, x : x + width]
            density = cv2.countNonZero(roi_mask) / float(width * height)
            if density < 0.10:
                continue

            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_texture = cv2.mean(texture, mask=contour_mask)[0]
            mean_saturation = cv2.mean(saturation, mask=contour_mask)[0]

            # Pontuação visual, não uma probabilidade médica/laboratorial.
            texture_score = np.clip(
                (mean_texture - texture_limit) / max(1.0, 35.0 - texture_limit),
                0.0,
                1.0,
            )
            color_score = np.clip(mean_saturation / 150.0, 0.0, 1.0)
            density_score = np.clip(density / 0.65, 0.0, 1.0)
            score = int(
                np.clip(
                    35 + 35 * texture_score + 20 * color_score + 10 * density_score,
                    35,
                    99,
                )
            )

            accepted.append((contour, (x, y, width, height), score))
            if len(accepted) == 12:
                break

        if show_mask and cv2.countNonZero(mask) > 0:
            full_mask = cv2.resize(
                mask,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
            red_layer = annotated.copy()
            red_layer[full_mask > 0] = (30, 30, 220)
            annotated = cv2.addWeighted(annotated, 0.78, red_layer, 0.22, 0)

        suspicious_pixels = 0.0
        inverse_scale = 1.0 / scale
        for contour, (x, y, width, height), score in accepted:
            x1 = int(x * inverse_scale)
            y1 = int(y * inverse_scale)
            x2 = int((x + width) * inverse_scale)
            y2 = int((y + height) * inverse_scale)
            suspicious_pixels += cv2.contourArea(contour) / (scale * scale)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Area suspeita - indice {score}/100"
            text_y = max(24, y1 - 8)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        coverage = min(
            100.0,
            100.0 * suspicious_pixels / float(original_width * original_height),
        )
        return annotated, len(accepted), coverage


class MoldDetectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.minsize(960, 650)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        self.worker: CameraWorker | None = None
        self.last_frame: np.ndarray | None = None
        self.last_annotated_frame: np.ndarray | None = None
        self.last_error_shown: str | None = None
        self.frame_counter = 0
        self.last_detection: tuple[int, float] = (0, 0.0)

        self.source_type = tk.StringVar(value=LOCAL_SOURCE)
        self.camera_index = tk.StringVar(value="0")
        self.camera_url = tk.StringVar()
        self.sensitivity = tk.DoubleVar(value=58)
        self.minimum_area = tk.IntVar(value=700)
        self.detection_enabled = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Câmera desligada")
        self.detection_text = tk.StringVar(value="Nenhuma imagem analisada")

        self._configure_style()
        self._build_interface()
        self._toggle_source_inputs()
        self.after(30, self._update_video)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_interface(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(header, text="Detector visual de mofo", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text=(
                "Analisa cores e texturas para indicar pontos que merecem inspeção. "
                "O resultado não confirma a presença de mofo."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        controls = ttk.LabelFrame(root, text="Configuração", padding=12)
        controls.grid(row=1, column=0, sticky="nsw", padx=(0, 12))
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Fonte da imagem").grid(row=0, column=0, sticky="w")
        source_combo = ttk.Combobox(
            controls,
            textvariable=self.source_type,
            values=(LOCAL_SOURCE, NETWORK_SOURCE),
            state="readonly",
            width=28,
        )
        source_combo.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        source_combo.bind("<<ComboboxSelected>>", lambda _event: self._toggle_source_inputs())

        ttk.Label(controls, text="Índice da webcam/câmera USB").grid(
            row=2, column=0, sticky="w"
        )
        local_row = ttk.Frame(controls)
        local_row.grid(row=3, column=0, sticky="ew", pady=(4, 10))
        local_row.columnconfigure(0, weight=1)
        self.camera_combo = ttk.Combobox(
            local_row,
            textvariable=self.camera_index,
            values=("0", "1", "2", "3", "4", "5"),
            state="readonly",
            width=9,
        )
        self.camera_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.scan_button = ttk.Button(
            local_row, text="Procurar", command=self.scan_local_cameras
        )
        self.scan_button.grid(row=0, column=1)

        ttk.Label(controls, text="URL da câmera Wi-Fi/IP").grid(
            row=4, column=0, sticky="w"
        )
        self.url_entry = ttk.Entry(controls, textvariable=self.camera_url, width=31)
        self.url_entry.grid(row=5, column=0, sticky="ew", pady=(4, 2))
        ttk.Label(
            controls,
            text="Ex.: http://192.168.1.20:8080/video\nou rtsp://usuario:senha@IP:554/stream1",
            foreground="#555555",
        ).grid(row=6, column=0, sticky="w", pady=(0, 12))

        ttk.Separator(controls).grid(row=7, column=0, sticky="ew", pady=4)

        ttk.Label(controls, text="Sensibilidade").grid(row=8, column=0, sticky="w")
        ttk.Scale(
            controls,
            variable=self.sensitivity,
            from_=0,
            to=100,
            orient="horizontal",
        ).grid(row=9, column=0, sticky="ew", pady=(3, 9))

        ttk.Label(controls, text="Área mínima da mancha (pixels)").grid(
            row=10, column=0, sticky="w"
        )
        ttk.Spinbox(
            controls,
            textvariable=self.minimum_area,
            from_=100,
            to=50000,
            increment=100,
            width=12,
        ).grid(row=11, column=0, sticky="w", pady=(4, 10))

        ttk.Checkbutton(
            controls,
            text="Ativar análise de manchas",
            variable=self.detection_enabled,
        ).grid(row=12, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            controls,
            text="Destacar máscara suspeita",
            variable=self.show_mask,
        ).grid(row=13, column=0, sticky="w", pady=2)

        button_row = ttk.Frame(controls)
        button_row.grid(row=14, column=0, sticky="ew", pady=(14, 6))
        button_row.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            button_row,
            text="Iniciar câmera",
            style="Accent.TButton",
            command=self.start_camera,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(
            button_row, text="Parar", command=self.stop_camera, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Button(
            controls,
            text="Salvar imagem analisada",
            command=self.save_snapshot,
        ).grid(row=15, column=0, sticky="ew", pady=(3, 12))

        ttk.Label(controls, textvariable=self.status_text, style="Status.TLabel").grid(
            row=16, column=0, sticky="w", pady=(4, 2)
        )
        ttk.Label(
            controls,
            textvariable=self.detection_text,
            wraplength=260,
        ).grid(row=17, column=0, sticky="w")

        video_frame = ttk.LabelFrame(root, text="Imagem ao vivo", padding=8)
        video_frame.grid(row=1, column=1, sticky="nsew")
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_frame,
            text="Selecione a fonte e clique em “Iniciar câmera”",
            bg="#17191c",
            fg="#e8e8e8",
            font=("Segoe UI", 13),
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            root,
            text=(
                "Dica: use boa iluminação, mantenha a câmera firme e ajuste a "
                "sensibilidade para reduzir sombras e falsos alertas."
            ),
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _toggle_source_inputs(self) -> None:
        local_selected = self.source_type.get() == LOCAL_SOURCE
        self.camera_combo.configure(state="readonly" if local_selected else "disabled")
        self.scan_button.configure(state="normal" if local_selected else "disabled")
        self.url_entry.configure(state="disabled" if local_selected else "normal")

    def _selected_source(self) -> Union[int, str] | None:
        if self.source_type.get() == LOCAL_SOURCE:
            try:
                return int(self.camera_index.get())
            except ValueError:
                messagebox.showerror("Fonte inválida", "Selecione um índice de câmera.")
                return None

        url = self.camera_url.get().strip()
        if not url:
            messagebox.showerror(
                "URL necessária",
                "Informe a URL HTTP ou RTSP fornecida pela câmera Wi-Fi/IP.",
            )
            return None
        if not url.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
            messagebox.showerror(
                "URL inválida", "A URL deve começar com http://, https://, rtsp:// ou rtmp://."
            )
            return None
        return url

    def start_camera(self) -> None:
        source = self._selected_source()
        if source is None:
            return

        self.stop_camera(clear_panel=False)
        self.last_error_shown = None
        self.status_text.set("Conectando...")
        self.detection_text.set("Aguardando a primeira imagem")
        self.worker = CameraWorker(source)
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def stop_camera(self, clear_panel: bool = True) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.stop()

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_text.set("Câmera desligada")
        if clear_panel:
            self.video_label.configure(
                image="", text="Selecione a fonte e clique em “Iniciar câmera”"
            )
            self.video_label.image = None

    def scan_local_cameras(self) -> None:
        if self.worker is not None:
            messagebox.showinfo(
                "Câmera em uso", "Pare a câmera atual antes de procurar outras câmeras."
            )
            return

        self.scan_button.configure(state="disabled")
        self.status_text.set("Procurando câmeras locais...")

        def scan() -> None:
            available: list[str] = []
            for index in range(6):
                capture = open_camera(index)
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        available.append(str(index))
                capture.release()
            self.after(0, lambda: self._finish_camera_scan(available))

        threading.Thread(target=scan, daemon=True).start()

    def _finish_camera_scan(self, available: list[str]) -> None:
        self.scan_button.configure(state="normal")
        if available:
            self.camera_combo.configure(values=available)
            self.camera_index.set(available[0])
            self.status_text.set(f"Câmeras encontradas: {', '.join(available)}")
        else:
            self.status_text.set("Nenhuma câmera local encontrada")
            messagebox.showwarning(
                "Nenhuma câmera",
                "Verifique o cabo, a permissão de câmera do Windows e se outro programa está usando o dispositivo.",
            )

    def _update_video(self) -> None:
        worker = self.worker
        if worker is not None:
            if worker.error:
                self.status_text.set("Falha na câmera")
                if self.last_error_shown != worker.error:
                    self.last_error_shown = worker.error
                    messagebox.showerror("Erro de câmera", worker.error)
                    self.stop_camera()
            else:
                frame = worker.latest_frame()
                if frame is not None:
                    self.last_frame = frame
                    self.frame_counter += 1

                    if self.detection_enabled.get():
                        try:
                            minimum_area = max(100, int(self.minimum_area.get()))
                        except (ValueError, tk.TclError):
                            minimum_area = 700

                        annotated, count, coverage = MoldDetector.analyze(
                            frame,
                            float(self.sensitivity.get()),
                            minimum_area,
                            self.show_mask.get(),
                        )
                        self.last_detection = (count, coverage)
                        if count:
                            self.detection_text.set(
                                f"{count} área(s) suspeita(s) • cobertura aproximada: {coverage:.2f}%"
                            )
                        else:
                            self.detection_text.set("Nenhuma área suspeita neste quadro")
                    else:
                        annotated = frame
                        self.last_detection = (0, 0.0)
                        self.detection_text.set("Análise desativada")

                    self.last_annotated_frame = annotated
                    self.status_text.set("Câmera conectada")
                    self._display_frame(annotated)

        self.after(30, self._update_video)

    def _display_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        panel_width = max(320, self.video_label.winfo_width())
        panel_height = max(240, self.video_label.winfo_height())
        image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (panel_width, panel_height), "#17191c")
        position = (
            (panel_width - image.width) // 2,
            (panel_height - image.height) // 2,
        )
        canvas.paste(image, position)

        photo = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def save_snapshot(self) -> None:
        if self.last_annotated_frame is None:
            messagebox.showinfo(
                "Sem imagem", "Inicie uma câmera antes de salvar uma captura."
            )
            return

        output_directory = Path(__file__).resolve().parent / "capturas"
        output_directory.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_directory / f"analise_mofo_{timestamp}.jpg"

        if cv2.imwrite(str(output_path), self.last_annotated_frame):
            messagebox.showinfo(
                "Imagem salva", f"A captura foi salva em:\n{output_path}"
            )
        else:
            messagebox.showerror("Erro", "Não foi possível salvar a imagem.")

    def close_app(self) -> None:
        self.stop_camera(clear_panel=False)
        self.destroy()


if __name__ == "__main__":
    MoldDetectorApp().mainloop()
