"""
=============================================================================
 DETECTOR AVANZADO DE EMERGENCIAS MÉDICAS
 Detección de ACV (Ictus) y Desmayos en Tiempo Real
 
 Detecta:
   1. Asimetría Ocular (un ojo cerrado, otro abierto > 5s)
   2. Parálisis Facial / Asimetría de Boca (caída > 15% por > 5s)
   3. Pérdida de Conciencia (inclinación de cabeza > 45° por > 3s)
   4. Ausencia de Parpadeo (sin parpadear > 15s)
 
 OpenCV + MediaPipe Face Landmarker (Tasks API)
=============================================================================
"""

import cv2
import numpy as np
import time
import math
import os
import sys
import urllib.request
import threading
import subprocess
import struct
import wave
import tempfile

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.21
EYE_ASYMMETRY_SECONDS = 5
BOTH_EYES_CLOSED_SECONDS = 5       # Ambos ojos cerrados
MOUTH_ASYMMETRY_THRESHOLD = 0.15   # 15% de diferencia
MOUTH_ASYMMETRY_SECONDS = 5
HEAD_TILT_ANGLE = 45               # Grados
HEAD_TILT_SECONDS = 3
NO_BLINK_WARNING_SECONDS = 10
CAMERA_INDEX = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARKS
# ─────────────────────────────────────────────────────────────────────────────
# Ojos - EAR (6 puntos por ojo)
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Contornos para dibujar
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                    466, 388, 387, 386, 385, 384, 398]

# Boca - landmarks clave
MOUTH_LEFT = 61       # Comisura izquierda
MOUTH_RIGHT = 291     # Comisura derecha
MOUTH_TOP = 13        # Labio superior centro
MOUTH_BOTTOM = 14     # Labio inferior centro
MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Cabeza - para calcular inclinación
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES PREMIUM
# ─────────────────────────────────────────────────────────────────────────────
# Fondo y estructura
COL_BG_DARK = (15, 15, 20)
COL_BG_PANEL = (25, 25, 35)
COL_BORDER = (50, 50, 65)

# Estados
COL_NORMAL = (100, 220, 80)       # Verde
COL_WARNING = (50, 180, 255)      # Naranja/Ámbar
COL_DANGER = (80, 80, 255)        # Rojo suave
COL_EMERGENCY = (0, 0, 255)       # Rojo intenso
COL_INFO = (200, 180, 100)        # Azul claro

# Elementos
COL_EYE_OPEN = (100, 255, 150)
COL_EYE_CLOSED = (100, 100, 240)
COL_MOUTH = (255, 180, 100)
COL_HEAD = (180, 130, 255)
COL_BLINK = (100, 220, 220)
COL_WHITE = (255, 255, 255)
COL_GRAY = (120, 120, 130)
COL_DARK_GRAY = (60, 60, 70)

# Gradientes para barras
GRADIENT_GREEN = [(80, 200, 60), (100, 255, 100)]
GRADIENT_ORANGE = [(30, 150, 230), (50, 200, 255)]
GRADIENT_RED = [(60, 60, 220), (80, 80, 255)]
GRADIENT_PURPLE = [(180, 100, 200), (220, 150, 255)]


# ═══════════════════════════════════════════════════════════════════════════
# SISTEMA DE SONIDO (usa aplay/paplay del sistema, sin sounddevice)
# ═══════════════════════════════════════════════════════════════════════════
def _generate_alarm_wav(filepath):
    """Genera un archivo WAV con un tono de alarma."""
    sample_rate = 22050
    duration = 0.35
    freq = 880
    n_samples = int(sample_rate * duration)
    
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # Envolvente suave
        env = min(t * 30, 1.0) * min((duration - t) * 30, 1.0)
        value = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t) * env)
        samples.append(struct.pack('<h', max(-32768, min(32767, value))))
    
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(samples))


def _find_audio_player():
    """Busca un reproductor de audio disponible en el sistema."""
    for cmd in ['paplay', 'aplay', 'play']:
        try:
            result = subprocess.run(['which', cmd], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except Exception:
            continue
    return None


class AlarmSound:
    """Genera alarma sonora usando comandos del sistema Linux."""
    
    def __init__(self):
        self.playing = False
        self._thread = None
        self._player = _find_audio_player()
        self._wav_path = os.path.join(SCRIPT_DIR, '_alarm_tone.wav')
        
        # Generar archivo WAV de alarma
        if self._player:
            try:
                _generate_alarm_wav(self._wav_path)
                print(f"[OK] Sonido configurado (reproductor: {self._player})")
            except Exception as e:
                print(f"[WARN] No se pudo generar tono: {e}")
                self._player = None
        else:
            print("[WARN] No se encontró reproductor de audio (paplay/aplay).")
            print("       Las alertas serán solo visuales.")
    
    def _play_loop(self):
        """Reproduce el pitido de alarma en loop."""
        while self.playing and self._player:
            try:
                subprocess.run(
                    [self._player, self._wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
                time.sleep(0.1)
            except Exception:
                break
    
    def start(self):
        if self.playing or not self._player:
            return
        self.playing = True
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self.playing = False
    
    def cleanup(self):
        self.stop()
        if os.path.exists(self._wav_path):
            try:
                os.remove(self._wav_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio para un ojo."""
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = coords
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    horiz = euclidean(p1, p4)
    if horiz == 0:
        return 0.0
    return (v1 + v2) / (2.0 * horiz)


def calculate_mouth_asymmetry(landmarks, w, h):
    """
    Calcula la asimetría de la boca comparando la distancia vertical
    de cada comisura respecto al centro de la nariz.
    Retorna (asimetría_ratio, lado_caido).
    """
    nose = landmarks[NOSE_TIP]
    left_corner = landmarks[MOUTH_LEFT]
    right_corner = landmarks[MOUTH_RIGHT]
    
    nose_y = nose.y * h
    left_y = left_corner.y * h
    right_y = right_corner.y * h
    
    # Distancia vertical de cada comisura al centro facial
    dist_left = left_y - nose_y     # Positivo = por debajo de la nariz
    dist_right = right_y - nose_y
    
    if max(abs(dist_left), abs(dist_right)) == 0:
        return 0.0, "ninguno"
    
    # Asimetría como diferencia relativa
    diff = abs(dist_left - dist_right)
    avg = (abs(dist_left) + abs(dist_right)) / 2.0
    
    if avg == 0:
        return 0.0, "ninguno"
    
    ratio = diff / avg
    
    # Determinar qué lado está más caído
    lado = "izquierdo" if dist_left > dist_right else "derecho"
    
    return ratio, lado


def calculate_head_tilt(landmarks, w, h):
    """
    Calcula el ángulo de inclinación (roll) de la cabeza.
    Usa la línea entre las esquinas externas de los ojos.
    Retorna el ángulo en grados (-180 a 180).
    """
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    
    lx, ly = left_eye.x * w, left_eye.y * h
    rx, ry = right_eye.x * w, right_eye.y * h
    
    dx = rx - lx
    dy = ry - ly
    
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def get_points(landmarks, indices, w, h):
    """Extrae puntos (x,y) de landmarks."""
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE DIBUJO
# ═══════════════════════════════════════════════════════════════════════════
def draw_gradient_bar(frame, x, y, width, height, progress, colors_start, colors_end):
    """Dibuja una barra de progreso con gradiente."""
    fill_w = int(progress * width)
    if fill_w <= 0:
        # Fondo vacío
        cv2.rectangle(frame, (x, y), (x + width, y + height), COL_DARK_GRAY, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), COL_BORDER, 1)
        return
    
    # Fondo
    cv2.rectangle(frame, (x, y), (x + width, y + height), COL_DARK_GRAY, -1)
    
    # Relleno con gradiente horizontal
    for i in range(fill_w):
        t = i / max(width, 1)
        color = tuple(int(colors_start[c] + t * (colors_end[c] - colors_start[c]))
                      for c in range(3))
        cv2.line(frame, (x + i, y), (x + i, y + height), color, 1)
    
    # Borde
    cv2.rectangle(frame, (x, y), (x + width, y + height), COL_BORDER, 1)
    
    # Brillo en la parte superior
    if fill_w > 2:
        highlight = tuple(min(255, c + 60) for c in colors_end)
        cv2.line(frame, (x + 1, y + 1), (x + fill_w - 1, y + 1), highlight, 1)


def draw_badge(frame, x, y, text, color, filled=True):
    """Dibuja una etiqueta/badge con estilo."""
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    pad = 6
    bw = text_size[0] + pad * 2
    bh = text_size[1] + pad * 2
    
    if filled:
        # Fondo
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        # Texto
        cv2.putText(frame, text, (x + pad, y + bh - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_WHITE, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 1)
        cv2.putText(frame, text, (x + pad, y + bh - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    return bw


def draw_contour(frame, points, color, thickness=1):
    """Dibuja contorno con puntos."""
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_main_panel(frame, detections, ear_threshold):
    """Dibuja el panel principal de información con diseño premium."""
    h, w = frame.shape[:2]
    panel_h = 170
    
    # Fondo del panel con gradiente vertical
    overlay = frame.copy()
    for row in range(panel_h):
        alpha_line = 0.88 - (row / panel_h) * 0.15
        color = tuple(int(COL_BG_PANEL[c] + row * 0.1) for c in range(3))
        cv2.rectangle(overlay, (0, row), (w, row + 1), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Línea separadora inferior con color de estado
    any_emergency = any(d.get("is_emergency", False) for d in detections.values())
    any_warning = any(d.get("active", False) for d in detections.values())
    
    if any_emergency:
        sep_color = COL_EMERGENCY
    elif any_warning:
        sep_color = COL_WARNING
    else:
        sep_color = COL_NORMAL
    
    cv2.line(frame, (0, panel_h), (w, panel_h), sep_color, 2)
    
    # ── Título ──
    title = "DETECTOR AVANZADO DE EMERGENCIAS MEDICAS"
    cv2.putText(frame, title, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COL_WHITE, 1, cv2.LINE_AA)
    
    # Línea divisoria del título
    cv2.line(frame, (15, 30), (w - 15, 30), COL_BORDER, 1)
    
    # ── Sección de detectores ──
    y_start = 42
    col_w = w // 4  # 4 columnas
    
    det_configs = [
        ("eyes", "OJOS", COL_EYE_OPEN, GRADIENT_GREEN, GRADIENT_ORANGE,
         EYE_ASYMMETRY_SECONDS),
        ("mouth", "BOCA", COL_MOUTH, GRADIENT_GREEN, GRADIENT_ORANGE,
         MOUTH_ASYMMETRY_SECONDS),
        ("head", "CABEZA", COL_HEAD, GRADIENT_GREEN, GRADIENT_PURPLE,
         HEAD_TILT_SECONDS),
        ("blink", "PARPADEO", COL_BLINK, GRADIENT_GREEN, GRADIENT_RED,
         NO_BLINK_WARNING_SECONDS),
    ]
    
    for i, (key, label, color, grad_ok, grad_warn, max_time) in enumerate(det_configs):
        x = 10 + i * col_w
        det = detections.get(key, {})
        active = det.get("active", False)
        is_emerg = det.get("is_emergency", False)
        elapsed = det.get("elapsed", 0.0)
        detail = det.get("detail", "")
        value_str = det.get("value_str", "")
        
        # Nombre del detector
        label_color = COL_EMERGENCY if is_emerg else (COL_WARNING if active else color)
        cv2.putText(frame, label, (x, y_start + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
        
        # Badge de estado
        if is_emerg:
            draw_badge(frame, x, y_start + 18, "ALERTA", COL_EMERGENCY)
        elif active:
            draw_badge(frame, x, y_start + 18, "ACTIVO", COL_WARNING)
        else:
            draw_badge(frame, x, y_start + 18, "OK", COL_NORMAL)
        
        # Valor actual
        if value_str:
            cv2.putText(frame, value_str, (x, y_start + 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_GRAY, 1, cv2.LINE_AA)
        
        # Detalle
        if detail:
            cv2.putText(frame, detail, (x, y_start + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_GRAY, 1, cv2.LINE_AA)
        
        # Barra de progreso del cronómetro
        if active and max_time > 0:
            progress = min(elapsed / max_time, 1.0)
            bar_grad = grad_warn if progress > 0.6 else grad_ok
            draw_gradient_bar(frame, x, y_start + 80, col_w - 25, 10,
                              progress, bar_grad[0], bar_grad[1])
            
            timer_str = f"{elapsed:.1f}s / {max_time}s"
            cv2.putText(frame, timer_str, (x, y_start + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_WARNING, 1, cv2.LINE_AA)
        else:
            # Barra vacía
            draw_gradient_bar(frame, x, y_start + 80, col_w - 25, 10,
                              0.0, grad_ok[0], grad_ok[1])
    
    # ── Línea inferior con umbral ──
    cv2.putText(frame, f"Umbral EAR: {ear_threshold:.2f}",
                (15, panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                COL_DARK_GRAY, 1, cv2.LINE_AA)


def draw_emergency_overlay(frame, alerts):
    """Dibuja overlay de emergencia con alertas activas."""
    h, w = frame.shape[:2]
    blink = int(time.time() * 3) % 2 == 0
    
    # Borde rojo pulsante
    thickness = 14 if blink else 10
    border_color = (0, 0, 255) if blink else (0, 0, 180)
    cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)
    
    # Viñeta roja en esquinas
    overlay = frame.copy()
    for corner_x, corner_y in [(0, 0), (w, 0), (0, h), (w, h)]:
        cv2.circle(overlay, (corner_x, corner_y), 200, (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Banner de emergencia
    num_alerts = len(alerts)
    banner_h = 80 + num_alerts * 40
    banner_y = h // 2 - banner_h // 2
    
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (30, banner_y), (w - 30, banner_y + banner_h),
                  (10, 10, 30), -1)
    cv2.addWeighted(overlay2, 0.9, frame, 0.1, 0, frame)
    
    # Bordes del banner
    cv2.rectangle(frame, (30, banner_y), (w - 30, banner_y + banner_h),
                  border_color, 3)
    
    # Triángulo de alerta
    tri_cx = w // 2
    tri_cy = banner_y + 30
    tri_s = 20
    pts = np.array([
        [tri_cx, tri_cy - tri_s],
        [tri_cx - tri_s, tri_cy + tri_s // 2],
        [tri_cx + tri_s, tri_cy + tri_s // 2]
    ], dtype=np.int32)
    
    if blink:
        cv2.fillPoly(frame, [pts], COL_EMERGENCY)
    else:
        cv2.polylines(frame, [pts], True, COL_EMERGENCY, 2)
    cv2.putText(frame, "!", (tri_cx - 5, tri_cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_WHITE if blink else COL_EMERGENCY,
                2, cv2.LINE_AA)
    
    # Texto EMERGENCIA
    main_text = "EMERGENCIA DETECTADA"
    ts = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    tx = (w - ts[0]) // 2
    ty = banner_y + 65
    
    text_col = COL_EMERGENCY if blink else COL_WHITE
    cv2.putText(frame, main_text, (tx + 1, ty + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, main_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_col, 2, cv2.LINE_AA)
    
    # Lista de alertas activas
    for i, alert_text in enumerate(alerts):
        ay = banner_y + 95 + i * 35
        # Icono de alerta
        cv2.circle(frame, (70, ay - 5), 8, COL_EMERGENCY, -1)
        cv2.putText(frame, "!", (67, ay - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_WHITE, 1, cv2.LINE_AA)
        # Texto
        cv2.putText(frame, alert_text, (90, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_WHITE, 1, cv2.LINE_AA)
    
    # Instrucción inferior
    inst = "BUSQUE ATENCION MEDICA INMEDIATA"
    is_size = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    ix = (w - is_size[0]) // 2
    cv2.putText(frame, inst, (ix, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COL_EMERGENCY if blink else (180, 180, 255), 1, cv2.LINE_AA)


def draw_no_face(frame):
    """Mensaje cuando no hay rostro."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 2 - 200, h // 2 - 40),
                  (w // 2 + 200, h // 2 + 40), COL_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.rectangle(frame, (w // 2 - 200, h // 2 - 40),
                  (w // 2 + 200, h // 2 + 40), COL_BORDER, 1)
    
    text = "Buscando rostro..."
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
    cv2.putText(frame, text, ((w - ts[0]) // 2, h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_GRAY, 1, cv2.LINE_AA)
    
    sub = "Posicione su rostro frente a la camara"
    ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(frame, sub, ((w - ss[0]) // 2, h // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DARK_GRAY, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# DESCARGA DE MODELO
# ═══════════════════════════════════════════════════════════════════════════
def download_model():
    if os.path.exists(MODEL_PATH):
        return True
    print(f"[INFO] Descargando modelo Face Landmarker...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[OK] Modelo descargado ({os.path.getsize(MODEL_PATH)} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo descargar: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DETECTOR AVANZADO DE EMERGENCIAS MÉDICAS")
    print("  ACV (Ictus) y Desmayos")
    print("=" * 60)
    print(f"  Asimetría Ocular:    {EYE_ASYMMETRY_SECONDS}s")
    print(f"  Ojos Cerrados:       {BOTH_EYES_CLOSED_SECONDS}s")
    print(f"  Parálisis Facial:    {MOUTH_ASYMMETRY_SECONDS}s (>{MOUTH_ASYMMETRY_THRESHOLD*100:.0f}%)")
    print(f"  Inclinación Cabeza:  {HEAD_TILT_SECONDS}s (>{HEAD_TILT_ANGLE}°)")
    print(f"  Sin Parpadeo:        {NO_BLINK_WARNING_SECONDS}s")
    print("=" * 60)
    print("  Q: Salir | R: Reiniciar | +/-: Umbral EAR")
    print("=" * 60)
    
    if not download_model():
        return
    
    # ── Inicializar Face Landmarker ──
    print("[INFO] Inicializando Face Landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("[OK] Face Landmarker listo.")
    
    # ── Cámara ──
    print(f"[INFO] Abriendo cámara {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se puede abrir la cámara.")
        landmarker.close()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Cámara: {aw}x{ah}")
    
    # ── Estado ──
    ear_threshold = EAR_THRESHOLD
    
    # Temporizadores
    eye_asym_start = None
    eyes_closed_start = None
    mouth_asym_start = None
    head_tilt_start = None
    last_blink_time = time.time()
    
    # Emergencias activas
    eye_emergency = False
    eyes_closed_emergency = False
    mouth_emergency = False
    head_emergency = False
    blink_warning = False
    
    # Parpadeo
    blink_count = 0
    blink_count_start = time.time()
    bpm = 0
    prev_eyes_closed = False  # Para detectar transición cerrado→abierto
    
    # Referencia de ángulo de cabeza (se calibra al inicio)
    baseline_head_angle = None
    head_angle_samples = []
    calibration_frames = 30  # Primeros 30 frames para calibrar
    frame_number = 0
    
    # Sonido
    alarm = AlarmSound()
    
    # FPS
    fps_count = 0
    fps = 0
    fps_timer = time.time()
    
    window_name = "Detector Avanzado - Emergencias Medicas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    
    print("[INFO] Detector activo.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_number += 1
        
        # FPS
        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_timer = time.time()
        
        # BPM (parpadeos por minuto)
        bpm_elapsed = time.time() - blink_count_start
        if bpm_elapsed >= 60:
            bpm = blink_count
            blink_count = 0
            blink_count_start = time.time()
        elif bpm_elapsed > 5:
            bpm = int(blink_count * (60.0 / bpm_elapsed))
        
        # ── Detección ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        
        try:
            result = landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            continue
        
        # Datos de detección para el panel
        detections = {
            "eyes": {"active": False, "is_emergency": False, "elapsed": 0,
                     "detail": "", "value_str": ""},
            "mouth": {"active": False, "is_emergency": False, "elapsed": 0,
                      "detail": "", "value_str": ""},
            "head": {"active": False, "is_emergency": False, "elapsed": 0,
                     "detail": "", "value_str": ""},
            "blink": {"active": False, "is_emergency": False, "elapsed": 0,
                      "detail": "", "value_str": ""},
        }
        
        if result.face_landmarks and len(result.face_landmarks) > 0:
            lm = result.face_landmarks[0]
            
            # ════════════════════════════════════════════════════════
            # 1. ASIMETRÍA OCULAR
            # ════════════════════════════════════════════════════════
            ear_left = calculate_ear(lm, LEFT_EYE_EAR, w, h)
            ear_right = calculate_ear(lm, RIGHT_EYE_EAR, w, h)
            left_open = ear_left > ear_threshold
            right_open = ear_right > ear_threshold
            
            # Dibujar contornos
            lc = get_points(lm, LEFT_EYE_CONTOUR, w, h)
            rc = get_points(lm, RIGHT_EYE_CONTOUR, w, h)
            l_col = COL_EYE_OPEN if left_open else COL_EYE_CLOSED
            r_col = COL_EYE_OPEN if right_open else COL_EYE_CLOSED
            draw_contour(frame, lc, l_col, 2)
            draw_contour(frame, rc, r_col, 2)
            
            # Etiquetas EAR sobre los ojos
            lc_center = np.mean(lc, axis=0).astype(int)
            rc_center = np.mean(rc, axis=0).astype(int)
            cv2.putText(frame, f"{ear_left:.2f}", (lc_center[0] - 20, lc_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, l_col, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{ear_right:.2f}", (rc_center[0] - 20, rc_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, r_col, 1, cv2.LINE_AA)
            
            detections["eyes"]["value_str"] = f"I:{ear_left:.2f} D:{ear_right:.2f}"
            
            # Lógica de asimetría ocular + ojos cerrados
            if left_open and right_open:
                # Ambos abiertos → Normal
                eye_asym_start = None
                eye_emergency = False
                eyes_closed_start = None
                eyes_closed_emergency = False
                detections["eyes"]["detail"] = "Ambos abiertos"
            elif not left_open and not right_open:
                # Ambos cerrados → Temporizador de ojos cerrados
                eye_asym_start = None
                eye_emergency = False
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                elapsed_closed = time.time() - eyes_closed_start
                detections["eyes"]["active"] = True
                detections["eyes"]["elapsed"] = elapsed_closed
                detections["eyes"]["detail"] = "Ambos cerrados!"
                if elapsed_closed >= BOTH_EYES_CLOSED_SECONDS:
                    eyes_closed_emergency = True
                    detections["eyes"]["is_emergency"] = True
            else:
                # Asimetría: un ojo abierto, otro cerrado
                eyes_closed_start = None
                eyes_closed_emergency = False
                if eye_asym_start is None:
                    eye_asym_start = time.time()
                elapsed = time.time() - eye_asym_start
                detections["eyes"]["active"] = True
                detections["eyes"]["elapsed"] = elapsed
                closed_side = "Izq cerrado" if not left_open else "Der cerrado"
                detections["eyes"]["detail"] = closed_side
                if elapsed >= EYE_ASYMMETRY_SECONDS:
                    eye_emergency = True
                    detections["eyes"]["is_emergency"] = True
            
            # ════════════════════════════════════════════════════════
            # 2. DETECCIÓN DE PARPADEO
            # ════════════════════════════════════════════════════════
            both_closed = not left_open and not right_open
            
            # Detectar transición: cerrado → abierto = 1 parpadeo
            if prev_eyes_closed and not both_closed:
                blink_count += 1
                last_blink_time = time.time()
            prev_eyes_closed = both_closed
            
            time_since_blink = time.time() - last_blink_time
            detections["blink"]["value_str"] = f"BPM: {bpm}"
            detections["blink"]["detail"] = f"Ultimo: {time_since_blink:.0f}s"
            
            if time_since_blink >= NO_BLINK_WARNING_SECONDS:
                blink_warning = True
                detections["blink"]["active"] = True
                detections["blink"]["elapsed"] = time_since_blink
                detections["blink"]["is_emergency"] = True
            else:
                blink_warning = False
            
            # ════════════════════════════════════════════════════════
            # 3. ASIMETRÍA DE BOCA (Parálisis Facial)
            # ════════════════════════════════════════════════════════
            mouth_asym, mouth_side = calculate_mouth_asymmetry(lm, w, h)
            
            # Dibujar contorno de boca
            mouth_pts = get_points(lm, MOUTH_CONTOUR, w, h)
            mouth_color = COL_MOUTH if mouth_asym < MOUTH_ASYMMETRY_THRESHOLD else COL_DANGER
            draw_contour(frame, mouth_pts, mouth_color, 2)
            
            detections["mouth"]["value_str"] = f"Asim: {mouth_asym*100:.1f}%"
            
            if mouth_asym >= MOUTH_ASYMMETRY_THRESHOLD:
                if mouth_asym_start is None:
                    mouth_asym_start = time.time()
                elapsed = time.time() - mouth_asym_start
                detections["mouth"]["active"] = True
                detections["mouth"]["elapsed"] = elapsed
                detections["mouth"]["detail"] = f"Lado {mouth_side} caido"
                if elapsed >= MOUTH_ASYMMETRY_SECONDS:
                    mouth_emergency = True
                    detections["mouth"]["is_emergency"] = True
            else:
                mouth_asym_start = None
                mouth_emergency = False
                detections["mouth"]["detail"] = "Simetrica"
            
            # ════════════════════════════════════════════════════════
            # 4. INCLINACIÓN DE CABEZA
            # ════════════════════════════════════════════════════════
            head_angle = calculate_head_tilt(lm, w, h)
            
            # Calibración inicial
            if frame_number <= calibration_frames:
                head_angle_samples.append(head_angle)
                if frame_number == calibration_frames:
                    baseline_head_angle = np.mean(head_angle_samples)
                    print(f"[OK] Ángulo base calibrado: {baseline_head_angle:.1f}°")
                detections["head"]["detail"] = "Calibrando..."
                detections["head"]["value_str"] = f"Ang: {head_angle:.0f} deg"
            else:
                if baseline_head_angle is not None:
                    angle_diff = abs(head_angle - baseline_head_angle)
                    detections["head"]["value_str"] = f"Incl: {angle_diff:.0f} deg"
                    
                    if angle_diff >= HEAD_TILT_ANGLE:
                        if head_tilt_start is None:
                            head_tilt_start = time.time()
                        elapsed = time.time() - head_tilt_start
                        detections["head"]["active"] = True
                        detections["head"]["elapsed"] = elapsed
                        detections["head"]["detail"] = f"Caida brusca!"
                        if elapsed >= HEAD_TILT_SECONDS:
                            head_emergency = True
                            detections["head"]["is_emergency"] = True
                    else:
                        head_tilt_start = None
                        head_emergency = False
                        detections["head"]["detail"] = "Estable"
            
            # ── Dibujar indicador de inclinación ──
            nose_pt = (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h))
            chin_pt = (int(lm[CHIN].x * w), int(lm[CHIN].y * h))
            cv2.line(frame, nose_pt, chin_pt, COL_HEAD, 1, cv2.LINE_AA)
            cv2.circle(frame, nose_pt, 3, COL_HEAD, -1, cv2.LINE_AA)
        
        else:
            # Sin rostro
            eye_asym_start = None
            eyes_closed_start = None
            mouth_asym_start = None
            head_tilt_start = None
            eye_emergency = False
            eyes_closed_emergency = False
            mouth_emergency = False
            head_emergency = False
            draw_no_face(frame)
        
        # ════════════════════════════════════════════════════════
        # GESTIÓN DE ALARMA SONORA
        # ════════════════════════════════════════════════════════
        any_emergency = (eye_emergency or eyes_closed_emergency or
                        mouth_emergency or head_emergency or blink_warning)
        if any_emergency:
            alarm.start()
        else:
            alarm.stop()
        
        # ════════════════════════════════════════════════════════
        # DIBUJAR UI
        # ════════════════════════════════════════════════════════
        if result.face_landmarks and len(result.face_landmarks) > 0:
            draw_main_panel(frame, detections, ear_threshold)
        
        # Overlay de emergencia
        emergency_alerts = []
        if eye_emergency:
            emergency_alerts.append("Asimetria Ocular Persistente")
        if eyes_closed_emergency:
            emergency_alerts.append("Ojos Cerrados Prolongado")
        if mouth_emergency:
            emergency_alerts.append("Paralisis Facial (Boca caida)")
        if head_emergency:
            emergency_alerts.append("Posible Perdida de Conciencia")
        if blink_warning:
            emergency_alerts.append(f"Sin parpadeo por {time.time()-last_blink_time:.0f}s")
        
        if emergency_alerts:
            draw_emergency_overlay(frame, emergency_alerts)
        
        # FPS y controles
        cv2.putText(frame, f"FPS: {fps}", (w - 90, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DARK_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, "Q:Salir  R:Reiniciar  +/-:Umbral",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    COL_DARK_GRAY, 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, frame)
        
        # ── Teclado ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            eye_asym_start = None
            eyes_closed_start = None
            mouth_asym_start = None
            head_tilt_start = None
            last_blink_time = time.time()
            eye_emergency = False
            eyes_closed_emergency = False
            mouth_emergency = False
            head_emergency = False
            blink_warning = False
            alarm.stop()
            print("[INFO] Detector reiniciado.")
        elif key == ord('+') or key == ord('='):
            ear_threshold = min(ear_threshold + 0.01, 0.40)
            print(f"[INFO] Umbral EAR: {ear_threshold:.3f}")
        elif key == ord('-') or key == ord('_'):
            ear_threshold = max(ear_threshold - 0.01, 0.10)
            print(f"[INFO] Umbral EAR: {ear_threshold:.3f}")
    
    # ── Limpiar ──
    alarm.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] Programa finalizado.")


if __name__ == "__main__":
    main()
