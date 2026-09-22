# -*- coding: utf-8 -*-
"""
pixelface - Reconocimiento facial y control de acceso, UNA FOTO POR PERSONA
==========================================================================

Aplicación interactiva tipo "Teachable Machine". Cada persona se registra con
UNA sola foto (subida desde el disco o capturada con la cámara web) y un
NOMBRE. Los registros se guardan en una base de datos SQLite local, así que se
conservan al cerrar el programa (sin límite práctico; se probó con 40+).
Con la cámara en vivo, el sistema identifica quién está frente a ella:

    * Rostro registrado    -> malla y recuadro en VERDE, "AUTORIZADO: <nombre>"
                              y el porcentaje de certeza.
    * Rostro no registrado -> malla y recuadro en ROJO,
                              "NO AUTORIZADO / DESCONOCIDO".

PIPELINE (una sola dirección)
-----------------------------
    Foto o fotograma (BGR)
      -> RGB -> Escala de grises -> Filtro de ruido -> CLAHE (contraste)
      -> Mapa de calor de intensidad (visualización)
      -> Malla facial de 468 puntos (MediaPipe FaceLandmarker)
      -> Normalización (centrado, inclinación, escala) y aplanado a un
         vector 1D de 468 x 3 = 1404 características
      -> Huella facial de 128 números (SFace, modelo de OpenCV)
      -> UNA NEURONA con salida sigmoide: 1 = misma persona (AUTORIZADO),
         0 = persona distinta (NO AUTORIZADO)

LA NEURONA (por qué es de "pares")
----------------------------------
Con una sola foto por persona no se puede entrenar un clasificador "una clase
por persona": no hay ejemplos suficientes y habría que reentrenar de cero cada
vez que se registra a alguien. Por eso la neurona VERIFICA PARES: recibe el
rostro que está frente a la cámara junto a un rostro registrado y responde
1 (es la misma persona) o 0 (no lo es):

    x = [ |malla_cámara - malla_registrada| (1404, vector plano de la malla),
          huella_cámara * huella_registrada (128) ]
    P(misma persona) = sigmoide(w . x + b)

Se compara con todos los registrados y gana el de mayor probabilidad; se
autoriza si esa probabilidad supera el UMBRAL (ajustable con + y -) y además la
persona destaca claramente de las demás (`Config.min_gap`): un desconocido
suele parecerse un poco a varios registrados, una persona registrada se parece
mucho a UNA.
Se entrena con pares hechos a partir de las fotos registradas (positivos: la
misma foto con giros, luz, desenfoque y tamaño distintos; negativos: fotos de
personas distintas). Sus pesos parten de un valor inicial razonable y el
entrenamiento solo los ajusta, para que con pocos registros no se vuelva
estricto de más.

NOTA HONESTA: la malla de puntos por sí sola describe la forma del rostro, pero
cambia tanto con la pose y el gesto que no basta para distinguir personas
(se midió). Por eso la neurona recibe también la huella facial; sin ella el
sistema no distinguiría bien a los desconocidos.

INSTALACIÓN
-----------
    pip install opencv-python mediapipe numpy

La primera vez descarga a `pixelface_data/` dos modelos: la malla facial de
MediaPipe (~4 MB) y la huella facial SFace de OpenCV (~37 MB). Si no hay
internet, descárgalos a mano desde `Config.landmarker_url` y
`Config.recognizer_url` y guárdalos allí con el mismo nombre.

USO
---
    python app_teachable_facial.py [--camera 0] [--register-folder CARPETA]

    --register-folder registra en lote todas las fotos de una carpeta; el
    NOMBRE de cada persona sale del nombre del archivo ("Ana Perez.jpg").

TECLAS (con la ventana de video seleccionada)
---------------------------------------------
    c   Registrar: captura una foto con la cámara y pide el nombre.
    u   Registrar: sube una imagen del disco y pide el nombre.
    b   Registrar en lote: elige una carpeta de fotos (nombre = archivo).
    l   Muestra/oculta la lista de personas registradas.
    x   Elimina a una persona registrada (pide el nombre).
    t   Vuelve a entrenar la neurona y muestra época, pérdida y precisión.
    + / -   Sube/baja el umbral de autorización (más o menos estricto).
    p   Muestra/oculta el panel de preprocesamiento (grises, CLAHE, heatmap).
    q / ESC   Salir.

RECOMENDACIONES PARA LA FOTO DE REGISTRO
----------------------------------------
* De frente, sin gafas de sol ni cubrebocas, con buena luz y el rostro bien
  visible (al menos ~60 px de ancho). Una foto de cerca funciona mejor.
* Si alguien no es reconocido en un día distinto, baja el umbral con la tecla
  '-' o registra una foto más parecida a cómo se ve normalmente. Si acepta a
  desconocidos, súbelo con '+'. Cada cambio queda guardado.
* Si dos registros son en realidad la misma persona, la app lo avisa: elimina
  uno con la tecla 'x'. Volver a registrar el MISMO nombre reemplaza su foto.
* El sistema no detecta "vida": una foto de una persona registrada mostrada a
  la cámara también se aceptará. No es un control de acceso de alta seguridad.
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError:  # mensaje amable si falta la librería
    sys.exit("Falta MediaPipe. Instala las dependencias con:\n"
             "    pip install opencv-python mediapipe numpy")


# =============================================================================
# 1. CONFIGURACIÓN CENTRAL
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Todos los parámetros ajustables en un solo lugar."""

    # --- Cámara -------------------------------------------------------------
    camera_index: int = 0
    frame_size: tuple[int, int] = (640, 480)        # (ancho, alto)

    # --- Archivos y modelos descargables -----------------------------------
    data_dir: Path = Path(__file__).resolve().parent / "pixelface_data"
    landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task")
    recognizer_url: str = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx")

    # --- Preprocesamiento ---------------------------------------------------
    blur_kernel: int = 5             # tamaño (impar) del filtro gaussiano
    clahe_clip: float = 2.0          # límite de contraste de CLAHE
    clahe_grid: int = 8              # rejilla de CLAHE (8x8 tiles)
    mesh_on_preprocessed: bool = True  # True: la malla se extrae de la imagen
    #                                    en grises filtrada; False: del RGB.

    # --- Malla facial y huella ---------------------------------------------
    n_landmarks: int = 468           # puntos de la malla topológica
    use_z: bool = True               # True: (X,Y,Z) -> 1404 ; False: (X,Y) -> 936
    embedding_size: int = 128        # tamaño de la huella facial de SFace

    # --- Registro (una foto por persona) ------------------------------------
    image_extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    max_image_side: int = 1280       # las fotos más grandes se reducen
    min_face_px: int = 60            # ancho mínimo del rostro en la foto
    n_variants: int = 8              # variantes sintéticas de la foto (ver abajo)
    thumb_size: int = 96             # miniatura guardada en la base de datos

    # --- Neurona de verificación de pares -----------------------------------
    # Valores iniciales de la neurona: P(misma persona) = sigmoide(pendiente *
    # (coseno - centro)). Con centro 0.40 la probabilidad es 0.5 cuando la
    # similitud de huellas es 0.40 (SFace sugiere ~0.36 para "misma persona").
    prior_slope: float = 14.0
    prior_center: float = 0.40
    mesh_scale: float = 0.05         # escala de la diferencia de mallas
    epochs: int = 200
    lr_scale: float = 1.0            # 1.0 = paso máximo estable
    anchor: float = 5.0              # cuánto se resiste a alejarse de los pesos iniciales
    neg_ratio: int = 3               # pares negativos por cada positivo
    min_persons_to_train: int = 3    # con menos personas se usan los pesos iniciales
    val_fraction: float = 0.2        # personas reservadas para validar (desde 10 registros)
    log_every: int = 20
    seed: int = 42

    # --- Inferencia ---------------------------------------------------------
    threshold: float = 0.50          # P(misma persona) mínima para AUTORIZAR
    threshold_step: float = 0.05
    smoothing: float = 0.5           # 0 = sin suavizado; 0.9 = muy suave
    # Además de superar el umbral, la persona más parecida debe DESTACAR de la
    # siguiente (diferencia de similitud >= min_gap). Un desconocido suele
    # parecerse un poco a varios registrados a la vez; una persona registrada
    # se parece mucho a UNA. Medido: con 30 registrados baja los desconocidos
    # aceptados de ~59 % a ~5 % y solo rechaza ~1 % de los registrados.
    # Ponlo en 0 para desactivarlo.
    min_gap: float = 0.10
    duplicate_cos: float = 0.60      # dos registros con esta similitud son la misma
    #                                  persona: no cuentan como "ambiguos" entre sí
    identify_every: int = 2          # la huella (lo más costoso) se calcula cada N
    #                                  cuadros; en los demás se reutiliza el resultado

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pixelface.db"

    @property
    def landmarker_path(self) -> Path:
        return self.data_dir / "face_landmarker.task"

    @property
    def recognizer_path(self) -> Path:
        return self.data_dir / "face_recognition_sface_2021dec.onnx"

    @property
    def n_mesh_features(self) -> int:
        return self.n_landmarks * (3 if self.use_z else 2)


# =============================================================================
# 2. PIPELINE DE PREPROCESAMIENTO DE IMAGEN
# =============================================================================
@dataclass
class PreprocessResult:
    """Todas las representaciones intermedias de una misma imagen."""
    rgb: np.ndarray        # (H,W,3) uint8, espacio RGB
    gray: np.ndarray       # (H,W)   uint8, escala de grises cruda
    clean: np.ndarray      # (H,W)   uint8, grises + filtro de ruido + CLAHE
    heatmap: np.ndarray    # (H,W,3) uint8 BGR, mapa de calor de intensidad
    mesh_input: np.ndarray  # (H,W,3) uint8 RGB que se entrega a MediaPipe


class Preprocessor:
    """RGB -> grises -> denoising -> normalización de contraste -> heatmap."""

    def __init__(self, cfg: Config) -> None:
        k = cfg.blur_kernel | 1  # fuerza tamaño impar (requisito de OpenCV)
        self._blur_kernel = (k, k)
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        self._mesh_on_preprocessed = cfg.mesh_on_preprocessed

    def process(self, frame_bgr: np.ndarray) -> PreprocessResult:
        # Paso 1: OpenCV entrega BGR; se convierte al espacio RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Paso 2: RGB -> escala de grises (3 canales -> 1 canal de intensidad).
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Paso 3: filtrado suave. El desenfoque gaussiano elimina el ruido del
        # sensor sin destruir los bordes faciales. Se hace ANTES de CLAHE
        # porque CLAHE amplificaría el ruido.
        denoised = cv2.GaussianBlur(gray, self._blur_kernel, 0)

        # Paso 4: CLAHE normaliza el contraste localmente (reduce el efecto de
        # sombras e iluminación desigual).
        clean = self._clahe.apply(denoised)

        # Paso 5: mapa de calor de la intensidad (para visualizar cómo "ve" la
        # intensidad el sistema).
        heatmap = cv2.applyColorMap(clean, cv2.COLORMAP_JET)

        # Paso 6: imagen que consume el detector de malla. MediaPipe exige 3
        # canales, así que la imagen de grises se replica en R, G y B.
        if self._mesh_on_preprocessed:
            mesh_input = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        else:
            mesh_input = rgb
        return PreprocessResult(rgb, gray, clean, heatmap, mesh_input)


# =============================================================================
# 3. MALLA FACIAL, VECTOR 1D Y HUELLA FACIAL
# =============================================================================
# Índices de las esquinas externas de los ojos en la malla de MediaPipe; sirven
# para medir la inclinación (roll) de la cabeza.
_EYE_RIGHT_OUTER = 33
_EYE_LEFT_OUTER = 263


@dataclass
class FaceMesh:
    """Malla de un rostro: X,Y en píxeles y Z en la misma escala que X."""
    points: np.ndarray  # (468, 3) float32

    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding Box (x1, y1, x2, y2) que envuelve todos los puntos."""
        x1, y1 = self.points[:, :2].min(axis=0)
        x2, y2 = self.points[:, :2].max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)


def download_model(url: str, path: Path, what: str) -> None:
    """Descarga un modelo si aún no existe (la primera vez que se usa la app)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pixelface] Descargando modelo de {what} -> {path}")
    partial = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)  # solo aparece completo, nunca a medias
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"No se pudo descargar el modelo de {what} ({exc}).\n"
            f"Descárgalo manualmente desde:\n  {url}\n"
            f"y guárdalo como:\n  {path}") from exc


class FaceMeshExtractor:
    """Envuelve MediaPipe FaceLandmarker (468 puntos por rostro).

    static_images=False -> modo VIDEO (cámara): sigue el rostro entre fotogramas.
    static_images=True  -> modo IMAGE (fotos): cada imagen es independiente.
    """

    def __init__(self, cfg: Config, static_images: bool = False) -> None:
        self._n = cfg.n_landmarks
        self._static = static_images
        download_model(cfg.landmarker_url, cfg.landmarker_path, "malla facial")
        # Se carga el modelo como bytes (no como ruta) para evitar problemas
        # con rutas que tengan tildes o caracteres especiales en Windows.
        mode = vision.RunningMode.IMAGE if static_images else vision.RunningMode.VIDEO
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=cfg.landmarker_path.read_bytes()),
            running_mode=mode,
            num_faces=1)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def extract(self, rgb: np.ndarray) -> Optional[FaceMesh]:
        """Devuelve la malla del rostro detectado o None si no hay rostro."""
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        if self._static:
            result = self._landmarker.detect(image)
        else:
            # En modo VIDEO los timestamps deben ser estrictamente crecientes.
            ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts
            result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        # MediaPipe entrega coordenadas normalizadas [0,1]; se pasan a píxeles
        # para que X e Y conserven la proporción real de la imagen.
        pts = np.array([[p.x * w, p.y * h, p.z * w]
                        for p in result.face_landmarks[0][:self._n]],
                       dtype=np.float32)
        return FaceMesh(pts)

    def close(self) -> None:
        self._landmarker.close()


def mesh_to_features(mesh: FaceMesh, use_z: bool = True) -> np.ndarray:
    """Normaliza la malla y la aplana a un vector 1D de características.

    La normalización hace que el vector dependa de la FORMA del rostro y no de
    dónde está en la imagen, a qué distancia ni cuán inclinada tiene la cabeza:
        1) Traslación: se centra la malla en su centroide.
        2) Roll: se rota en el plano para dejar los ojos en horizontal.
        3) Escala: se divide por el radio RMS para obtener tamaño unitario.
        4) Aplanado: (468,3) -> (1404,) en orden x0,y0,z0,x1,y1,z1,...
    """
    pts = mesh.points.astype(np.float64).copy()

    # 1) Centrado
    pts -= pts.mean(axis=0)

    # 2) Alineación del roll con la recta entre las esquinas externas de ojos
    eye_vec = pts[_EYE_LEFT_OUTER, :2] - pts[_EYE_RIGHT_OUTER, :2]
    angle = np.arctan2(eye_vec[1], eye_vec[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2] @ rot.T

    # 3) Escala unitaria
    scale = np.sqrt((pts ** 2).sum(axis=1).mean())
    pts /= max(scale, 1e-6)

    # 4) Aplanado a un vector 1D
    if not use_z:
        pts = pts[:, :2]
    return pts.reshape(-1).astype(np.float32)


class IdentityVerifier:
    """Huella facial: 128 números que describen la IDENTIDAD de un rostro.

    Es la salida de SFace, una red entrenada específicamente para reconocer
    personas. Dos fotos de la misma persona dan huellas muy parecidas (coseno
    alto) aunque cambien la pose, la luz o la distancia; personas distintas
    dan coseno bajo. La malla de MediaPipe se reutiliza para alinear el
    rostro: sus 5 puntos clave (ojos, nariz, comisuras) le indican a SFace
    cómo recortarlo.
    """

    # Índices de la malla: [esquina ext., esquina int.] de cada ojo, punta de
    # la nariz y comisuras de la boca.
    _EYE_A, _EYE_B = (33, 133), (263, 362)
    _NOSE, _MOUTH_A, _MOUTH_B = 1, 61, 291

    def __init__(self, cfg: Config) -> None:
        download_model(cfg.recognizer_url, cfg.recognizer_path, "huella facial (SFace)")
        self._recognizer = cv2.FaceRecognizerSF.create(str(cfg.recognizer_path), "")

    def embed(self, frame_bgr: np.ndarray, mesh: FaceMesh) -> Optional[np.ndarray]:
        """Huella normalizada (norma 1) del rostro, o None si no se pudo calcular."""
        p = mesh.points[:, :2]
        # SFace espera los puntos ordenados de izquierda a derecha en la imagen.
        eyes = sorted((p[list(self._EYE_A)].mean(axis=0),
                       p[list(self._EYE_B)].mean(axis=0)), key=lambda q: q[0])
        mouth = sorted((p[self._MOUTH_A], p[self._MOUTH_B]), key=lambda q: q[0])
        x1, y1, x2, y2 = mesh.bbox()
        face_row = np.array([[x1, y1, x2 - x1, y2 - y1, *eyes[0], *eyes[1],
                              *p[self._NOSE], *mouth[0], *mouth[1], 1.0]],
                            dtype=np.float32)
        try:
            aligned = self._recognizer.alignCrop(frame_bgr, face_row)
            vec = self._recognizer.feature(aligned).flatten().astype(np.float32)
        except cv2.error:
            return None
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None


# =============================================================================
# 4. BASE DE DATOS DE PERSONAS REGISTRADAS (SQLite, persistente)
# =============================================================================
@dataclass
class Snapshot:
    """Copia en memoria del registro, lista para comparar rápido.

    Cada persona tiene varias filas de referencia: la foto ORIGINAL y sus
    variantes sintéticas. Están ordenadas por persona y la original va primero.
    """
    names: list[str]
    ref_person: np.ndarray      # (N,) índice de persona de cada referencia
    ref_mesh: np.ndarray        # (N, 1404) vector plano de la malla
    ref_emb: np.ndarray         # (N, 128) huella facial
    ref_is_original: np.ndarray  # (N,) bool

    @property
    def n_persons(self) -> int:
        return len(self.names)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE COLLATE NOCASE,
    source  TEXT,
    created TEXT,
    thumb   BLOB
);
CREATE TABLE IF NOT EXISTS refs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    is_original INTEGER NOT NULL,
    mesh        BLOB NOT NULL,
    emb         BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value BLOB
);
"""


def clean_name(name: str) -> str:
    """Nombre sin espacios sobrantes y de longitud razonable."""
    return " ".join(name.split())[:60]


class Registry:
    """Guarda personas, sus vectores y los ajustes en un archivo SQLite."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA foreign_keys = ON")  # para el borrado en cascada
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self.revision = 0  # sube con cada cambio; permite refrescar cachés

    # ---- personas ------------------------------------------------------------
    def add(self, name: str, source: str, thumb_jpeg: bytes,
            refs: list[tuple[np.ndarray, np.ndarray]]) -> bool:
        """Guarda a una persona (reemplaza si el nombre ya existe).
        Devuelve True si reemplazó un registro anterior."""
        name = clean_name(name)
        with self._db:  # una sola transacción: o se guarda todo o nada
            row = self._db.execute(
                "SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
            if row is not None:
                self._db.execute("DELETE FROM persons WHERE id = ?", (row[0],))
            cur = self._db.execute(
                "INSERT INTO persons(name, source, created, thumb) VALUES (?,?,?,?)",
                (name, source, datetime.now().isoformat(timespec="seconds"), thumb_jpeg))
            self._db.executemany(
                "INSERT INTO refs(person_id, is_original, mesh, emb) VALUES (?,?,?,?)",
                [(cur.lastrowid, int(i == 0),
                  mesh.astype(np.float32).tobytes(), emb.astype(np.float32).tobytes())
                 for i, (mesh, emb) in enumerate(refs)])
        self.revision += 1
        return row is not None

    def remove(self, name: str) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM persons WHERE name = ?", (clean_name(name),))
        self.revision += 1
        return cur.rowcount > 0

    def exists(self, name: str) -> bool:
        return self._db.execute("SELECT 1 FROM persons WHERE name = ?",
                                (clean_name(name),)).fetchone() is not None

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    def thumbnails(self) -> list[tuple[str, bytes]]:
        return [(n, t) for n, t in self._db.execute(
            "SELECT name, thumb FROM persons ORDER BY id")]

    def snapshot(self) -> Snapshot:
        rows = self._db.execute(
            "SELECT p.id, p.name, r.mesh, r.emb, r.is_original FROM refs r "
            "JOIN persons p ON p.id = r.person_id "
            "ORDER BY p.id, r.is_original DESC, r.id").fetchall()
        names: list[str] = []
        index: dict[int, int] = {}
        person, mesh, emb, orig = [], [], [], []
        for pid, name, m, e, is_orig in rows:
            if pid not in index:
                index[pid] = len(names)
                names.append(name)
            person.append(index[pid])
            mesh.append(np.frombuffer(m, dtype=np.float32))
            emb.append(np.frombuffer(e, dtype=np.float32))
            orig.append(bool(is_orig))
        if not rows:
            return Snapshot([], np.empty(0, int), np.empty((0, 0), np.float32),
                            np.empty((0, 0), np.float32), np.empty(0, bool))
        return Snapshot(names, np.array(person), np.stack(mesh), np.stack(emb),
                        np.array(orig))

    # ---- ajustes (umbral, pesos de la neurona) -------------------------------
    def get_setting(self, key: str):
        row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_setting(self, key: str, value) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)",
                             (key, value))

    def close(self) -> None:
        self._db.close()


# =============================================================================
# 5. FOTOS: CARGA, VARIANTES SINTÉTICAS Y REGISTRO DE UNA PERSONA
# =============================================================================
def load_image_file(path: Path, max_side: int) -> Optional[np.ndarray]:
    """Lee una imagen como BGR listo para el pipeline, o None si falla."""
    try:
        # np.fromfile + imdecode en lugar de cv2.imread: imread falla en
        # Windows con rutas que tienen tildes, eñes u otros caracteres
        # no ASCII. imdecode además respeta la orientación EXIF del móvil.
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None
    if img is None:
        return None
    side = max(img.shape[:2])
    if side > max_side:  # las fotos enormes hacen lenta la detección
        k = max_side / side
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    # La cámara se muestra en modo espejo (selfie); se voltea la foto para que
    # una foto y la cámara produzcan mallas con la misma orientación.
    return cv2.flip(img, 1)


def augment_photo(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variante sintética de la foto: giro, tamaño, desplazamiento, luz,
    desenfoque y compresión. Imita las diferencias entre la foto de registro
    y cómo se ve la persona frente a la cámara otro día."""
    h, w = img.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.85, 1.1))
    rot[:, 2] += rng.uniform(-0.04, 0.04, size=2) * (w, h)
    out = cv2.warpAffine(img, rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    if rng.random() < 0.5:  # cámara de baja calidad: menos resolución
        k = rng.uniform(0.5, 0.9)
        out = cv2.resize(cv2.resize(out, None, fx=k, fy=k), (w, h))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (int(rng.choice([3, 5])),) * 2, 0)
    if rng.random() < 0.5:  # compresión JPEG
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def face_thumbnail(img: np.ndarray, mesh: FaceMesh, size: int) -> bytes:
    """Miniatura cuadrada del rostro, codificada en JPEG."""
    x1, y1, x2, y2 = mesh.bbox()
    half = int(max(x2 - x1, y2 - y1) * 0.65)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # Se rellena con `half` de borde para que el recorte nunca se salga de la
    # imagen; el centro (cx, cy) queda en (cx + half, cy + half) en la imagen
    # rellena, así que el recorte empieza en (cx, cy).
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
    x0 = int(np.clip(cx, 0, pad.shape[1] - 2 * half))
    y0 = int(np.clip(cy, 0, pad.shape[0] - 2 * half))
    crop = cv2.resize(pad[y0:y0 + 2 * half, x0:x0 + 2 * half], (size, size),
                      interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


@dataclass
class FaceRecord:
    """Resultado de procesar UNA foto para registrarla."""
    mesh: FaceMesh
    pre: PreprocessResult
    thumb: bytes
    refs: list[tuple[np.ndarray, np.ndarray]]   # [(vector_malla, huella)]; [0] = original


class FaceRegistrar:
    """De una foto a los datos que se guardan: vector de malla + huella, tanto
    de la foto original como de sus variantes sintéticas."""

    def __init__(self, cfg: Config, preprocessor: Preprocessor,
                 extractor: FaceMeshExtractor, identity: IdentityVerifier) -> None:
        self._cfg, self._pre = cfg, preprocessor
        self._extractor, self._identity = extractor, identity

    def _describe(self, img: np.ndarray):
        pre = self._pre.process(img)                       # grises / CLAHE / heatmap
        mesh = self._extractor.extract(pre.mesh_input)     # malla de 468 puntos
        if mesh is None:
            return None
        emb = self._identity.embed(img, mesh)              # huella facial
        if emb is None:
            return None
        return pre, mesh, mesh_to_features(mesh, self._cfg.use_z), emb

    def process(self, img: np.ndarray, rng: np.random.Generator) -> Optional[FaceRecord]:
        """None si no hay rostro utilizable en la foto."""
        base = self._describe(img)
        if base is None:
            return None
        pre, mesh, vec, emb = base
        refs = [(vec, emb)]
        for _ in range(self._cfg.n_variants):
            variant = self._describe(augment_photo(img, rng))
            if variant is not None:
                refs.append((variant[2], variant[3]))
        return FaceRecord(mesh, pre, face_thumbnail(img, mesh, self._cfg.thumb_size), refs)


# =============================================================================
# 6. RED NEURONAL: UNA NEURONA SIGMOIDE QUE VERIFICA PARES
# =============================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide numéricamente estable: evita overflow en exp() con |z| grande."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class TrainingHistory:
    """Métricas por época (las de validación quedan vacías sin validación)."""
    loss: list[float] = field(default_factory=list)
    acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


class VerificationNeuron:
    """UNA neurona:  P(misma persona) = sigmoide(w . (x - mu) + b)

    La entrada `x` es un vector plano que combina, para el par (rostro en
    cámara, rostro registrado):
        - |malla_cámara - malla_registrada| / escala   (1404 valores)
        - huella_cámara * huella_registrada * 128       (128 valores)
    Etiqueta 1 = misma persona (AUTORIZADO), 0 = personas distintas.

    Los pesos parten de un valor inicial razonable (`w0`, `b0`): un rostro con
    huella parecida da probabilidad alta y la malla no aporta nada. El
    entrenamiento por descenso de gradiente minimiza la entropía cruzada
    binaria (BCE) más una penalización que la mantiene cerca de `w0`; así, con
    pocos registros la neurona no se vuelve más estricta de lo razonable.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.n_mesh, self.n_emb = cfg.n_mesh_features, cfg.embedding_size
        n = self.n_mesh + self.n_emb
        self.w0 = np.concatenate([np.zeros(self.n_mesh),
                                  np.full(self.n_emb, cfg.prior_slope / self.n_emb)])
        self.b0 = -cfg.prior_slope * cfg.prior_center
        self.w, self.b, self.mu = self.w0.copy(), self.b0, np.zeros(n)
        self.trained = False

    # ---- entrada y salida ----------------------------------------------------
    def pair_features(self, q_mesh: np.ndarray, q_emb: np.ndarray,
                      r_mesh: np.ndarray, r_emb: np.ndarray) -> np.ndarray:
        """Vector plano de entrada para el/los par(es) (consulta, referencia)."""
        f_mesh = np.abs(r_mesh - q_mesh) / self._cfg.mesh_scale
        f_emb = r_emb * q_emb * self.n_emb
        return np.concatenate([f_mesh, f_emb], axis=-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return sigmoid((x - self.mu) @ self.w + self.b)

    # ---- entrenamiento -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray,
            x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            on_epoch: Optional[Callable[[int, TrainingHistory], None]] = None
            ) -> TrainingHistory:
        cfg = self._cfg
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # Paso 1: centrar las entradas con la media del entrenamiento. Se
        # ajusta el sesgo para que la función inicial no cambie.
        self.mu = x.mean(axis=0)
        z = x - self.mu
        self.w, self.b = self.w0.copy(), self.b0 + float(self.w0 @ self.mu)

        # Paso 2: pesos por muestra para que positivos y negativos pesen igual.
        weights = np.ones_like(y)
        for cls in (0.0, 1.0):
            mask = y == cls
            if mask.any():
                weights[mask] = len(y) / (2.0 * mask.sum())

        # Paso 3: paso de aprendizaje estable = 1 / (curvatura máxima de la
        # pérdida), estimada con iteración de potencia sobre z'z / N.
        v = np.random.default_rng(0).normal(size=z.shape[1])
        for _ in range(30):
            v = z.T @ (z @ v) / len(z)
            v /= np.linalg.norm(v) + 1e-12
        lam = float(np.linalg.norm(z.T @ (z @ v) / len(z)))
        lr = cfg.lr_scale / (0.25 * lam + cfg.anchor)

        z_val = None if x_val is None else x_val.astype(np.float64) - self.mu
        hist = TrainingHistory()
        for epoch in range(1, cfg.epochs + 1):
            # -- Forward: probabilidad de "misma persona" para cada par -------
            p = sigmoid(z @ self.w + self.b)

            # -- Métricas (pérdida BCE ponderada y precisión) -----------------
            hist.loss.append(self._bce(y, p, weights))
            hist.acc.append(float(((p >= 0.5) == (y == 1)).mean()))
            if z_val is not None:
                pv = sigmoid(z_val @ self.w + self.b)
                hist.val_loss.append(self._bce(y_val, pv))
                hist.val_acc.append(float(((pv >= 0.5) == (y_val == 1)).mean()))

            # -- Backward: dL/dz = (p - y); más la penalización hacia w0 -----
            err = (p - y) * weights
            grad_w = z.T @ err / len(y) + cfg.anchor * (self.w - self.w0)
            grad_b = err.mean()

            # -- Actualización por descenso de gradiente ---------------------
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            if on_epoch is not None:
                on_epoch(epoch, hist)
        self.trained = True
        return hist

    @staticmethod
    def _bce(y: np.ndarray, p: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.average(loss, weights=weights))

    # ---- persistencia --------------------------------------------------------
    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, w=self.w, b=self.b, mu=self.mu, trained=self.trained)
        return buf.getvalue()

    def load_bytes(self, blob: bytes) -> bool:
        try:
            data = np.load(io.BytesIO(blob))
            if data["w"].shape != self.w.shape:
                return False
            self.w, self.b, self.mu = data["w"], float(data["b"]), data["mu"]
            self.trained = bool(data["trained"])
            return True
        except Exception:
            return False


def build_training_pairs(snap: Snapshot, neuron: VerificationNeuron, cfg: Config,
                         rng: np.random.Generator):
    """Pares (consulta, referencia) etiquetados 1 (misma persona) / 0 (distinta).

    Positivos: una variante de la foto contra la foto original de la MISMA persona.
    Negativos: un rostro de una persona contra el de OTRA persona registrada.
    Desde 10 personas se reserva una parte para validar: sus pares no se usan
    para entrenar, así la validación mide cómo se comporta con personas nuevas.
    Devuelve (X, y, X_val, y_val); los de validación son None si no aplica.
    """
    n = snap.n_persons
    by_person = [np.flatnonzero(snap.ref_person == p) for p in range(n)]
    val_persons: set[int] = set()
    if n >= 10:
        val_persons = set(rng.choice(n, max(2, int(round(n * cfg.val_fraction))),
                                     replace=False).tolist())
    train_persons = [p for p in range(n) if p not in val_persons]

    def positives(persons):
        return [(k, by_person[p][0]) for p in persons for k in by_person[p][1:]]

    def negatives(count, query_persons, ref_persons):
        pairs = []
        while len(pairs) < count:
            i, j = int(rng.choice(query_persons)), int(rng.choice(ref_persons))
            if i != j:
                pairs.append((int(rng.choice(by_person[i])), int(rng.choice(by_person[j]))))
        return pairs

    def to_xy(pos, neg):
        pairs = pos + neg
        q, r = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
        x = neuron.pair_features(snap.ref_mesh[q], snap.ref_emb[q],
                                 snap.ref_mesh[r], snap.ref_emb[r])
        return x.astype(np.float32), np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    pos = positives(train_persons)
    neg = negatives(cfg.neg_ratio * max(len(pos), 1), train_persons, train_persons)
    x, y = to_xy(pos, neg)
    if not val_persons:
        return x, y, None, None
    pos_v = positives(sorted(val_persons))
    neg_v = negatives(cfg.neg_ratio * max(len(pos_v), 1), sorted(val_persons), list(range(n)))
    x_v, y_v = to_xy(pos_v, neg_v)
    return x, y, x_v, y_v


def score_persons(neuron: VerificationNeuron, snap: Snapshot, q_mesh: np.ndarray,
                  q_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada persona registrada: (probabilidad de ser la misma, similitud coseno).

    Se compara el rostro de la cámara SOLO con la foto original de cada
    persona. Las variantes sintéticas sirven para entrenar la neurona, pero no
    para comparar: como referencia no mejoraban el reconocimiento y sí
    aumentaban los falsos aceptados (se midió)."""
    orig = snap.ref_is_original          # una fila por persona, en el mismo orden que `names`
    r_mesh, r_emb = snap.ref_mesh[orig], snap.ref_emb[orig]
    prob = neuron.predict(neuron.pair_features(q_mesh, q_emb, r_mesh, r_emb))
    return prob, r_emb @ q_emb


# =============================================================================
# 7. DIÁLOGOS DEL SISTEMA (nombre, archivo, carpeta)
# =============================================================================
def _tk_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # que aparezca sobre la ventana de video
    root.update()
    return root


def ask_text(title: str, prompt: str, initial: str = "") -> Optional[str]:
    """Pide un texto (p. ej. el nombre). Si no hay tkinter, lo pide por consola."""
    try:
        from tkinter import simpledialog
        root = _tk_root()
        text = simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)
        root.destroy()
    except Exception:
        text = input(f"{prompt} [{initial}]: ").strip() or initial
    return clean_name(text) if text and text.strip() else None


def ask_yes_no(title: str, message: str) -> bool:
    try:
        from tkinter import messagebox
        root = _tk_root()
        answer = messagebox.askyesno(title, message, parent=root)
        root.destroy()
        return bool(answer)
    except Exception:
        return input(f"{message} (s/n): ").strip().lower().startswith("s")


def pick_image_file(title: str, extensions: tuple[str, ...]) -> Optional[Path]:
    try:
        from tkinter import filedialog
        root = _tk_root()
        chosen = filedialog.askopenfilename(
            title=title, parent=root,
            filetypes=[("Imagenes", " ".join(f"*{e}" for e in extensions)), ("Todos", "*.*")])
        root.destroy()
    except Exception:
        chosen = input(f"{title}\nRuta de la imagen (Enter = cancelar): ").strip().strip('"')
    return Path(chosen) if chosen else None


def pick_folder(title: str) -> Optional[Path]:
    try:
        from tkinter import filedialog
        root = _tk_root()
        chosen = filedialog.askdirectory(title=title, parent=root)
        root.destroy()
    except Exception:
        chosen = input(f"{title}\nRuta de la carpeta (Enter = cancelar): ").strip().strip('"')
    return Path(chosen) if chosen else None


# =============================================================================
# 8. DIBUJO / INTERFAZ VISUAL
# =============================================================================
GREEN = (0, 220, 0)      # BGR
RED = (0, 0, 230)
NEUTRAL = (255, 200, 0)  # azul claro: vista previa de un registro
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)


def ascii_text(text: str) -> str:
    """Las fuentes de OpenCV no dibujan tildes ni eñes: se muestran sin ellas
    (el nombre guardado en la base de datos conserva los acentos)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def put_text(img: np.ndarray, text: str, org: tuple[int, int],
             color: tuple[int, int, int] = WHITE, scale: float = 0.55,
             thickness: int = 1) -> None:
    """Texto con contorno negro para que se lea sobre cualquier fondo.

    El contorno se dibuja desplazando 1 px el mismo texto (mismo grosor). No
    se usa un grosor mayor porque en OpenCV 5 el grosor cambia el ancho de las
    letras y el contorno quedaría desalineado del relleno."""
    text = ascii_text(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, text, (x + dx, y + dy), font, scale, (0, 0, 0),
                    thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def text_width(text: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(ascii_text(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


@dataclass
class Verdict:
    """Decisión para el rostro que está frente a la cámara."""
    authorized: bool
    name: Optional[str]     # persona más parecida (None si no hay registrados)
    prob: float             # P(misma persona) suavizada
    cosine: float           # similitud de huellas con esa persona
    threshold: float
    ambiguous: bool = False       # supera el umbral pero no destaca de otra persona
    second: Optional[str] = None  # la otra persona parecida (si es ambiguo)
    gap: float = 1.0              # diferencia de similitud con la siguiente persona


class Renderer:
    """Compone la ventana principal: video + malla + HUD + panel lateral."""

    PANEL_W = 220

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        # Aristas de la teselación facial, como matriz (N, 2) de índices
        conns = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        self._edges = np.array([(c.start, c.end) for c in conns], dtype=np.int32)

    def draw_mesh(self, img: np.ndarray, mesh: FaceMesh,
                  color: tuple[int, int, int]) -> None:
        """Malla semitransparente + Bounding Box del rostro."""
        xy = mesh.points[:, :2].astype(np.int32)
        overlay = img.copy()
        cv2.polylines(overlay, list(xy[self._edges]), False, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        x1, y1, x2, y2 = mesh.bbox()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def draw_hud(self, img: np.ndarray, lines: list[str]) -> None:
        """Banda oscura superior con el estado de la aplicación."""
        band = 22 * len(lines) + 8
        img[:band] = (img[:band] * 0.35).astype(np.uint8)
        for i, line in enumerate(lines):
            put_text(img, line, (8, 20 + 22 * i))

    def draw_footer(self, img: np.ndarray, text: str) -> None:
        h = img.shape[0]
        img[h - 26:] = (img[h - 26:] * 0.35).astype(np.uint8)
        put_text(img, text, (8, h - 8), GRAY, 0.45)

    def draw_verdict(self, img: np.ndarray, mesh: FaceMesh, v: Verdict) -> None:
        """Etiqueta, nombre, certeza y barra de probabilidad sobre el rostro."""
        color = GREEN if v.authorized else RED
        if v.authorized:
            title = f"AUTORIZADO: {v.name}"
            info = (f"Certeza {v.prob * 100:4.1f}% | Similitud {v.cosine:.2f}"
                    f" | umbral {v.threshold * 100:.0f}%")
        else:
            title = "NO AUTORIZADO / DESCONOCIDO"
            if v.name is None:
                info = "Sin personas registradas"
            elif v.ambiguous:
                info = f"Ambiguo: {v.name} o {v.second} (dif. {v.gap:.2f})"
            else:
                info = (f"Mas parecido: {v.name} ({v.prob * 100:.0f}%)"
                        f" | umbral {v.threshold * 100:.0f}%")
        img_h, img_w = img.shape[:2]
        x1, y1, x2, y2 = mesh.bbox()

        # Recuadro de información: se ajusta al texto. Va encima del Bounding
        # Box; si choca con el HUD, va debajo.
        box_w = max(text_width(title, 0.6, 2), text_width(info, 0.5), x2 - x1) + 14
        box_w, box_h = min(box_w, img_w), 58
        left = int(np.clip(x1, 0, max(img_w - box_w, 0)))
        top = y1 - box_h - 4
        if top < 56:
            top = min(y2 + 4, img_h - box_h - 30)
        img[top:top + box_h, left:left + box_w] = (
            img[top:top + box_h, left:left + box_w] * 0.3).astype(np.uint8)

        put_text(img, title, (left + 6, top + 20), color, 0.6, 2)
        put_text(img, info, (left + 6, top + 40), WHITE, 0.5)
        # Barra horizontal: probabilidad; la marca blanca es el umbral.
        bar_w = box_w - 12
        cv2.rectangle(img, (left + 6, top + 46), (left + 6 + bar_w, top + 54), GRAY, 1)
        cv2.rectangle(img, (left + 6, top + 46),
                      (left + 6 + int(bar_w * min(max(v.prob, 0.0), 1.0)), top + 54), color, -1)
        tick = left + 6 + int(bar_w * v.threshold)
        cv2.line(img, (tick, top + 43), (tick, top + 57), WHITE, 2)

    def side_panel(self, pre: PreprocessResult, height: int) -> np.ndarray:
        """Columna con las tres etapas visuales del preprocesamiento."""
        thumb_h = height // 3
        views = [("1) Grises", cv2.cvtColor(pre.gray, cv2.COLOR_GRAY2BGR)),
                 ("2) Filtro + CLAHE", cv2.cvtColor(pre.clean, cv2.COLOR_GRAY2BGR)),
                 ("3) Heatmap", pre.heatmap)]
        tiles = []
        for title, view in views:
            tile = cv2.resize(view, (self.PANEL_W, thumb_h))
            put_text(tile, title, (6, 18), WHITE, 0.5)
            tiles.append(tile)
        panel = np.vstack(tiles)
        if panel.shape[0] < height:  # relleno si la división no es exacta
            pad = np.zeros((height - panel.shape[0], self.PANEL_W, 3), np.uint8)
            panel = np.vstack([panel, pad])
        return panel


def fit_tile(img: np.ndarray, w: int, h: int
             ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Ajusta una imagen a w x h conservando proporción (con bandas negras).
    Devuelve (mosaico, escala, desplazamiento) para poder escalar la malla."""
    scale = min(w / img.shape[1], h / img.shape[0])
    new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
    tile = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - new_w) // 2, (h - new_h) // 2
    tile[oy:oy + new_h, ox:ox + new_w] = cv2.resize(img, (new_w, new_h))
    return tile, scale, (ox, oy)


def compose_preview(renderer: Renderer, header: list[str], frame: Optional[np.ndarray],
                    pre: Optional[PreprocessResult], mesh: Optional[FaceMesh]) -> np.ndarray:
    """Vista de un registro: foto + malla | grises + filtro | heatmap."""
    tw, th = 280, 210
    blank = np.zeros((th, tw, 3), np.uint8)
    if frame is None or pre is None:
        tiles = [blank, blank.copy(), blank.copy()]
    else:
        # Malla dibujada sobre el mosaico (no sobre la foto original, donde
        # las líneas de 1 px desaparecerían al reducir).
        photo, scale, (ox, oy) = fit_tile(frame, tw, th)
        if mesh is not None:
            pts = mesh.points.copy()
            pts[:, :2] = pts[:, :2] * scale + (ox, oy)
            renderer.draw_mesh(photo, FaceMesh(pts), NEUTRAL)
        gray = fit_tile(cv2.cvtColor(pre.clean, cv2.COLOR_GRAY2BGR), tw, th)[0]
        heat = fit_tile(pre.heatmap, tw, th)[0]
        tiles = [photo, gray, heat]
    for tile, title in zip(tiles, ("Malla facial", "Grises + filtro", "Heatmap")):
        put_text(tile, title, (6, 18), WHITE, 0.5)
    top = np.full((30 + 24 * len(header), tw * 3, 3), 28, np.uint8)
    for i, line in enumerate(header):
        put_text(top, line, (10, 22 + 24 * i))
    return np.vstack([top, np.hstack(tiles)])


def render_registry(items: list[tuple[str, bytes]], thumb: int = 96, cols: int = 8,
                    max_items: int = 64) -> np.ndarray:
    """Cuadrícula con la miniatura y el nombre de cada persona registrada."""
    tile_w, tile_h = thumb + 8, thumb + 30
    shown = items[:max_items]
    rows = max(1, -(-len(shown) // cols))
    canvas = np.full((36 + rows * tile_h, cols * tile_w, 3), 28, np.uint8)
    title = f"Personas registradas: {len(items)}"
    if len(items) > max_items:
        title += f" (se muestran {max_items})"
    put_text(canvas, title if items else "Sin personas registradas", (8, 24), WHITE, 0.55)
    for i, (name, jpeg) in enumerate(shown):
        r, c = divmod(i, cols)
        x, y = c * tile_w + 4, 36 + r * tile_h + 2
        face = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if face is not None:
            canvas[y:y + thumb, x:x + thumb] = cv2.resize(face, (thumb, thumb))
        label = ascii_text(name)
        if len(label) > 13:
            label = label[:12] + "."
        put_text(canvas, label, (x, y + thumb + 16), WHITE, 0.4)
    return canvas


def render_training_curves(hist: TrainingHistory, total_epochs: int,
                           size: tuple[int, int] = (760, 340)) -> np.ndarray:
    """Dibuja las curvas de pérdida y precisión en una imagen de OpenCV."""
    w, h = size
    canvas = np.full((h, w, 3), 28, np.uint8)
    n = len(hist.loss)
    if n == 0:
        return canvas

    def panel(x0: int, title: str, series: list[tuple[list[float], tuple]],
              y_max: float) -> None:
        top, bottom, left, right = 60, h - 30, x0 + 45, x0 + w // 2 - 20
        put_text(canvas, title, (left, 40), WHITE, 0.6)
        cv2.rectangle(canvas, (left, top), (right, bottom), (90, 90, 90), 1)
        for frac in (0.0, 0.5, 1.0):  # etiquetas del eje Y
            y = int(bottom - frac * (bottom - top))
            put_text(canvas, f"{frac * y_max:.2f}", (x0 + 4, y + 4), GRAY, 0.4)
        for values, color in series:
            if not values:
                continue
            xs = left + (np.arange(len(values)) / max(total_epochs - 1, 1)) * (right - left)
            ys = bottom - np.clip(np.array(values) / y_max, 0, 1) * (bottom - top)
            cv2.polylines(canvas, [np.stack([xs, ys], 1).astype(np.int32)],
                          False, color, 2, cv2.LINE_AA)

    train_c, val_c = (255, 160, 60), (60, 180, 255)  # BGR
    panel(0, "Perdida (BCE)", [(hist.loss, train_c), (hist.val_loss, val_c)],
          max(max(hist.loss), 1e-6))
    panel(w // 2, "Precision", [(hist.acc, train_c), (hist.val_acc, val_c)], 1.0)

    status = f"Epoca {n}/{total_epochs}  loss={hist.loss[-1]:.4f}  acc={hist.acc[-1] * 100:.1f}%"
    if hist.val_acc:
        status += f"  val_acc={hist.val_acc[-1] * 100:.1f}%"
    put_text(canvas, status, (10, h - 8), WHITE, 0.5)
    put_text(canvas, "azul: entrenamiento", (w - 330, 20), train_c, 0.45)
    put_text(canvas, "naranja: validacion", (w - 170, 20), val_c, 0.45)
    return canvas


# =============================================================================
# 9. APLICACIÓN: REGISTRO (foto + nombre) -> ENTRENAMIENTO -> IDENTIFICACIÓN
# =============================================================================
class PixelFaceApp:
    MAIN_WIN = "pixelface"
    TRAIN_WIN = "pixelface - entrenamiento"
    LIST_WIN = "pixelface - registrados"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.registry = Registry(cfg.db_path)
        self.preprocessor = Preprocessor(cfg)
        self.extractor = FaceMeshExtractor(cfg)                       # video (cámara)
        self.static_extractor = FaceMeshExtractor(cfg, static_images=True)  # fotos
        self.identity = IdentityVerifier(cfg)
        self.registrar = FaceRegistrar(cfg, self.preprocessor, self.static_extractor,
                                       self.identity)
        self.renderer = Renderer(cfg)
        self.rng = np.random.default_rng(cfg.seed)

        # Neurona: pesos guardados en la base de datos (o valores iniciales)
        self.neuron = VerificationNeuron(cfg)
        blob = self.registry.get_setting("neuron")
        if blob is not None and not self.neuron.load_bytes(blob):
            print("[pixelface] Los pesos guardados no son compatibles; se reinician.")
        saved = self.registry.get_setting("threshold")
        self.threshold = float(saved) if saved is not None else cfg.threshold

        # Estado de la interfaz
        self.show_panel = True
        self.show_list = False
        self._snapshot: Optional[Snapshot] = None
        self._snapshot_rev = -1
        self._ema: Optional[np.ndarray] = None    # probabilidades suavizadas por persona
        self._last_verdict: Optional[Verdict] = None
        self._frame_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._toast: tuple[str, float] = ("", 0.0)
        print(f"[pixelface] {self.registry.count()} persona(s) registrada(s) en {cfg.db_path}")

    # ---- utilidades ------------------------------------------------------------
    def _snap(self) -> Snapshot:
        """Copia del registro en memoria, actualizada cuando cambia la base."""
        if self._snapshot_rev != self.registry.revision or self._snapshot is None:
            self._snapshot = self.registry.snapshot()
            self._snapshot_rev = self.registry.revision
            self._ema = None
        return self._snapshot

    def _notify(self, text: str, seconds: float = 3.5) -> None:
        """Aviso temporal sobre el video (y también en consola)."""
        print(f"[pixelface] {text}")
        self._toast = (text, time.monotonic() + seconds)

    # ---- bucle principal ---------------------------------------------------
    def run(self) -> None:
        cap = self._open_camera()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[pixelface] No se pudo leer la cámara.")
                    break
                frame = cv2.resize(cv2.flip(frame, 1), self.cfg.frame_size)
                self._last_frame = frame

                # Etapa 1 y 2: preprocesamiento + malla facial
                pre = self.preprocessor.process(frame)
                mesh = self.extractor.extract(pre.mesh_input)

                # Etapa 3: comparar con los rostros registrados
                verdict = None
                self._frame_count += 1
                if mesh is not None:
                    if (self._last_verdict is None
                            or self._frame_count % self.cfg.identify_every == 0):
                        self._last_verdict = self._identify(frame, mesh)
                    verdict = self._last_verdict
                else:
                    self._ema = self._last_verdict = None  # al perder el rostro se reinicia

                cv2.imshow(self.MAIN_WIN, self._compose(frame, pre, mesh, verdict))
                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if cv2.getWindowProperty(self.MAIN_WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            self.extractor.close()
            self.static_extractor.close()
            self.registry.close()
            cv2.destroyAllWindows()

    def _open_camera(self) -> cv2.VideoCapture:
        # CAP_DSHOW abre la cámara mucho más rápido en Windows.
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.camera_index, backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara {self.cfg.camera_index}. "
                "Prueba con --camera 1 o cierra otras apps que la usen.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_size[1])
        return cap

    # ---- identificación --------------------------------------------------------
    def _identify(self, frame: np.ndarray, mesh: FaceMesh) -> Verdict:
        """Compara el rostro de la cámara con todos los registrados.

        Se AUTORIZA si la persona más parecida (1) supera el umbral de
        probabilidad de la neurona y (2) destaca de las demás al menos
        `min_gap`; si no, el rostro es DESCONOCIDO."""
        snap = self._snap()
        if snap.n_persons == 0:
            return Verdict(False, None, 0.0, 0.0, self.threshold)
        emb = self.identity.embed(frame, mesh)
        if emb is None:
            return Verdict(False, None, 0.0, 0.0, self.threshold)
        prob, cos = score_persons(self.neuron, snap, mesh_to_features(mesh, self.cfg.use_z), emb)

        # Media móvil exponencial por persona: evita parpadeos entre cuadros.
        a = self.cfg.smoothing
        stacked = np.stack([prob, cos])
        self._ema = stacked if self._ema is None else a * self._ema + (1 - a) * stacked
        prob, cos = self._ema
        best = int(np.argmax(prob))
        p = float(prob[best])

        # ¿Destaca de las demás personas? Los registros que son claramente la
        # misma persona que la mejor (duplicados) no cuentan como "otra".
        r_emb = snap.ref_emb[snap.ref_is_original]
        others = np.flatnonzero(r_emb @ r_emb[best] < self.cfg.duplicate_cos)
        second, gap = None, 1.0
        if len(others):
            j = int(others[np.argmax(cos[others])])
            second, gap = snap.names[j], float(cos[best] - cos[j])
        ambiguous = p >= self.threshold and gap < self.cfg.min_gap
        return Verdict(p >= self.threshold and not ambiguous, snap.names[best], p,
                       float(cos[best]), self.threshold, ambiguous, second, gap)

    # ---- registro de personas (una foto por persona) --------------------------
    def register_frame(self, frame: np.ndarray, source: str,
                       name: Optional[str] = None) -> Optional[str]:
        """Registra a UNA persona a partir de UNA imagen BGR (ya en modo espejo).
        Sin `name`, muestra la vista previa y pide el nombre. Devuelve el nombre
        guardado o None si se canceló / la foto no sirve."""
        record = self.registrar.process(frame, self.rng)
        if record is None:
            self._notify("No se detecto un rostro utilizable en la foto")
            return None
        x1, _, x2, _ = record.mesh.bbox()
        if x2 - x1 < self.cfg.min_face_px:
            self._notify(f"Rostro muy pequeno ({x2 - x1}px): usa una foto mas cercana")
            return None

        if name is None:
            # Vista previa del pipeline (foto + malla | grises | heatmap) y nombre.
            cv2.imshow(self.MAIN_WIN, compose_preview(
                self.renderer, ["Registrando una persona: escribe su nombre",
                                f"Rostro detectado ({x2 - x1}px). Fuente: {source}"],
                frame, record.pre, record.mesh))
            cv2.waitKey(1)
            name = ask_text("Registrar persona", "Nombre de la persona:")
            if not name:
                self._notify("Registro cancelado")
                return None
            if self.registry.exists(name) and not ask_yes_no(
                    "Ya existe", f"'{name}' ya esta registrado. Reemplazar su foto?"):
                self._notify("Registro cancelado")
                return None
        replaced = self.registry.add(name, source, record.thumb, record.refs)
        self._after_registry_change()
        self._notify(f"{'Actualizado' if replaced else 'Registrado'}: {name} "
                     f"({self.registry.count()} en total)", seconds=5.0)
        self._report_duplicates(focus=name)
        return name

    def _report_duplicates(self, focus: Optional[str] = None) -> None:
        """Avisa si dos registros parecen ser la MISMA persona con nombres
        distintos (`focus`: solo los que involucran a ese nombre)."""
        snap = self._snap()
        if snap.n_persons < 2:
            return
        r = snap.ref_emb[snap.ref_is_original]
        sims = r @ r.T
        pairs = [(snap.names[i], snap.names[j], float(sims[i, j]))
                 for i in range(snap.n_persons) for j in range(i + 1, snap.n_persons)
                 if sims[i, j] >= self.cfg.duplicate_cos
                 and (focus is None or focus in (snap.names[i], snap.names[j]))]
        for a, b, s in pairs[:10]:
            print(f"[pixelface] AVISO: '{a}' y '{b}' parecen ser la misma persona "
                  f"(similitud {s:.2f}). Si es un duplicado, elimina uno con la tecla x.")
        if pairs:
            self._notify(f"Aviso: '{pairs[0][0]}' y '{pairs[0][1]}' parecen la misma "
                         "persona (ver consola)", seconds=6.0)

    def register_from_camera(self) -> None:
        """Captura UNA foto con la cámara y la registra."""
        if self._last_frame is None:
            return
        self.register_frame(self._last_frame.copy(), "camara")

    def register_from_file(self) -> None:
        """Sube UNA imagen del disco y la registra."""
        path = pick_image_file("Foto de la persona a registrar", self.cfg.image_extensions)
        if path is None:
            return
        frame = load_image_file(path, self.cfg.max_image_side)
        if frame is None:
            self._notify(f"No se pudo leer la imagen: {path.name}")
            return
        self.register_frame(frame, f"archivo:{path.name}")  # el nombre se pide tras la vista previa

    def register_folder(self, folder: Path) -> None:
        """Registra en lote: cada imagen de la carpeta es UNA persona y su
        nombre es el del archivo ('Ana Perez.jpg' -> 'Ana Perez')."""
        if not folder.is_dir():
            self._notify(f"No existe la carpeta: {folder}")
            return
        images = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in self.cfg.image_extensions)
        if not images:
            self._notify("La carpeta no tiene imagenes compatibles")
            return
        print(f"\n[pixelface] Registrando {len(images)} foto(s) de '{folder}'")
        done, skipped = 0, []
        for i, path in enumerate(images, start=1):
            name = clean_name(path.stem.replace("_", " "))
            frame = load_image_file(path, self.cfg.max_image_side)
            record = None if frame is None else self.registrar.process(frame, self.rng)
            ok = record is not None and record.mesh.bbox()[2] - record.mesh.bbox()[0] >= self.cfg.min_face_px
            if ok:
                self.registry.add(name, f"archivo:{path.name}", record.thumb, record.refs)
                done += 1
            else:
                skipped.append(path.name)
            header = [f"Registrando en lote ({i}/{len(images)})   ESC cancela",
                      f"{name}: {'ok' if ok else 'sin rostro utilizable'}   "
                      f"registrados: {done}   omitidos: {len(skipped)}"]
            cv2.imshow(self.MAIN_WIN, compose_preview(
                self.renderer, header, frame, record.pre if record else None,
                record.mesh if record else None))
            if cv2.waitKey(1) & 0xFF == 27:
                print("[pixelface] Registro en lote cancelado")
                break
        for name in skipped[:15]:
            print(f"    sin rostro utilizable: {name}")
        self._after_registry_change()  # un solo reentrenamiento para todo el lote
        self._report_duplicates()
        self._notify(f"Lote: {done} registrada(s), {len(skipped)} omitida(s) "
                     f"({self.registry.count()} en total)", seconds=6.0)

    def register_folder_with_picker(self) -> None:
        folder = pick_folder("Carpeta con UNA foto por persona (nombre = archivo)")
        if folder is not None:
            self.register_folder(folder)

    def remove_person(self) -> None:
        name = ask_text("Eliminar persona", "Nombre exacto de la persona a eliminar:")
        if not name:
            return
        if not self.registry.exists(name):
            self._notify(f"No existe '{name}'")
        elif ask_yes_no("Eliminar", f"Eliminar a '{name}' del registro?"):
            self.registry.remove(name)
            self._after_registry_change()
            self._notify(f"Eliminado: {name} ({self.registry.count()} en total)")

    def _after_registry_change(self) -> None:
        """Tras registrar/eliminar: refresca la memoria y reentrena la neurona."""
        self._snapshot = None
        self._train(verbose=False)
        if self.show_list:
            cv2.imshow(self.LIST_WIN, render_registry(self.registry.thumbnails(),
                                                      self.cfg.thumb_size))

    # ---- entrenamiento ---------------------------------------------------------
    def _train(self, verbose: bool) -> None:
        """Entrena la neurona con pares hechos a partir de las fotos registradas."""
        snap = self._snap()
        cfg = self.cfg
        if snap.n_persons < cfg.min_persons_to_train:
            self.neuron = VerificationNeuron(cfg)  # pesos iniciales
            self.registry.set_setting("neuron", self.neuron.to_bytes())
            if verbose:
                self._notify(f"Se entrena desde {cfg.min_persons_to_train} personas "
                             f"(hay {snap.n_persons}); se usan los pesos iniciales")
            return
        rng = np.random.default_rng(cfg.seed)
        x, y, x_val, y_val = build_training_pairs(snap, self.neuron, cfg, rng)
        if verbose:
            print(f"\n[pixelface] Entrenando la neurona con {len(y)} pares "
                  f"({int(y.sum())} misma persona, {int((1 - y).sum())} distinta), "
                  f"{x.shape[1]} caracteristicas de entrada")
        callback = self._on_epoch if verbose else None
        hist = self.neuron.fit(x, y, x_val, y_val, on_epoch=callback)
        self.registry.set_setting("neuron", self.neuron.to_bytes())
        summary = f"Neurona entrenada: acc={hist.acc[-1] * 100:.1f}%"
        if hist.val_acc:
            summary += f", val_acc={hist.val_acc[-1] * 100:.1f}% (personas no vistas)"
        print(f"[pixelface] {summary}")
        if verbose:
            self._notify(summary)

    def _on_epoch(self, epoch: int, hist: TrainingHistory) -> None:
        """Muestra el avance en consola y en la ventana de curvas."""
        total = self.cfg.epochs
        if epoch == 1 or epoch % self.cfg.log_every == 0 or epoch == total:
            line = (f"  Epoca {epoch:>4}/{total} | loss {hist.loss[-1]:.4f} "
                    f"| acc {hist.acc[-1] * 100:5.1f}%")
            if hist.val_acc:
                line += (f" | val_loss {hist.val_loss[-1]:.4f} "
                         f"| val_acc {hist.val_acc[-1] * 100:5.1f}%")
            print(line)
            cv2.imshow(self.TRAIN_WIN, render_training_curves(hist, total))
            cv2.waitKey(1)  # deja que la ventana se refresque

    # ---- composición de la ventana -----------------------------------------
    def _compose(self, frame: np.ndarray, pre: PreprocessResult,
                 mesh: Optional[FaceMesh], verdict: Optional[Verdict]) -> np.ndarray:
        img = frame.copy()
        if mesh is not None and verdict is not None:
            self.renderer.draw_mesh(img, mesh, GREEN if verdict.authorized else RED)
            self.renderer.draw_verdict(img, mesh, verdict)
        elif mesh is None:
            put_text(img, "SIN ROSTRO", (img.shape[1] // 2 - 60, img.shape[0] // 2),
                     GRAY, 0.8, 2)

        n = self.registry.count()
        hud = [f"pixelface | Registrados: {n} | Umbral: {self.threshold * 100:.0f}%",
               "Registrar: c camara  u archivo  b carpeta" if n else
               "Sin personas: registra con c (camara), u (archivo) o b (carpeta)"]
        self.renderer.draw_hud(img, hud)

        text, until = self._toast
        if time.monotonic() < until:
            put_text(img, text, (8, img.shape[0] - 40), (0, 255, 255), 0.55, 1)
        self.renderer.draw_footer(
            img, "c camara  u archivo  b carpeta  l lista  x eliminar  t entrenar  "
                 "+/- umbral  p panel  q salir")
        if self.show_panel:
            img = np.hstack([img, self.renderer.side_panel(pre, img.shape[0])])
        return img

    # ---- teclado -----------------------------------------------------------
    def _handle_key(self, key: int) -> bool:
        """Procesa una tecla. Devuelve False cuando hay que salir."""
        if key in (ord("q"), 27):
            return False
        if key == ord("c"):
            self.register_from_camera()
        elif key == ord("u"):
            self.register_from_file()
        elif key == ord("b"):
            self.register_folder_with_picker()
        elif key == ord("x"):
            self.remove_person()
        elif key == ord("t"):
            self._train(verbose=True)
        elif key in (ord("+"), ord("=")):
            self._set_threshold(self.threshold + self.cfg.threshold_step)
        elif key in (ord("-"), ord("_")):
            self._set_threshold(self.threshold - self.cfg.threshold_step)
        elif key == ord("p"):
            self.show_panel = not self.show_panel
        elif key == ord("l"):
            self.show_list = not self.show_list
            if self.show_list:
                cv2.imshow(self.LIST_WIN, render_registry(self.registry.thumbnails(),
                                                          self.cfg.thumb_size))
            else:
                cv2.destroyWindow(self.LIST_WIN)
        return True

    def _set_threshold(self, value: float) -> None:
        self.threshold = float(np.clip(value, 0.05, 0.95))
        self.registry.set_setting("threshold", str(self.threshold))
        self._notify(f"Umbral de autorizacion: {self.threshold * 100:.0f}% "
                     f"(menor = mas flexible, mayor = mas estricto)")


# =============================================================================
# 10. PUNTO DE ENTRADA
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="pixelface - reconocimiento facial, una foto por persona")
    parser.add_argument("--camera", type=int, default=Config.camera_index,
                        help="índice de la cámara (por defecto 0)")
    parser.add_argument("--register-folder", type=Path, metavar="CARPETA",
                        help="registra en lote las fotos de una carpeta (nombre = archivo)")
    args = parser.parse_args()

    cfg = Config(camera_index=args.camera)
    try:
        app = PixelFaceApp(cfg)
        if args.register_folder is not None:
            app.register_folder(args.register_folder)
        app.run()
    except RuntimeError as exc:
        sys.exit(f"[pixelface] Error: {exc}")


if __name__ == "__main__":
    main()
