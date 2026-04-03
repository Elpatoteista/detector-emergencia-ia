# Detector de Emergencias Médicas Súbitas 🚨

Sistema de detección en tiempo real de **asimetría facial** como indicador temprano de:
- Parálisis facial (posible ACV/Stroke)
- Pérdida de conciencia parcial

## Tecnologías
- **OpenCV** - Captura y procesamiento de video
- **MediaPipe Face Mesh** - Detección de 478 landmarks faciales
- **NumPy** - Cálculos numéricos

## Cómo funciona

### Eye Aspect Ratio (EAR)
El sistema calcula el **EAR** de cada ojo de forma independiente:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

- **EAR alto (~0.3)**: Ojo abierto
- **EAR bajo (~0.1)**: Ojo cerrado

### Lógica de detección
| Ojo Izquierdo | Ojo Derecho | Acción |
|:---:|:---:|:---|
| ✅ Abierto | ✅ Abierto | Estado Normal |
| ❌ Cerrado | ❌ Cerrado | Ignorar (parpadeo/sueño) |
| ✅ Abierto | ❌ Cerrado | ⏱️ Iniciar cronómetro |
| ❌ Cerrado | ✅ Abierto | ⏱️ Iniciar cronómetro |

Si la **asimetría persiste por más de 5 segundos**, se activa la alerta de **EMERGENCIA**.

## Instalación

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
python detector_emergencia.py
```

### Controles de teclado
| Tecla | Acción |
|:---:|:---|
| `Q` | Salir del programa |
| `R` | Reiniciar el detector |
| `+` | Aumentar umbral EAR (+0.01) |
| `-` | Disminuir umbral EAR (-0.01) |

## Configuración

En el archivo `detector_emergencia.py` puedes ajustar:

```python
EAR_THRESHOLD = 0.21          # Umbral para ojo "cerrado"
ASYMMETRY_ALERT_SECONDS = 5   # Segundos para activar emergencia
CAMERA_INDEX = 0              # Índice de cámara
```

## ⚠️ Aviso

Este es un **prototipo de investigación**. No está diseñado para uso médico real ni como sustituto de diagnóstico profesional.
