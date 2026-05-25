# Long-Term Time Series Forecasting

El objetivo del proyecto es evaluar métodos de pronóstico de series temporales a largo plazo (LTSF). En particular, se compara la literatura reciente sobre modelos lineales simples propuestos en "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2022) contra una implementación de una red LSTM entrenada sobre el dataset ETTh1 para predecir la temperatura del aceite (`OT`).

El notebook `LTSF.ipynb` contiene el análisis exploratorio, las funciones de preprocesamiento, la implementación de una LSTM en PyTorch y la evaluación con métricas estándar (MSE, MAE). Los experimentos incluyen horizontes de predicción H = 96 y H = 336.

## Autores
- [Allison Castro](https://github.com/allisoncastro150-spec)
- [Daniel Suárez](https://github.com/Dass-19)
- [Joselyn Ortega](https://github.com/JoselynOrtegaV)
- [Magno Barco](https://github.com/Rask78)
- [Marlon Salazar](https://github.com/marlonSalazar12345)
- [Paul Delgado](https://github.com/PaulDelgado07)


## Arquitectura del modelo
Este repositorio implementa una red neuronal recurrente Long Short-Term Memory (LSTM) robusta y multicapa utilizando PyTorch, diseñada específicamente para el modelado de secuencias y la predicción de series temporales.

### Características Clave
- Arquitectura Dinámica: Configuración flexible de dimensiones (input_size, hidden_size, num_layers y output_size) para adaptarse fácilmente a diferentes estructuras de datos.

- Inicialización Avanzada de Pesos:

  - Implementa la inicialización Xavier Uniform (xavier_uniform_) tanto en la capa lineal como en los pesos recurrentes de la LSTM para mitigar problemas de desvanecimiento o explosión de gradiente.

  - Los sesgos (biases) se inicializan en cero, a excepción del sesgo de la puerta de olvido (Forget Gate) de la LSTM, que se fuerza explícitamente a 1.0 para mejorar la retención de dependencias a largo plazo al inicio del entrenamiento.

- Flujo de Optimización y Regularización:

  - Integrado con el optimizador AdamW para una regularización eficiente mediante decaimiento de pesos (weight decay).

  - Incluye el planificador ReduceLROnPlateau, el cual reduce dinámicamente la tasa de aprendizaje si la pérdida de validación deja de mejorar, asegurando una convergencia estable.

  - Incorpora capas de Dropout adaptativas para mitigar el sobreajuste (overfitting).

### Diagrama

```plaintext
[ Entrada: (Batch, Seq_Len, Input_Size) ]
                    │
                    ▼
          ┌───────────────────┐
          │     nn.LSTM       │ ◄── Pesos: Xavier Uniform / Forget Gate Bias = 1.0
          └───────────────────┘
                    │ (Batch, Seq_Len, Hidden_Size)
                    ▼
        [ Truncamiento de Secuencia ] ◄── Extrae el último paso de tiempo: [:, -1, :]
                    │
                    ▼
          ┌───────────────────┐
          │    nn.Dropout     │ ◄── Regularización
          └───────────────────┘
                    │ (Batch, Hidden_Size)
                    ▼
          ┌───────────────────┐
          │     nn.Linear     │ ◄── Capa de proyección final
          └───────────────────┘
                    │
                    ▼
[ Predicción Final: (Batch, Output_Size) ] ──► [ nn.MSELoss ] ──► [ AdamW + Scheduler ]
```

### Dimensiones de los Tensores
- Entrada($x$): Tensor 3D con la forma (batch_size, seq_len, input_size).
- Salida ($out$): Tensor 2D con la forma (batch_size, output_size), que contiene las predicciones generadas por el modelo.

## Resultados relevantes
- Para H = 96 (escala estandarizada): MSE ≈ `0.124`, MAE ≈ `0.285`.
- Para H = 336 (escala estandarizada): MSE ≈ `0.227`, MAE ≈ `0.396`.

## Instalación
1. Clonar el repositorio:

```powershell
git clone <tu-repo-url>
cd LTSF
```

2. Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Uso
- Abrir y ejecutar `LTSF.ipynb` en Jupyter Notebook, JupyterLab o VS Code.
- El código de entrenamiento mueve explícitamente tensores y modelo a CUDA (`.cuda()`), por lo que se recomienda disponer de GPU para entrenamientos completos. En entornos sin GPU será necesario editar las llamadas a `.cuda()` para usar CPU.