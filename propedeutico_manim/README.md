# Propedeutico ManimGL

Primera coleccion de escenas para explicar ejercicios seleccionados del propedeutico de matematicas. Todo esta escrito como codigo Python reproducible para ManimGL y usa la sintaxis del fork:

```python
from manimlib import *
```

No se incluyen PDFs completos, capturas de libros ni videos renderizados.

## Como correr una escena

Desde la raiz del repositorio:

```bash
poetry run manimgl propedeutico_manim/lunes_factorizacion.py LunesFactorComunSimple
```

## Renderizar imagen final

Usa `-s` para saltar al ultimo frame:

```bash
poetry run manimgl propedeutico_manim/lunes_factorizacion.py LunesFactorComunSimple -s
```

En este fork de ManimGL, `-s` solo abre el visor en modo interactivo. Para escribir el PNG y cerrar sin visor, usa `-s -w`:

```bash
poetry run manimgl propedeutico_manim/lunes_factorizacion.py LunesFactorComunSimple -s -w
```

## Renderizar video

Usa `-w` para escribir el video:

```bash
poetry run manimgl propedeutico_manim/lunes_factorizacion.py LunesFactorComunSimple -w
```

Los videos generados no deben subirse al repositorio. Evita agregar `media/`, `videos/`, `mp4`, `mov`, `mkv` u otros archivos pesados.

## Ejecutar escenas desde menú interactivo

Para elegir escenas sin escribir el nombre completo cada vez:

```bash
poetry run python propedeutico_manim/run_scene_loop.py
```

Al iniciar, el script pregunta el modo de ejecución:

- `1`: interactivo normal.
- `2`: estático / validación, usando `-s -w --video_dir /tmp/manim_propedeutico_validate`.
- `3`: renderizar video, usando `-w`.

Después muestra un menú por día: Lunes, Martes, Miércoles, Jueves, Viernes, Todas las escenas y Salir. Al elegir un día, las escenas aparecen numeradas. Puedes elegir una escena por número, por nombre exacto de clase o por alias con acentos; por ejemplo, `LunesAgrupación` se normaliza y ejecuta `LunesAgrupacion`.

Para salir del menú escribe `q`, `quit`, `salir`, `exit` o `0`.

Dentro de la ventana interactiva de Manim se sale con `q` o `esc`. Cuando la escena termine o cierres esa ventana, el script vuelve al menú principal. No escribas `q` en la terminal después de que ya regresó al shell; ese `q` sólo sirve dentro de la ventana de Manim o dentro del menú del script.

Para listar las escenas declaradas y validar que las clases existen sin ejecutar Manim:

```bash
poetry run python propedeutico_manim/run_scene_loop.py --list
```

Los nombres de clases Python usan ASCII sin acentos, pero el menú acepta aliases con acentos y los normaliza.

## Escenas disponibles

- `propedeutico_manim/lunes_factorizacion.py`
  - `LunesFactorComunSimple`
  - `LunesAgrupacion`
  - `LunesDiferenciaCubos`
  - `LunesFactorComunPolinomio`
  - `LunesAgrupacionConSignos`
  - `LunesCuboPerfectoBinomio`
  - `LunesProblemaProductoSuma`
- `propedeutico_manim/martes_factorizacion_cocientes.py`
  - `MartesTrinomioCuadradoPorAdicion`
  - `MartesTrinomioBasico`
  - `MartesCocienteNotableBasico`
  - `MartesSophieGermain`
  - `MartesTrinomioNoMonico`
  - `MartesTrinomioConSignos`
  - `MartesCocienteCubos`
- `propedeutico_manim/miercoles_potencias_fracciones.py`
  - `MiercolesBinomioPotencias`
  - `MiercolesFraccionCompleja`
  - `MiercolesSumaPotenciasImpares`
  - `MiercolesVariosFactores`
  - `MiercolesMetodoEvaluacion`
  - `MiercolesIndeterminacionAparente`
- `propedeutico_manim/jueves_ecuaciones_lineales.py`
  - `JuevesEcuacionLineal`
  - `JuevesProblemaVerbal`
  - `JuevesEcuacionConParentesis`
  - `JuevesEdadesDobleTriple`
  - `JuevesPadreHijo`
  - `JuevesEnjambre`
- `propedeutico_manim/viernes_ecuaciones_fraccionarias.py`
  - `ViernesEcuacionFraccionaria`
  - `ViernesProblemaFracciones`
  - `ViernesEcuacionDenominadoresAlgebraicos`
  - `ViernesEcuacionLiteralFraccionaria`
  - `ViernesNumeroDisminuido`
  - `ViernesNumerosConsecutivosFracciones`
  - `ViernesProblemaFraccionNumerador`

## Helpers

Los helpers reutilizables viven en `propedeutico_manim/common/algebra_helpers.py` e incluyen:

- titulo de escena;
- bloque de ejercicio;
- bloque de objetivo;
- columna de ecuaciones alineadas;
- resaltado de paso;
- caja de resultado;
- rutina comun para resolver paso a paso.
