# TimbreMind

TimbreMind is a hands-on exploration of how audio descriptors can be
translated into visually rich artwork.  The project ships with a
commented Python implementation that analyses ``.wav`` files and renders
Perlin-noise-based flow fields whose colours and motion respond to the
sound's loudness and brightness.

## Features

- **Detailed audio descriptors** – RMS loudness and spectral centroid are
  extracted with clear explanations in the code so the maths stays
  understandable.
- **Perlin noise flow field** – A small custom implementation of Perlin
  noise drives flowing line trajectories reminiscent of particle motion
  in a fluid.
- **Accessible CLI** – Generate artwork with a single command and tune
  parameters such as resolution, grid density, and random seed.
- **Live visualiser** – A Tkinter screen lets you watch the Perlin field
  react in real time while the audio plays.

## Requirements

- Python 3.10+
- ``pip`` for installing dependencies

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
```

The dependencies stay approachable: NumPy for numerical computations,
SciPy for robust ``.wav`` loading, Matplotlib for rendering, and
SimpleAudio for cross-platform playback inside the live viewer.

## Usage

1. Prepare a ``.wav`` file you would like to visualise.
2. Run the generator:

   ```bash
   python -m timbremind.main path/to/audio.wav --output artwork.png
   ```

3. Inspect ``artwork.png`` to see the audio translated into flowing
   lines.

### Live visualiser

Prefer to experience the artwork as it evolves?  Launch the Tkinter
interface:

```bash
python -m timbremind.ui
```

1. Click **Load WAV** and pick the file you want to explore.
2. Press **Start** to play the audio and animate the flow field in real time.
3. Use **Stop** to pause playback and freeze the current field state.

The arrows show the direction and energy of the flow.  Loud moments speed
up the motion, while bright timbres push the colours toward warm hues.

### Command line options

```
python -m timbremind.main --help
```

Key switches include:

- ``--resolution WIDTH HEIGHT`` – change the output image size.
- ``--grid COLUMNS ROWS`` – adjust how many flow lines are seeded.
- ``--seed`` – choose a different random seed to explore new textures.

## How it works

1. **Audio analysis** – ``timbremind/audio.py`` loads the ``.wav`` file,
   normalises it to mono, and splits it into short overlapping windows.
   For each window we compute:

   - **RMS loudness** – approximates perceived volume and smooths
     transients to avoid jittery visuals.
   - **Spectral centroid** – indicates whether the sound is dark or
     bright, which later influences the colour palette.

2. **Perlin noise flow field** – ``timbremind/perlin.py`` implements a
   compact version of Ken Perlin's gradient noise.  By sampling the noise
   at grid points we derive angles that represent the direction of the
   flow at each location.

3. **Visual synthesis** – ``timbremind/visualize.py`` seeds lines across
   the canvas, walks each line by following the flow direction, and uses
   the audio descriptors to modulate:

   - the number of steps taken (energy of the path),
   - the colour hue and opacity, and
   - the line thickness.

   The result is a high-resolution PNG rendered with Matplotlib.

4. **Command line orchestration** – ``timbremind/main.py`` ties everything
   together, providing an ergonomic interface for running the analysis
   and saving the artwork.

## Customising the visuals

- Modify ``FlowFieldConfig`` in ``visualize.py`` to adjust resolution,
  grid density, or step sizes globally.
- Experiment with different Perlin ``scale`` values or angle mapping to
  produce tighter or more relaxed flows.
- Incorporate additional audio descriptors (e.g. spectral roll-off) by
  expanding ``audio.py`` and mapping them to new visual dimensions.

## Example workflow

```bash
python -m timbremind.main demo.wav --output demo.png --resolution 1280 720 --grid 100 60 --seed 42
```

The command above creates a 1280×720 artwork using 100×60 seed points
and a deterministic random seed so the result is repeatable.

## Troubleshooting

- Ensure the input file is a PCM ``.wav``.  Compressed formats such as
  MP3 are not directly supported.
- Very long files can take a few seconds to analyse because the Fourier
  transform scales with window count.  Lower the ``--grid`` values or use
  a shorter excerpt for faster iteration.

## License

MIT License.  See [LICENSE](LICENSE) if present in the repository.
