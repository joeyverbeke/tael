
# T.A.E.L.
A recursively degrading system, eating its own output as input, flattening the human stories it consumes.

## Prerequisities

- Python 3.10
- CUDA 11.8

## Setup

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## Run

```bash
python controller.py
```

## Structure

### controller.py
High-level controller script to run main.py as a separate subprocess for each urban legend

### main.py
Main script, capture image -> pass to model -> send transcript as osc -> wait for osc message for next iteration

### camera.py
Handles camera input

### model_llava.py
Handles visual understanding task on image, using LLaVA NeXT (1.6), to transcribe image

### model.py
Alternative to llava, uses phi 3.5

### urban_legends.py
Array of urban legends as strings

### utils.py
Assists with catching unwanted outputs from model

### tael.toe
TD file to handle visual output of transcript