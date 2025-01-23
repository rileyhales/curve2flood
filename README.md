# Curve2Flood
Curve2Flood is a Python library and CLI tool to create a flood inundation maps (and topobathymetric surface) based on rating curves from the ARC model.


## Installation

Clone or download this repository. Once you've got a local copy, navigate to the curve2flood directory and use pip to install Curve2Flood into your Python environment with the command below.

```bash
pip install .
```

## Usage

Once installed, you can run Curve2Flood as a typical Python library or a command-line utility.

### As a Library

```python
from curve2flood import Curve2Flood_MainFunction

Curve2Flood_MainFunction("path/to/input_file.txt")
```

### Command Line

```bash
curve2flood path/to/input_file.txt
```

### Input File Format

The input file should be a plain text file with key-value pairs, e.g.:

```
DEM_File  path/to/dem.tif
Stream_File  path/to/strm.tif
LU_Raster_SameRes  path/to/land.tif
StrmShp_File  path/to/streams.shp
OutFLD  path/to/output_flood.tif
LAND_WaterValue  80
Q_Fraction  0.5
TopWidthPlausibleLimit  200
TW_MultFact  1.0
Set_Depth  0.1
LocalFloodOption  True
Flood_WaterLC_and_STRM_Cells  False
```

In the near future, we will develop additional documentation on the parameter options in Curve2Flood.
"""
