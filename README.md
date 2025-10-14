# SMPL Painting and Charting Tools

Python library for SMPL Model Texture Painting.

## Directory Overview

- `smplpact.py` - rendering library.
- `smpl_paint_demo.py` - demo script showcasing `smplpact` capabilities.

## Running the demo

1. Install dependencies:
    - opencv-python
    - torch
    - trimesh
    - pyrender
    - smplx
    - chumpy
    - rtree

2. Download demo data from [[here]](https://stevens0-my.sharepoint.com/:u:/g/personal/jdibenes_stevens_edu/EVadG40Vl55LrrBOOEgR3QwBMa2V1Yj8xon_6uCwUPVs2Q?e=dwn0fd) (~1.51 MB).

3. Extract the `smpl_paint_demo` folder from the demo data zip file.

4. Copy the SMPL model files (`SMPL_NEUTRAL.pkl`, etc.) into `smpl_paint_demo/data/smpl`. Alternatively, you can edit the paths in the `smpl_paint_demo.py` script. You can download the SMPL models from [CameraHMR](https://camerahmr.is.tue.mpg.de/index.html).

5. Run the `smpl_paint_demo.py` script.

See `smpl_paint_demo.py` for keyboard controls and other details.

## References

- SMPL Models and UV Map: [SMPL](https://smpl.is.tue.mpg.de/) | [CameraHMR](https://camerahmr.is.tue.mpg.de/index.html)
- Textures: [SMPL_texture_samples](https://github.com/Meshcapade/SMPL_texture_samples)
