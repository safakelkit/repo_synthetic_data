# Active GenAI Generation Pipeline

This file is the sole active description of the restarted GenAI pipeline.
Earlier failed pilot records are pending approved cleanup and do not define the
current implementation.

## Method

- SDXL base and Canny ControlNet revisions remain fixed.
- Every candidate is generated as one complete 1024x1024 image in one diffusion
  pass. The pipeline performs no compositing, inpainting, or source-pixel reuse.
- Fifteen classes retain the class-specific silhouette-Canny profiles that
  produced usable outputs in the previous acceptance test.
- Aerosol can uses a concise, text-only full-scene branch. Its ControlNet scale
  is zero; the supplied inert control image is recorded only because the shared
  SDXL ControlNet pipeline requires an image argument.
- Aerosol prompts name concrete retail products (`spray-paint aerosol can` and
  `deodorant body-spray aerosol can`) and request cap/nozzle hardware explicitly.
- All candidate images remain forbidden from training until review, annotation,
  degradation, and final QC pass.

## Acceptance test

The first run contains 64 images: all 16 classes in their four frozen scene
families. Production stays locked unless every class passes at least three of
four samples and the complete test passes at least 56 of 64.

Source files:

- `configs/generation/sdxl_generation_v1.yaml`
- `src/generation/generate_sdxl_dataset.py`

Preflight:

```bash
../env_sam3/bin/python src/generation/generate_sdxl_dataset.py --preflight-only
```

Generation:

```bash
../env_sam3/bin/python src/generation/generate_sdxl_dataset.py
```

The output root is `data/synthetic/sdxl_generation_v1/acceptance_test/`.
