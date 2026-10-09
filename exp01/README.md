# Experiment 01: observation shards, pure rotation

Goal: test whether an image model can extend an existing shard without
repainting it, so that a world built from shards stays consistent.

## Loop

1. Shard 00 is generated from the world bible prompt.
2. `tools/rotate_canvas.py` re-projects a known shard into a camera rotated
   in place (pure yaw, so no depth is needed). Pixels the source never saw,
   plus masked regions such as the generator watermark, are filled flat gray.
3. The canvas goes to the image model with an "only fill the gray" prompt.
4. The returned shard is compared against the canvas in the known region to
   measure how much the model repainted.
5. `tools/compose_shard.py` applies the first-come ownership rule: the canvas
   pixels win wherever they were known, the model's pixels fill only the gray,
   and a feather band inside the known region hides the seam.
6. `tools/make_canvas.py` renders every shard so far at the next yaw to make
   the next canvas, and `tools/render_pan.py` renders a camera pan as video.

## Results so far

| Shard | Yaw | Known in canvas | Model drift in known region |
|---|---|---|---|
| 01 | 30 | 55% | No pixel shift. Colour PSNR 39 dB after blur, 23 dB raw: only fine texture was re-synthesised |

Pan render 0 to 30 to 0 degrees: the first and last frames are bit-identical
before video encoding (max difference 0), so loop closure holds by
construction. Holes peak at 0.32% of a frame, all from the masked watermark.
The CPU renderer takes about 3 s per 1920x1072 frame.

## Assumptions

- 24mm lens on a 36mm-wide sensor, so a horizontal field of view of about 74 degrees.
- Shard 00 is level with no tilt or roll.

## Files

| File | Meaning |
|---|---|
| `shards/shard00.jpg` | Big-bang shard, generated with Flow |
| `shards/shard01_canvas.png` | Shard 00 rotated 30 degrees right, unknown area gray |
| `shards/shard01_known_mask.png` | White where the canvas holds shard 00 pixels |
| `shards/shard01_raw.jpg` | Flow output for the shard 01 canvas |
| `shards/shard00.png`, `shards/shard01.png` | Composed shards, alpha 0 where unseen (watermarks) |
| `shards/shard02_canvas.png` | Shards 00 and 01 rendered at 60 degrees |
| `renders/pan_0_30_0.mp4` | Camera pans 0 to 30 degrees and back |

Rebuild the canvas:

```
python3 tools/rotate_canvas.py shards/shard00.jpg --yaw 30 \
  --invalid '[[2595,1380,2685,1465]]' \
  --out shards/shard01_canvas.png --mask-out shards/shard01_known_mask.png
```
