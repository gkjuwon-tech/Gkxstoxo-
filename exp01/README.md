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

## Assumptions

- 24mm lens on a 36mm-wide sensor, so a horizontal field of view of about 74 degrees.
- Shard 00 is level with no tilt or roll.

## Files

| File | Meaning |
|---|---|
| `shards/shard00.jpg` | Big-bang shard, generated with Flow |
| `shards/shard01_canvas.png` | Shard 00 rotated 30 degrees right, unknown area gray |
| `shards/shard01_known_mask.png` | White where the canvas holds shard 00 pixels |

Rebuild the canvas:

```
python3 tools/rotate_canvas.py shards/shard00.jpg --yaw 30 \
  --invalid '[[2595,1380,2685,1465]]' \
  --out shards/shard01_canvas.png --mask-out shards/shard01_known_mask.png
```
