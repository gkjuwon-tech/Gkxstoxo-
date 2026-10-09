# Observation-shard pipeline, pure rotation

The world is a set of shards. Each shard is an image the model painted from one
camera. A new shard is painted only where existing shards leave the view empty,
and once born a shard never changes.

## Loop

1. Shard 00 is generated from the world bible prompt.
2. `make_canvas.py` renders every shard so far at the next yaw (pure
   rotation, so no depth is needed). Pixels no shard has seen, and boxes passed
   with `--invalid` such as a generator watermark, are flat gray.
3. The canvas goes to the image model with an "only fill the gray" prompt.
4. The returned image is compared against the canvas in the known region.
5. `compose_shard.py` applies the first-come ownership rule: canvas pixels
   win wherever they were known, the model's pixels fill only the gray, and a
   feather band inside the known region hides the seam.
6. `render_pan.py` renders a pan on the CPU. `gpu_render.py` does the same on
   a GPU in real time, and `kaggle_run.py` runs it on a Kaggle T4.

## Camera pose

Each shard has a yaw and a pitch. The camera pans about the world vertical and
tilts about its own horizontal axis, so a shard generated looking slightly up
keeps a level horizon all the way round. Estimate the pitch of shard 00 from
where the far flat ground meets the sky: pitch = atan((cy - y_horizon) / f),
with f the focal length in pixels.

## GPU bench on a Kaggle T4, 1080p, 12 shards

| Version | Render fps | With download to CPU |
|---|---|---|
| Sample every shard | 16 | 15 |
| Sample only shards in view | 37 | 35 |

Culled and full renders match exactly, and a 360 degree pan closes with zero
difference. Stats are in `bench/`.

## Rules learned

- **Ownership.** The model keeps layout and colour of the known region but
  re-draws fine texture. Always paste the known region back.
- **Boundary rule.** Anything the canvas cuts at the gray boundary (rugs, floor,
  furniture, walls) must be named in the prompt as continuing into the gray.
  Never ask for new walls or furniture where an existing object runs into the gray.
- **Structure check.** The model may rebuild known structure near the boundary,
  for example turning a wall into an opening. Ownership then pastes the old wall
  back against the new opening and the world contradicts itself. Measure drift
  per region of the known area and reject a shard whose structure changed.
- **No repair.** A world whose geometry has contradicted itself is discarded and
  started over, not patched.
- **Resolution.** Accepted shards are upscaled in Flow before upload.

## Assumptions

- 24mm lens on a 36mm-wide sensor, so a horizontal field of view of about 74 degrees.
- No roll.
