# Experiment 01: observation shards, pure rotation

The world is a set of shards. Each shard is an image the model painted from one
camera. A new shard is painted only where existing shards leave the view empty,
and once born a shard never changes.

## Loop

1. Shard 00 is generated from the world bible prompt.
2. `tools/make_canvas.py` renders every shard so far at the next yaw (pure
   rotation, so no depth is needed). Pixels no shard has seen are flat gray.
   `tools/rotate_canvas.py` does the same from a single source image.
3. The canvas goes to the image model with an "only fill the gray" prompt.
4. The returned image is compared against the canvas in the known region.
5. `tools/compose_shard.py` applies the first-come ownership rule: canvas pixels
   win wherever they were known, the model's pixels fill only the gray, and a
   feather band inside the known region hides the seam.
6. `tools/render_pan.py` renders a camera pan through the shards as video.

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
- Shard 00 is level with no tilt or roll.
