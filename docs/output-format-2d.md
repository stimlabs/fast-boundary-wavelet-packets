# 2-D Wavelet Packet Transform: Output Format

## Output shape

```
input:  [..., H, W]
output: [..., max_level, H, W]
```

The forward transform inserts a new **scale** dimension immediately before the
two spatial dimensions.  The spatial dimensions shift one position to the right.

| Input shape     | `dims`     | Output shape               |
|-----------------|------------|----------------------------|
| `[H, W]`        | `(-2, -1)` | `[max_level, H, W]`        |
| `[batch, H, W]` | `(-2, -1)` | `[batch, max_level, H, W]` |
| `[B, C, H, W]`  | `(-2, -1)` | `[B, C, max_level, H, W]`  |

`max_level` is auto-computed as the minimum feasible level across both spatial
dimensions.  For dimensions `H` and `W` with filter length `F`, the maximum
level is the largest `L` such that `H % 2^L == 0`, `W % 2^L == 0`,
`H / 2^(L-1) >= F`, and `W / 2^(L-1) >= F`.

## What each level contains

Each slice along the scale dimension is a **complete tiling** of the input's
spatial extent into a 2-D grid of subbands.

```
output[..., 0, :, :]   →  level 1:  2×2 grid   =   4 subbands, each H/2 × W/2
output[..., 1, :, :]   →  level 2:  4×4 grid   =  16 subbands, each H/4 × W/4
output[..., 2, :, :]   →  level 3:  8×8 grid   =  64 subbands, each H/8 × W/8
...
output[..., L-1, :, :] →  level L:  2^L × 2^L  grid, each H/2^L × W/2^L
```

At every level, the subbands tile the full `H x W` space with no gaps or
overlaps.  Each level is self-contained: it represents the full image, just
decomposed into different numbers of spatial-frequency bands.

## How it works: separable filtering

The 2-D transform is **separable** — it applies two independent 1-D wavelet
analyses per level, one along each spatial axis:

1. **Along W (width/columns):** each row is split into a lowpass and a
   highpass half.
2. **Along H (height/rows):** each column is split into a lowpass and a
   highpass half.

These two 1-D splits combine to produce four quadrants per parent subband.
At each subsequent level, every existing subband is split the same way.

## Subband naming: A, H, V, D

At each decomposition step, every subband splits into four children named by
the 2-D quadrant they occupy:

```
              W-axis lowpass     W-axis highpass
            ┌────────────────┬────────────────┐
H-axis      │                │                │
lowpass     │   A (approx.)  │  V (vert.)     │
            ├────────────────┼────────────────┤
H-axis      │                │                │
highpass    │   H (horiz.)   │  D (diag.)     │
            └────────────────┴────────────────┘
```

- **A** — Approximation: smooth in both directions.
- **H** — Horizontal detail: varies rapidly along H (top-to-bottom), smooth
  along W. Responds to **horizontal edges**.
- **V** — Vertical detail: smooth along H, varies rapidly along W
  (left-to-right). Responds to **vertical edges**.
- **D** — Diagonal detail: varies rapidly in both directions. Responds to
  **diagonal textures**.

At deeper levels, each letter in the compound name corresponds to one
decomposition step, reading left to right from coarsest to finest:

```
AH = first step → A (approximation), second step → H (horizontal detail)
```

## Grid layout within a level

The subbands are tiled into the `[H, W]` output in **frequency order**:

- **Moving down** (rows of the tiled output): increasing frequency along the
  H axis. Top rows contain subbands that are smooth vertically; bottom rows
  contain subbands that oscillate rapidly vertically.
- **Moving right** (columns of the tiled output): increasing frequency along
  the W axis. Left columns contain subbands that are smooth horizontally; right
  columns contain subbands that oscillate rapidly horizontally.

Both axes independently follow the same **Gray code** frequency ordering used
by the 1-D transform.

---

## Haar wavelet examples (filter length = 2)

All examples use an `8 x 8` image.  With Haar, `max_level = 3`.

### Level 1 — 2x2 grid of 4x4 blocks

The image is split once along each axis, producing 4 subbands:

```
              ┌──── W-axis frequency ────┐
              low                   high

         ┌──────────────┬──────────────┐
  H low  │              │              │
         │      A       │      V       │
         │   (4×4)      │   (4×4)      │
         ├──────────────┼──────────────┤
  H high │              │              │
         │      H       │      D       │
         │   (4×4)      │   (4×4)      │
         └──────────────┴──────────────┘
```

- **A** (top-left): smooth approximation of the image.
- **V** (top-right): vertical edges — sharp changes as you move left-to-right.
- **H** (bottom-left): horizontal edges — sharp changes as you move top-to-bottom.
- **D** (bottom-right): diagonal detail — checkerboard-like patterns.

### Level 2 — 4x4 grid of 2x2 blocks

Each of the 4 level-1 subbands is split again.  Both axes independently follow
the Gray code frequency order from the 1-D case:

```
1-D Gray code at level 2:  [aa, ad, dd, da]  →  [low, ..., high]
```

The 4x4 frequency-ordered grid, with each cell showing its compound A/H/V/D
label:

```
                ┌────── W-axis frequency ─────┐
                aa       ad        dd        da

           ┌─────────┬─────────┬─────────┬─────────┐
       aa  │   AA    │   AV    │   VV    │   VA    │
           ├─────────┼─────────┼─────────┼─────────┤
 H-    ad  │   AH    │   AD    │   VD    │   VH    │
axis       ├─────────┼─────────┼─────────┼─────────┤
freq   dd  │   HH    │   HD    │   DD    │   DH    │
           ├─────────┼─────────┼─────────┼─────────┤
       da  │   HA    │   HV    │   DV    │   DA    │
           └─────────┴─────────┴─────────┴─────────┘

Each block is 2×2 pixels.  16 subbands tile the full 8×8 output.
```

Reading the compound labels: the first letter comes from the level-1 split, the
second from the level-2 split.  For example, `HD` at position (row=2, col=1)
means:

- Level 1: **H** (horizontal detail) — this part of the image had high H-axis
  frequency and low W-axis frequency.
- Level 2: **D** (diagonal detail) — within that, this subband captures the
  diagonal variations.

The top-left corner (**AA**) is the smoothest approximation.  The bottom-right
corner (**DA**) corresponds to the highest frequency bands in both axes.

### Level 3 — 8x8 grid of 1x1 blocks

Every subband shrinks to a single pixel.  Each axis uses the 1-D Gray code
order for 3 splits:

```
1-D Gray code at level 3:  [aaa, aad, add, ada, dda, ddd, dad, daa]
```

The 8x8 grid has 64 subbands, each with a 3-letter A/H/V/D label.  The top-left
4x4 corner (lowest frequencies in both axes):

```
                ┌──────── W-axis frequency ────────
                aaa       aad       add       ada     ...

           ┌─────────┬─────────┬─────────┬─────────┬────
      aaa  │  AAA    │  AAV    │  AVV    │  AVA    │
           ├─────────┼─────────┼─────────┼─────────┤
 H-   aad  │  AAH    │  AAD    │  AVD    │  AVH    │
axis       ├─────────┼─────────┼─────────┼─────────┤
freq  add  │  AHH    │  AHD    │  ADD    │  ADH    │
           ├─────────┼─────────┼─────────┼─────────┤
      ada  │  AHA    │  AHV    │  ADV    │  ADA    │
           ├─────────┼─────────┼─────────┼─────────┤
     ...   │                  ...
```

The bottom-right corner (highest frequencies) contains labels starting with
**D** — e.g. **DAA** at position (7, 7).

### How to compute the label at any grid position

The label at grid position `(r, c)` at level `l` is determined by:

1. Look up the 1-D Gray code sequences for level `l`:
   - H-axis chain = `gray[r]`  (the r-th entry)
   - W-axis chain = `gray[c]`  (the c-th entry)
2. For each decomposition step `p` (0 to l-1), combine the p-th character of
   each chain into a 2-D label:

| H-axis char | W-axis char | Letter |
|-------------|-------------|--------|
| a           | a           | A      |
| d           | a           | H      |
| a           | d           | V      |
| d           | d           | D      |

### Full output tensor

The complete output for `[batch, 8, 8]` input is `[batch, 3, 8, 8]`:

```
output[:, 0, :, :]  →  level 1:   2×2 grid of 4×4 blocks    (4 subbands)
output[:, 1, :, :]  →  level 2:   4×4 grid of 2×2 blocks   (16 subbands)
output[:, 2, :, :]  →  level 3:   8×8 grid of 1×1 blocks   (64 subbands)
```

Visualized as three layers:

```
Level 1                 Level 2                  Level 3
┌──────┬──────┐         ┌──┬──┬──┬──┐           ┌─┬─┬─┬─┬─┬─┬─┬─┐
│      │      │         │  │  │  │  │           ├─┼─┼─┼─┼─┼─┼─┼─┤
│  A   │  V   │         ├──┼──┼──┼──┤           ├─┼─┼─┼─┼─┼─┼─┼─┤
│      │      │         │  │  │  │  │           ├─┼─┼─┼─┼─┼─┼─┼─┤
├──────┼──────┤         ├──┼──┼──┼──┤           ├─┼─┼─┼─┼─┼─┼─┼─┤
│      │      │         │  │  │  │  │           ├─┼─┼─┼─┼─┼─┼─┼─┤
│  H   │  D   │         ├──┼──┼──┼──┤           ├─┼─┼─┼─┼─┼─┼─┼─┤
│      │      │         │  │  │  │  │           ├─┼─┼─┼─┼─┼─┼─┼─┤
└──────┴──────┘         └──┴──┴──┴──┘           └─┴─┴─┴─┴─┴─┴─┴─┘
 4 subbands              16 subbands             64 subbands
 coarse resolution       medium                  maximum resolution
```

At every level, the **top-left** corner is always the smoothest approximation
and the **bottom-right** is always the highest-frequency detail.

### Relationship to the 1-D frequency order

The 2-D grid is a **Cartesian product** of two 1-D Gray code orderings.
If the 1-D Gray code order at level `l` is the sequence `G_l`, then:

```
grid[r][c]  has  H-axis chain = G_l[r],  W-axis chain = G_l[c]
```

Everything from the 1-D frequency ordering applies independently to each axis
of the 2-D grid.

## Extracting individual subbands

To extract a single subband at level `l` and grid position `(r, c)`:

```cpp
int64_t bands = 1 << l;  // 2^l bands per axis
int64_t bh = H / bands;
int64_t bw = W / bands;

auto level_slice = coeffs.select(/*dim=*/-3, /*index=*/l - 1);
auto subband = level_slice.slice(-2, r * bh, (r + 1) * bh)
                          .slice(-1, c * bw, (c + 1) * bw);
// subband shape: [batch, bh, bw]
```
