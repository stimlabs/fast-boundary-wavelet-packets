# 1-D Wavelet Packet Transform: Output Format

## Output shape

```
input:  [..., N]
output: [..., max_level, N]
```

The forward transform inserts a new **scale** dimension at the position of the
analyzed dimension.  The analyzed dimension shifts one position to the right.

| Input shape     | `dim` | Output shape               |
|-----------------|-------|----------------------------|
| `[N]`           | `-1`  | `[max_level, N]`           |
| `[batch, N]`    | `-1`  | `[batch, max_level, N]`    |
| `[B1, B2, N]`   | `-1`  | `[B1, B2, max_level, N]`   |
| `[batch, N, C]` | `1`   | `[batch, max_level, N, C]` |

`max_level` is auto-computed if not specified.  For a signal of length `N` and
a wavelet with filter length `F`, the maximum level is the largest `L` such
that `N % 2^L == 0` and `N / 2^(L-1) >= F`.

## What each level contains

Each slice along the scale dimension is a **complete tiling** of the original
signal's frequency content at a different resolution.

```
output[..., 0, :]   →  level 1:  2 subbands, each of size N/2
output[..., 1, :]   →  level 2:  4 subbands, each of size N/4
output[..., 2, :]   →  level 3:  8 subbands, each of size N/8
...
output[..., L-1, :] →  level L:  2^L subbands, each of size N/2^L
```

At every level, the subbands are **concatenated contiguously** along the last
dimension and always sum to exactly `N` samples.  Each level is self-contained:
it represents the full signal, just decomposed into different numbers of
frequency bands.

## Subband ordering (frequency / Gray code)

Within each level, subbands are arranged in **frequency order** from lowest to
highest, following the Gray code convention used by pywt/ptwt.

In frequency order, consecutive subbands are always "neighbors" in frequency
space -- they differ by exactly one filtering step.  This makes the
representation natural for tasks that care about frequency locality (e.g.
spectral features, denoising, compression).

---

## Haar wavelet examples (filter length = 2)

The Haar wavelet is the simplest case.  Its filters are:
- Lowpass (a):  `[1/sqrt(2),  1/sqrt(2)]` — local average
- Highpass (d): `[-1/sqrt(2), 1/sqrt(2)]` — local difference

All examples below use a signal of length `N = 8`.  With Haar, `max_level = 3`
(the smallest subband at level 3 has size `8 / 2^3 = 1`).

### Notation

- `a` = lowpass (approximation) — averages neighboring samples
- `d` = highpass (detail) — differences neighboring samples
- Filter paths read left to right: `ad` means "first lowpass, then highpass"
- Subscripts index samples within a subband

### Level 1 — 2 subbands of size 4

The analysis matrix splits the signal into one lowpass and one highpass subband:

```
signal:    [ x0  x1  x2  x3  x4  x5  x6  x7 ]
                  ↓ analysis (A, 8×8)
              lowpass               highpass
           ┌─────────────────┬─────────────────┐
level 1:   │ a0  a1  a2  a3  │ d0  d1  d2  d3  │
           └─────────────────┴─────────────────┘
            ← subband 0 (a) → ← subband 1 (d) →

  a_i = (x_{2i} + x_{2i+1}) / sqrt(2)     (average of pair)
  d_i = (x_{2i+1} - x_{2i}) / sqrt(2)     (difference of pair)
```

Frequency order: `[a, d]` (2 subbands, low → high).

### Level 2 — 4 subbands of size 2

Each of the 2 level-1 subbands is split again.  The block-diagonal analysis
matrix applies the same 4x4 split to both subbands simultaneously:

```
from level 1:  [ a0  a1  a2  a3 | d0  d1  d2  d3 ]

                   ↓ block-diagonal analysis (A⊕A, 8×8)

natural order: [  aa0  aa1 | ad0  ad1 | da0  da1 | dd0  dd1  ]
                    ↓ Gray code permutation [0, 1, 3, 2]

               ┌───────────┬───────────┬───────────┬───────────┐
level 2:       │ aa0   aa1 │ ad0   ad1 │ dd0   dd1 │ da0   da1 │
               └───────────┴───────────┴───────────┴───────────┘
                subband 0    subband 1    subband 2    subband 3
                (lowest f)                             (highest f)
```

Frequency order: `[aa, ad, dd, da]`.

Note the swap of `da` and `dd` compared to the natural tree order.  The Gray
code permutation ensures subbands are sorted by increasing center frequency.

### Level 3 — 8 subbands of size 1

Each of the 4 level-2 subbands is split one final time.  Each subband shrinks
to a single coefficient:

```
from level 2
(natural order):  [ aa0 aa1 | ad0 ad1 | da0 da1 | dd0 dd1 ]

                       ↓ block-diagonal analysis (A⊕A⊕A⊕A, 8×8)

natural order:    [ aaa | aad | ada | add | daa | dad | dda | ddd ]
                       ↓ Gray code permutation [0, 1, 3, 2, 6, 7, 5, 4]

                  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
level 3:          │ aaa │ aad │ add │ ada │ dda │ ddd │ dad │ daa │
                  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                   sb 0   sb 1  sb 2  sb 3  sb 4  sb 5  sb 6  sb 7
                  lowest                                     highest
                  frequency                                frequency
```

Frequency order: `[aaa, aad, add, ada, dda, ddd, dad, daa]`.

In binary (`a=0, d=1`): `000, 001, 011, 010, 110, 111, 101, 100` — a standard
reflected Gray code.

### Full output tensor

Stacking all three levels, the complete output for `[batch, 8]` input is
`[batch, 3, 8]`:

```
             sample index along N=8
             0     1     2     3     4     5     6     7
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
level 1    │  a0 │  a1 │  a2 │  a3 │  d0 │  d1 │  d2 │  d3 │  2 subbands × 4
           ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
level 2    │ aa0 │ aa1 │ ad0 │ ad1 │ dd0 │ dd1 │ da0 │ da1 │  4 subbands × 2
           ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
level 3    │ aaa │ aad │ add │ ada │ dda │ ddd │ dad │ daa │  8 subbands × 1
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### Tree view

The decomposition follows a binary tree.  Each node is split into a lowpass (a)
and highpass (d) child:

```
                              x
                     ┌────────┴────────┐
                     a                 d            ← level 1
                 ┌───┴───┐         ┌───┴───┐
                aa       ad       da       dd       ← level 2
               ┌─┴─┐   ┌─┴─┐     ┌─┴─┐   ┌─┴─┐
             aaa aad ada add   daa dad dda ddd      ← level 3
```

The output at each level stores the **leaves of that level's tree** in
frequency order (reading the bottom row left-to-right after Gray code
reordering), not in tree order.
