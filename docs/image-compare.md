# ImageCompare

[RayTracing](../README.md) › ImageCompare

Every pixel of a render is a Monte Carlo estimate, an average of random samples. So two renders of one scene differ in every pixel unless they use the same random numbers. What correct renderers must share is each pixel's expected value: the picture both converge to as the rays per pixel grow. `ImageCompare` tests whether two renders differ by more than their noise explains. It is how the CUDA renderer was [verified against the CPU](gpu.md#verification).

- [Usage](#usage)
- [How it decides](#how-it-decides)
- [Validation](#validation)
- [Using it correctly](#using-it-correctly)

## Usage

`ImageCompare` reads PFM files, which keep the renderer's own linear numbers; a PNG's 8 bits after gamma correction would hide a difference of 0.1%. For example, render a scene twice to PFM, with different seeds and otherwise the same settings:

```
RayTracing materials -w 480 -s 64 -o a.pfm
RayTracing materials -w 480 -s 64 --seed 1 -o b.pfm
ImageCompare a.pfm b.pfm
```

```
A: a.pfm: 480×270, mean radiance 0.266767
B: b.pfm: 480×270, mean radiance 0.266774

The difference d = B - A in each pixel, over 129600 pixels:
              mean of d          SE        z   relative   RMS of d
  red       -0.00003300  0.00007180    -0.46   -0.0117%   0.025847
  green     -0.00005197  0.00007913    -0.65   -0.0145%   0.028485
  blue      +0.00010775  0.00008614    +1.25   +0.0671%   0.031009
  average   +0.00000759  0.00006400    +0.11   +0.0028%   0.023039
  SE: the standard error of the mean; z ≈ mean / SE, the mean in standard errors (less a rounding
  margin of a millionth of the pixel values); relative: the mean as a share of the pixel values;
  RMS: the typical size of d, the two images' noise together
  sign test: B is brighter in 50.10% of the 127341 pixels that differ by more than 0.1%,
  z = +0.69 (noise of the same shape in A and B gives 50%)

Tiles of 32×32 pixels (smaller at the right and bottom edges): 135 tested, by the channel average of d
  all tiles together: Σz/√T = +0.66
  largest |z|: 2.69 at x 96–127, y 96–127 (A is brighter there)
  tiles with |z| > 3: 0, while noise alone gives 0.4 on average

Checks (by chance alone each fails in at most 1 of 400 comparisons, any of the 4 in 1 of 100):
  whole image   the largest |z| of its 4 rows        1.25   fails above 3.42       passed
  all tiles     |Σz/√T| of the T tiles               0.66   fails above 3.02       passed
  largest tile  the largest |z| of a tile            2.69   fails above 4.28       passed
  tile count    the number of tiles with |z| > 3        0   fails at 4 or more     passed

Verdict: consistent: A and B differ by no more than noise
```

Its options, as its help lists them:

```
ImageCompare A.pfm B.pfm [options]
  --tile N         the side of the square tiles that are tested one by one, in pixels (default 32)
  --zmap FILE.png  save a map of the tiles' z-scores: image A in gray, tinted red where B is
                   brighter and blue where it is darker, in full color at the largest tile's
                   alarm threshold
  -h, --help       show this help
```

The exit code gives the verdict to scripts: 0 consistent, 1 different, 2 an error. The README shows [two such maps](../README.md#verified-against-the-cpu), of a CPU render against another CPU render and against a GPU render.

## How it decides

It looks at the difference d = B − A in every pixel. If A and B are the same picture, every d has the expected value 0. The mean of N of them is then as close to 0 as the standard error SE = s/√N predicts, where s is the spread of the d's. This is the √N law that the [MonteCarlo program](sampling.md#the-montecarlo-program) demonstrates. z = mean / SE is that distance in standard errors. The image is also cut into tiles of 32×32 pixels (`--tile N`), each with its own z. Four checks look for four kinds of difference:

- **Whole image:** the mean of d per channel and for their average. It catches a bias that covers the image, such as a render 0.1% too dark.
- **All tiles:** the tiles' z added up and divided by √T (Stouffer's method). Every tile weighs the same, so one noisy region (glass, fireflies) can't drown a small bias everywhere else.
- **Largest tile:** a difference confined to one object, against a threshold that grows with the number of tiles, because some tiles are far off by chance (Bonferroni's correction).
- **Tile count:** more tiles with |z| > 3 than chance gives (a binomial test), for a weak difference spread over many tiles.

Each check fails by chance in at most 1 of 400 comparisons of the same picture, so the verdict is a false alarm in at most 1 of 100. The source documents the statistics it relies on: the standard error, Student's t for small tiles, Bonferroni's correction, Stouffer's method and the sign test.

## Validation

The program was validated on thousands of synthetic pairs, with and without fireflies (rare samples far brighter than the rest), and on 1725 pairs of real renders with different seeds. It called the same picture different in 0.5–0.9% of the pairs. It found a sky 0.1% too bright, or one sphere's albedo 1% too low, in 30 of 30 pairs of `materials` at 480×270 and 64 rays per pixel.

## Using it correctly

A and B must be renders of the same scene at the same size, with the same rays per pixel and the same sampling method. Then their noise has the same distribution, which the checks assume. They must also be independent: renders with different seeds, or renders from different renderers, as the CPU and the GPU draw unrelated random numbers. To compare the GPU's `float` and `double`, give them different seeds, as their generators are the same.

With different rays per pixel the noise of A and B is distributed differently. A pixel of 16 rays usually misses the rare bright samples that one of 1024 catches, and the checks can fail for the same picture. The `sign test` line watches for this: with noise of the same shape, B is brighter in half the pixels, like heads in tosses of a fair coin. The program prints a note when that looks like the cause. Identical files are refused.
