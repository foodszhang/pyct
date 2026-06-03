# Phantom Source Notes

## User Geometry Note

The current three-source phantom should be treated as three physical sources with
different sizes. Each source diameter is roughly `0.8-1.2 cm`. The source shape
is not spherical: it is closer to a chess-piece profile, approximately a
semicircular cap plus a protruding raised part.

For now this is recorded as acquisition metadata only. Future phantom fitting
should avoid assuming equal-radius Gaussian spheres once CT can directly show the
source geometry.

## Optical Medium

The phantom medium recipe is:

```text
45 ml water + 1 g agar powder
```

This is `2.22% w/v` agar-water gel. The current MCX estimate for the phantom
background medium is:

```text
mua  = 0.002 mm^-1
musp = 0.060 mm^-1
g    = 0.90
mus  = 0.600 mm^-1
n    = 1.334
```

These parameters are estimates only. They are separate from the mouse optical
parameters, which remain the original multi-tissue design.
