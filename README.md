# Mario Reinforcement Learning

## Single-Threaded Version

The single-threaded implementation achieved the following results:

- At 1 million iterations, it never beats level 1-1.
- At 2 million iterations, it occasionally beats level 1-1.
- At 4 million iterations, it consistently beats level 1-1.
- At 5 million iterations, it flawlessly beats level 1-1.

## How long did this take?
I ran these on my desktop, so they took longer than normal. Roughly 2 hours per million iterations with an RTX 3070 Ti.

## "Multi-Threaded" Implementation

Currently, the multi-threaded implementation is not true multithreading. It can be likened to assigning 8 individuals to perform the same task simultaneously. True multi-threading involves a team of 8 people collaborating on the task. Therefore, I will not be spending time running iterations to test for performance differences, as it is incredibly unlikely to have any positive effect at this moment. However, I will return to this eventually to finish implementing proper multithreading.

# How to Run This Yourself?

Frankly, I doubt anyone cares enough to do so, but I will include this nonetheless.

## Requirements

- Python: 3.6.0 - 3.9.0
- Stable-Baselines3: 1.5.0
- Pip: 21.0 (or newer)
- Gym: 0.21.0
- Wheel: 0.38.0
- Setuptools: 65.5.0

(all can be downloaded via pip)

(I cannot guarantee this will function properly on any version other than those listed above, as that is what I developed with.)
