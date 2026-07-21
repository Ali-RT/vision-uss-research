"""Track B: MF4 <-> video <-> USS-distance alignment.

Self-contained and OPTIONAL. Nothing in the v1/v2 detector pipeline
(notebooks 01-08) imports this package: the trained model needs no distance
signal. This exists to enable the *enhancement* experiments - geometric metric
height and camera/USS fusion - on top of the existing detector.

See docs/mf4_video_alignment.md for the reverse-engineered recipe.
"""
