Facial Expression + Body Movement Analysis

This module is responsible for the visual side.

The webcam does not directly determine a person's psychological condition. Instead, it extracts two types of observable signals:

Facial-expression features → used to estimate facial-expression/emotion probabilities.
Body-pose and movement features → used to estimate behavioral/movement states.

These outputs will later be combined with text, situation, conversation history, and other signals in the multimodal fusion module.


---------------steps----------------
Datasets to Download:
FER-2013 (Emotion) → 140 MB, 35K images
MediaPipe (Hand/Body) → Built-in library, no download
20BN-Jester (Optional for custom gestures) → 5-10 GB subset

5-Day Timeline:
Day 1: Setup + Emotion model architecture (5-6h)
Day 2: Train emotion model (6-7h)
Day 3: Hand & body gesture detection (5-6h)
Day 4: Testing, API, data export (5-6h)
Day 5: Optimization & documentation (5-6h)