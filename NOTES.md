#Notes for findings througout the developing experience

In ben_s1, the first video capture of 80 frames of each test position (FIST, PALM, PEACE),
it was found the around 6 frames failed the crophand() function.
The distribution in frame number failure was most around the middle and the end, most likely when I was moving
the most so I will try to move slower for the future tests, ben_s2 and bens3

## Preprocessing
- 9 sessions, 3 classes (FIST/PALM/PEACE), captured with capture_data.py
- Cropped with crop_hand_square, padding=0.25, resized to 256px
- MediaPipe detection failure rate: 2.5%
  -> this is the pipeline's error floor; end-to-end accuracy cannot exceed 97.5%
EOF
