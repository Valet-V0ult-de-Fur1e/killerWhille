"""
Global parameters for generation of log mel spectrograms.
Adapted from  https://github.com/tensorflow/models/tree/master/research/audioset
"""

#V2
# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 30240  # TODO: TEST. was 16000.
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125  # TODO: test. was 125, then 500 (12.04.22)
MEL_MAX_HZ = 15000  # TODO: test. was 7500. (30240 / 2)
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.

NEXT_SEGMENT_START = 0.5  # (0, 1]: 1 - no overlap, 0 - 100% overlap (impossible), 0.5 - 50% overlap

#GET RID OF BELOW
EXAMPLE_WINDOW_SECONDS = 0.96# Each example contains 96 10ms frames. TODO: Must not change this? NUM_FRAMES * _HOP_LENGTH_?
EXAMPLE_HOP_SECONDS = 0.32     # with zero overlap. #TODO: No. test



#
# # Architectural constants.
# NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
# NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
# EMBEDDING_SIZE = 128  # Size of embedding layer.
#
# # Hyperparameters used in feature and example generation.
# SAMPLE_RATE = 30240  # TODO: TEST. was 16000.
# STFT_WINDOW_LENGTH_SECONDS = 0.025
# STFT_HOP_LENGTH_SECONDS = 0.010
# NUM_MEL_BINS = NUM_BANDS
# MEL_MIN_HZ = 125  # TODO: test. was 125, then 500 (12.04.22)
# MEL_MAX_HZ = 15120  # TODO: test. was 7500.
# LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
# EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
# EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap. #TODO: test
