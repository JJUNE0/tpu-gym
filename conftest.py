"""Make the pinned learner source available to local and upstream tests."""

from tpu_gym._vendor import register_flash_chord

register_flash_chord()
