# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Make this directory importable so the demos' ``from scale_up_common import ...`` works
under pytest (which otherwise imports the modules via their full package path)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
