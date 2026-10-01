# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Common base class for the errors K_fit raises on bad input.

Every module's own error (StructureError, FetchError, PairsCsvError,
WriterError, PipelineError) derives from ``KFitError``. A program using
K_fit can catch ``KFitError`` to report user mistakes plainly and still let
real bugs raise. Give any new error class this base too.
"""


class KFitError(Exception):
    """Base class of all K_fit errors caused by input or user choices."""
