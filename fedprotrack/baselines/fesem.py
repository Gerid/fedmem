from __future__ import annotations

"""FeSEM baseline.

The original FlexCFL reference treats FeSEM as IFCA-style client
selection with plain averaging per cluster. In this repo that maps
cleanly onto the existing IFCA baseline, so the FeSEM adapter simply
re-exports the IFCA client/server/upload types under FeSEM names.
"""

from .ifca import IFCAClient, IFCAServer, IFCAUpload

FeSEMClient = IFCAClient
FeSEMServer = IFCAServer
FeSEMUpload = IFCAUpload

