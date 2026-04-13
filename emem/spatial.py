from typing import Dict, List, Optional, Tuple

import numpy as np
from rtree import index as rtree_index


def _to_xyz(coordinates: np.ndarray) -> Tuple[float, float, float]:
    """Normalize a 2D or 3D coordinate array to an ``(x, y, z)`` tuple."""
    x, y = float(coordinates[0]), float(coordinates[1])
    z = float(coordinates[2]) if len(coordinates) > 2 else 0.0
    return x, y, z


class SpatialIndex:
    """R-tree based spatial index for 3D coordinates.

    Stores ``(id_int, x, y, z)`` tuples. Uses rtree's 3D index.
    IDs are mapped from string UUIDs to integers internally.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        p = rtree_index.Property()
        p.dimension = 3
        p.overwrite = False
        if path:
            self._index = rtree_index.Index(path, properties=p)
        else:
            self._index = rtree_index.Index(properties=p)
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._coords: Dict[str, Tuple[float, float, float]] = {}
        self._counter = 0

    def _get_int_id(self, str_id: str) -> int:
        """Return the internal integer id for *str_id*, assigning one if needed."""
        if str_id not in self._str_to_int:
            self._counter += 1
            self._str_to_int[str_id] = self._counter
            self._int_to_str[self._counter] = str_id
        return self._str_to_int[str_id]

    def insert(self, str_id: str, coordinates: np.ndarray) -> None:
        """Insert *str_id* at *coordinates* into the index."""
        int_id = self._get_int_id(str_id)
        x, y, z = _to_xyz(coordinates)
        self._coords[str_id] = (x, y, z)
        self._index.insert(int_id, (x, y, z, x, y, z))

    def query_radius(self, center: np.ndarray, radius: float) -> List[str]:
        """Find all items within Euclidean distance *radius* of *center*.

        :param center: 3D coordinate array.
        :param radius: Search radius.
        :returns: List of matching string IDs.
        :rtype: List[str]
        """
        cx, cy, cz = _to_xyz(center)
        bbox = (
            cx - radius,
            cy - radius,
            cz - radius,
            cx + radius,
            cy + radius,
            cz + radius,
        )
        candidates = list(self._index.intersection(bbox))
        r_sq = radius * radius
        results = []
        for int_id in candidates:
            str_id = self._int_to_str.get(int_id)
            if str_id is None:
                continue
            ox, oy, oz = self._coords[str_id]
            dist_sq = (ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2
            if dist_sq <= r_sq:
                results.append(str_id)
        return results

    def query_nearest(self, point: np.ndarray, k: int = 5) -> List[str]:
        """Find *k* nearest items to *point*.

        :param point: 3D coordinate array.
        :param k: Number of nearest neighbours.
        :returns: List of matching string IDs.
        :rtype: List[str]
        """
        x, y, z = _to_xyz(point)
        nearest = list(self._index.nearest((x, y, z, x, y, z), k))
        return [
            self._int_to_str[int_id] for int_id in nearest if int_id in self._int_to_str
        ]

    def delete(self, str_id: str, coordinates: np.ndarray) -> None:
        """Remove *str_id* at *coordinates* from the index."""
        int_id = self._str_to_int.get(str_id)
        if int_id is None:
            return
        x, y, z = _to_xyz(coordinates)
        self._index.delete(int_id, (x, y, z, x, y, z))
        del self._str_to_int[str_id]
        del self._int_to_str[int_id]
        self._coords.pop(str_id, None)

    @property
    def size(self) -> int:
        """Number of items currently stored in the index."""
        return len(self._str_to_int)
