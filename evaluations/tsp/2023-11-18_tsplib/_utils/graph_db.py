import functools
import json
from zipfile import ZIP_LZMA, ZipFile
import tarfile
from pathlib import Path
import networkx as nx
import typing
import gzip
import re
import itertools


class GraphInstanceDb:
    """
    Simple helper to store and load the instances.
    Compressed zip to save disk space and making it small
    enough for git.
    """

    def __init__(self, path: Path):
        self.path = path

    @functools.lru_cache(10)
    def __getitem__(self, name):
        with ZipFile(self.path, "r") as z, z.open(name + ".json", "r") as f:
            return nx.json_graph.node_link.node_link_graph(json.load(f))

    def __setitem__(self, name, graph):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.__getitem__.cache_clear()
        with ZipFile(self.path, compression=ZIP_LZMA, mode="a") as instance_archive:
            with instance_archive.open(name + ".json", "w") as f:
                f.write(
                    json.dumps(nx.json_graph.node_link.node_link_data(graph)).encode()
                )

    def __iter__(self):
        if not self.path.exists():
            return
        with ZipFile(self.path, "r") as z:
            for f in z.filelist:
                yield f.filename[:-5]




class TspLibGraphInstanceDb:
    def __init__(self, archive_path: Path = Path("./ALL_tsp.tar.gz")):
        self.archive_path = archive_path
        self.instance_names = [
            # Integer coordinate based instances <= 1_000 nodes
            "att48",
            "att532",
            "eil101",
            "eil51",
            "eil76",
            "gil262",
            "kroA100",
            "kroA150",
            "kroA200",
            "kroB100",
            "kroB150",
            "kroB200",
            "kroC100",
            "kroD100",
            "kroE100",
            "lin105",
            "lin318",
            "linhp318",
            "pr107",
            "pr124",
            "pr136",
            "pr144",
            "pr152",
            "pr226",
            "pr264",
            "pr299",
            "pr439",
            "pr76",
            "st70",
        ]

    def download(self):
        if not self.archive_path.exists():
            # download the file from the internet.
            import urllib.request

            urllib.request.urlretrieve(
                "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz",
                self.archive_path,
            )

    def _parse_points(self, lines: typing.Iterable[str]):
        points = []
        start_parsing = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                start_parsing = True
                continue
            if start_parsing:
                if line.startswith("EOF"):
                    break
                point_data = line.split(" ")
                if not len(point_data) == 3:
                    raise ValueError("Instance is not 2d-coordinate based.")
                x = float(point_data[1])
                y = float(point_data[2])
                points.append(tuple(float(x) for x in line.split()[1:]))
        if not start_parsing:
            raise ValueError("Instance is not coordinate based.")
        return points

    def _create_graph(self, points):
        g = nx.Graph()
        for i, p in enumerate(points):
            g.add_node(i, pos=p)

        def dist(a, b):
            return round(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

        for v, w in itertools.combinations(range(len(points)), 2):
            g.add_edge(v, w, weight=dist(points[v], points[w]))
        assert g.number_of_nodes() == len(points)
        assert g.number_of_edges() == len(points) * (len(points) - 1) // 2
        return g

    def __getitem__(self, name):
        # The instance will be in "name.tsp.gz"
        with tarfile.open(self.archive_path, "r:gz") as t:
            with t.extractfile(name + ".tsp.gz") as f:
                f = gzip.GzipFile(fileobj=f)
                lines = f.readlines()
                lines = [line.decode() for line in lines]  # to string
                return self._create_graph(self._parse_points(lines))

    def __iter__(self):
        yield from self.instance_names

    def deduce_number_of_nodes_from_name(self, instance_name):
        match = re.search(r"\d+$", instance_name)
        return int(match.group()) if match else None

    def selection(self, min_nodes: int, max_nodes: int):
        for instance_name in self:
            n = self.deduce_number_of_nodes_from_name(instance_name)
            assert n is not None
            if min_nodes <= n <= max_nodes:
                yield instance_name
