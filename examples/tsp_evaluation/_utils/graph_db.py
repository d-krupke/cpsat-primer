import functools
import json
from zipfile import ZIP_LZMA, ZipFile
from pathlib import Path
import networkx as nx


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
