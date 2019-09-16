from data_types import Genome
import pygraphviz as pgv
import os

class GenomeRenderer:

    def __init__(self, genome: Genome, filename: str = 'genome_render.png', dirname: str = 'renders'):
        """Draws a genome in png format using pygraphviz
        
        Arguments:
            genome {Genome} -- genome to draw
        """
        graph = pgv.AGraph()
        for (src, dst) in [(inn.src, inn.dst) for inn in genome.innovations]:
            graph.add_edge(src, dst)
        graph.layout()

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        graph.draw(path = os.path.join(dirname, filename))
