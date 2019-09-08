from data_types import Genome
import pygraphviz as pgv


class GenomeRenderer:

    def __init__(self, genome: Genome, filename: str = 'genome_render.png'):
        """Draws a genome in png format using pygraphviz
        
        Arguments:
            genome {Genome} -- genome to draw
        """
        graph = pgv.AGraph()
        for (src, dst) in [(inn.src, inn.dst) for inn in genome.innovations]:
            graph.add_edge(src, dst)
        graph.layout()
        graph.draw(filename)
