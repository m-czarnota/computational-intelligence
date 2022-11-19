from src.Graph.RandomGraph import RandomGraph
from src.Graph.ScaleFreeGraph import ScaleFreeGraph
from src.Graph.SmallWorldGraph import SmallWorldGraph

if __name__ == '__main__':
    random_graph = RandomGraph()
    small_world_graph = SmallWorldGraph()
    scale_free_graph = ScaleFreeGraph()

    for graph in [random_graph, small_world_graph, scale_free_graph]:
        graph.save_visualisation_to_file()
        graph.save_animation_to_file()
