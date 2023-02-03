from src.Service.GraphModelGenerator import GraphModelGenerator
from src.Service.ModelNetworkTester import ModelNetworkTester


if __name__ == '__main__':
    # networks = [
    #     './data/networks/4_1.txt',
    #     './data/networks/4_2.txt',
    #     './data/networks/4_3.txt',
    #     './data/networks/4_4.txt',
    #     './data/networks/4_5.txt',
    #     './data/networks/12_1.txt',
    #     './data/networks/12_2.txt',
    #     './data/networks/12_3.txt',
    #     './data/networks/12_4.txt',
    #     './data/networks/12_5.txt',
    # ]
    #
    # model_network_tester = ModelNetworkTester(verbosity_level=0)
    # model_network_tester.test_for_networks(networks)

    graph_model_generator = GraphModelGenerator()

    # graph_model_generator.generate_mean_coverage_by_network()
    # graph_model_generator.generate_mean_coverage_by_pp()
    # graph_model_generator.generate_mean_coverage_by_sf()
    # graph_model_generator.generate_mean_coverage_by_measure()
    #
    # graph_model_generator.generate_mean_steps_by_network()
    # graph_model_generator.generate_mean_steps_by_pp()
    # graph_model_generator.generate_mean_steps_by_sf()
    # graph_model_generator.generate_mean_steps_by_measure()

    graph_model_generator.generate_max_coverage_by_method()


    """
    przygotować funkcje do zapisu
    plik | które centrality | seedy | model | propagation propability | procent zarażonych w sieci 
    liczba seedów powinna być jako % węzłów w sieci, a w pliku powinna to być wartość ile tych węzłów było wybieranych
    N to która sieć, jako numer
    N_C to która podsieć
    PP to propagation probability
    SM to metoda do wybierania seedów: CC, DG, Random
    SF (seed fraction) to liczba seedów
    można zapisać to w csv
    
    sieci do liczenia: 4, 12, 
    _[1...5]
    pp = [0.1, ..., 0.9]
    sf = [5, 10, 25]
    ss = [cc, dg, random]
    
    wykresy:
    1) średnie pokrycie (coverage) dla poszczególnych sieci
    2) średnie pokrycie (coverage) dla pp
    3) średnie pokrycie (coverage) dla SF (seed fraction)
    4) średnie pokrycie (coverage) dla SM (metody)
    5) średnia liczba kroków dla poszczególnych sieci
    6) średnia liczba kroków dla pp
    7) średnia liczba kroków dla SF
    8) średnia liczba kroków dla SM (metody)
    
    wykresy dla największych różnic, które możemy zaobserwowaćw danych pomiędzy metodami
    znaleźć największe pokrycie dla każdej z metod i porównać je między sobą wypisując parametry dla metody (jak pp, sf)
    """
