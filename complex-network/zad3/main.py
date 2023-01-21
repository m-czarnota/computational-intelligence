from src.Service.ModelNetworkTester import ModelNetworkTester


if __name__ == '__main__':
    model_network_tester = ModelNetworkTester(verbosity_level=2)

    networks = ['./data/networks/4_1.txt']
    model_network_tester.test_for_networks(networks)


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
    """
