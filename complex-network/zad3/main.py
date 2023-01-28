from src.Service.ModelNetworkTester import ModelNetworkTester


if __name__ == '__main__':
    networks = [
        './data/networks/4_1.txt',
        './data/networks/4_2.txt',
        './data/networks/4_3.txt',
        './data/networks/4_4.txt',
        './data/networks/4_5.txt',
        './data/networks/12_1.txt',
        './data/networks/12_2.txt',
        './data/networks/12_3.txt',
        './data/networks/12_4.txt',
        './data/networks/12_5.txt',
    ]

    model_network_tester = ModelNetworkTester(verbosity_level=0)
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
    
    sieci do liczenia: 4, 12, 
    _[1...5]
    pp = [0.1, ..., 0.9]
    sf = [5, 10, 25]
    ss = [cc, dg, random]
    """
