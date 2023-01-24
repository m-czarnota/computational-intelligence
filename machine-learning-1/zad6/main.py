import pandas as pd

from RegressionClassifierTest import RegressionClassifierTest


if __name__ == '__main__':
    regressionClassifierTest = RegressionClassifierTest()
    x, y = regressionClassifierTest.generate_linear_dataset()

    # plt.figure()
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()

    # regressionClassifierTest.experiment(dataset_type='linear', strong_noise=True)
    regressionClassifierTest.experiment(dataset_type='nonlinear', polynomial_degree=2, strong_noise=True)
    print(regressionClassifierTest.results)
    print(pd.DataFrame(regressionClassifierTest.results).to_markdown())

"""
regresja liniowa - zwykły model. jest to optymalizowane mse
mse + l1
mae + l1
mae + l2
sprawdzić czy jest różnica pomiędzy MEA i MSE
dopóki ta przestrzeń cech nie będzie bardzo duża i przewymiarowana, to różnicy pomiędzy L1 i L2 tak łatwo nie zauważymy

zad1:
dane układajace się w postać liniową, wokół linii
wprowadzić szum który nie jest normalny
lista modeli
później for po wszystkich modelach dla takich danych
później dane z zaszumieniem
dane nieliniowe (wysoki stopień wielomianu: 10, 20, 30, 50) stroimy za pomocą modelu liniowego

ze zmiennej x tworzymy cechy wielomianwe za pomocą PolynomialFeatures
a później tworzomy model liniowy

6 modeli: MSE, MAE, (MSE, MAE) x (L1, L2)
MAE szukamy w grupie modeli które mają nazwę Huber
MEA L1, MAE L2 jest w SGDRegressor - może wymagać kilku iteracji, aby zbiec
niektóre modele mogą wychodzić słabo, ale nie należy ich od razu dyskfalifikować, ponieważ ciężko te modele zachęcić do działania

regresja grzbietowa - Ridge model to MSE i L2
ElasticNet działa dla L1 i L2 na danych rzadkich

MSE: LinearRegression
MEA: Huber
MSE + L1: Lasso
MSE + L2: Ridge
MAE + L1: SGDRegressor 
MAE + L2: SGDRegressor

^2 = S (square)
||w||_2 = L2
||w||_1 = L1
loss oznacza, że do MSE możemy zastosować
Huber to MAE

parameter tol oraz max_iter, który trzeba dotknąć, aby uzyskać pożądany efekt
eta0 ma wpływ na szybkość uczenia

liczba parametrów powinna być większa niż liczba próbek, jeżeli nie chcemy obserwować złożoności próbkowej
4 zestawy danych
szum może wychylać w dwie strony próbki. może też w dwie, ale nie może być gaussowski, aby nie były symetrycznie
jest taki rozkład kochego albo mediego

pipeline

w tabelce MSE i MAE
można obniżyć próbki i zobaczyć jak się to zachowuje

"""