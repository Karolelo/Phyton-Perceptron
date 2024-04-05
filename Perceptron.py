import pandas as pd
import random

class Perceptron:
    def __init__(self, a,t,vecSize,possbileValues):
        self.learning_rate = a
        self.threshold = t
        self.weights =[0.0 for _ in range(vecSize)]
        self.setOfValues=list(possbileValues)

    def compute(self, vec):
        activation = 0
        for i in range(int(len(vec) - 1)):
            activation += self.weights[i] * vec[i]
        activation -= self.threshold
        return self.setOfValues[0] if activation >= 0 else self.setOfValues[1]


    def learn(self, vec):
        actual_output=self.compute(vec[:-1])
        if actual_output == vec[-1]:
            return
        if actual_output != self.setOfValues[0]:
            expected_output=0
            actual_output=1
        else :
            expected_output=1
            actual_output=0
        for x in range(len(self.weights)):
            self.weights[x]=self.weights[x]-(expected_output-actual_output)*self.learning_rate*vec[x]

        self.threshold=self.threshold-(expected_output-actual_output)*self.learning_rate



class Trainer:
    def __init__(self,pathToTrainFile,pathToTestFile):

        self.trainFile=self.readFile(pathToTrainFile)
        self.testFile=self.readFile(pathToTestFile)
        random.shuffle(self.trainFile)
        random.shuffle(self.testFile)
        unique_name=set()
        for x in self.trainFile:
            unique_name.add(x[-1])
            if len(unique_name) == 2:
                break
        vecSize=len(self.trainFile[0])-1
        self.perceptron=Perceptron(0.1,1,vecSize,unique_name)


    def readFile(self,file_name):
        df = pd.read_csv(file_name, delimiter=';')
        return df.values.tolist()

    def train(self):
        for x in self.trainFile:
            self.perceptron.learn(x)

    def evaluate(self):
        correctFirstClass=0
        correctSecondClass=0
        for x in self.testFile:
            predicted=self.perceptron.compute(x[:-1])
            predicted = self.perceptron.compute(x[:-1])
            if ((predicted == x[-1]) & (self.perceptron.setOfValues[0] == predicted)):
                correctFirstClass += 1
            if ((predicted == x[-1]) & (self.perceptron.setOfValues[1] == predicted)):
                correctSecondClass += 1

        print((correctFirstClass + correctSecondClass) / len(self.testFile))

class UI:
    def __init__(self):
        self.trainer = None

    def main_menu(self):
        while True:
            print("\n=== Menu ===")
            print("1. Ustaw ścieżki do plików i zainicjuj trening")
            print("2. Rozpocznij trening")
            print("3. Przeprowadź test")
            print("4. Testuj własny wektor")
            print("5. Wyjdź")

            choice = input("Wybierz opcję: ")

            if choice == '1':
                self.setup_and_initiate()
            elif choice == '2':
                self.start_training()
            elif choice == '3':
                self.perform_test()
            elif choice == '4':
                self.test_custom_vector()
            elif choice == '5':
                print("Zamykanie programu.")
                break
            else:
                print("Nieznana opcja, spróbuj ponownie.")

    def setup_and_initiate(self):
        path_to_train = input("Podaj ścieżkę do pliku treningowego: ")
        path_to_test = input("Podaj ścieżkę do pliku testowego: ")
        self.trainer = Trainer(path_to_train, path_to_test)
        print("Inicjalizacja zakończona sukcesem.")

    def start_training(self):
        if self.trainer:
            self.trainer.train()
            print("Trening zakończony.")
        else:
            print("Trainer nie został zainicjalizowany.")

    def perform_test(self):
        if self.trainer:
            self.trainer.evaluate()
        else:
            print("Trainer nie został zainicjalizowany.")

    def test_custom_vector(self):
        if self.trainer:
            vector = input("Podaj wektor (oddziel wartości przecinkami): ")
            vector = [float(val) for val in vector.split(',')]
            prediction = self.trainer.perceptron.compute(vector)
            print(f"Przewidywana klasa dla wektora {vector}: {prediction}")
        else:
            print("Trainer nie został zainicjalizowany.")

##interface=UI()
##interface.main_menu()

trainer=Trainer('TrainToPerceptron.csv','TestToPerceptron.csv')
trainer.evaluate()
trainer.train()
trainer.evaluate()

