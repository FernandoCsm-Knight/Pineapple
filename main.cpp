#include "inc/pineapple.hpp"

// int main() {
//     Tensor<float> tensor(3, 3);
//     Tensor<float> other(3, 3);

//     other.fill(1.0f);

//     tensor(1) += other(1);

//     std::cout << tensor << std::endl;
//     std::cout << other << std::endl;
//     return 0;
// }

int main() {
    const int NUM_SAMPLES = 1000;
    const int NUM_CLASSES = 2;
    const int NUM_FEATURES = 2;
    const float LEARNING_RATE = 0.001f;
    const int EPOCHS = 1000;
    const int BATCH_SIZE = 32;

    auto [features, labels] = pineapple::create_zebra<float>(NUM_SAMPLES, NUM_CLASSES);

    Partition<float> partition = Partition<float>(features, labels, false);
    auto [X_train, y_train, X_test, y_test] = partition.split(0.2f);

    LinearLayer<float>* layer1 = new LinearLayer<float>(NUM_FEATURES, 16);
    ReLU<float>* relu1 = new ReLU<float>();
    
    LinearLayer<float>* layer2 = new LinearLayer<float>(16, 8);
    ReLU<float>* relu2 = new ReLU<float>();
    
    LinearLayer<float>* layer3 = new LinearLayer<float>(8, NUM_CLASSES);
    Softmax<float>* softmax = new Softmax<float>();
    
    NeuralNetwork<float> model(
        new Sequential<float>({
            layer1, relu1,
            layer2, relu2,
            layer3, softmax
        }),
        new CrossEntropyLoss<float>(),
        new GD<float>(LEARNING_RATE)
    );

    std::cout << "Iniciando treinamento..." << std::endl;
    model.train(X_train, y_train, EPOCHS, BATCH_SIZE);
    
    float accuracy_train = model.evaluate(X_train, y_train);
    float accuracy_test = model.evaluate(X_test, y_test);
    
    std::cout << "Acurácia de treinamento: " << (accuracy_train * 100) << "%" << std::endl;
    std::cout << "Acurácia de teste: " << (accuracy_test * 100) << "%" << std::endl;

    return 0;
}