#include "pineapple.hpp"

int main() {
    const int NUM_SAMPLES = 1000;
    const int NUM_CLASSES = 2;
    const int NUM_FEATURES = 2;
    const float LEARNING_RATE = 0.001f;
    const int EPOCHS = 100;
    const int BATCH_SIZE = 128;

    omp_set_num_threads(16);

    auto [features, labels] = pineapple::create_bubbles<float>(NUM_SAMPLES, NUM_CLASSES, 0.8f);

    Partition<float> partition = Partition<float>(features, labels, false);
    auto [X_train, y_train, X_test, y_test] = partition.split(0.2f);

    LinearLayer<float>* layer1 = new LinearLayer<float>(NUM_FEATURES, 16);
    ReLU<float>* relu1 = new ReLU<float>();
    Dropout<float>* dropout = new Dropout<float>(0.2f);
    
    LinearLayer<float>* layer2 = new LinearLayer<float>(16, 8);
    ReLU<float>* relu2 = new ReLU<float>();
    
    LinearLayer<float>* layer3 = new LinearLayer<float>(8, NUM_CLASSES);
    Softmax<float>* softmax = new Softmax<float>();

    NeuralNetwork<float> model(
        new Sequential<float>({
            layer1, relu1, dropout,
            layer2, relu2,
            layer3, softmax
        }),
        new CrossEntropyLoss<float>(),
        new GD<float>(LEARNING_RATE)
    );

    if(pineapple::is_cuda_available()) {
        model.to(Device::GPU);
    }

    std::cout << "Iniciando treinamento..." << std::endl;
    model.train(X_train, y_train, EPOCHS, BATCH_SIZE);

    return 0;
}