#include <pineapple/pineapple.hpp>

int main() {
    omp_set_num_threads(16); // Change if your performance is not optimal

    const int NUM_SAMPLES = 1000;
    const int NUM_CLASSES = 3;
    const int NUM_FEATURES = 2;

    const float LEARNING_RATE = 0.001f;

    auto [data, target] = pineapple::create_moons<float>(NUM_SAMPLES, NUM_CLASSES);

    Partition<float> partition(data, target, true);
    auto [train_data, train_target, test_data, test_target] = partition.stratified_split(target, 0.2f);

    std::cout << "Train samples: " << train_data.shape(0) << std::endl;
    std::cout << "Test samples: " << test_data.shape(0) << std::endl;

    LinearLayer<float>* hidden1 = new LinearLayer<float>(NUM_FEATURES, 32);
    LeakyReLU<float>* activation1 = new LeakyReLU<float>();
    Dropout<float>* dropout = new Dropout<float>(0.25f);

    LinearLayer<float>* hidden2 = new LinearLayer<float>(32, 16);
    LeakyReLU<float>* activation2 = new LeakyReLU<float>();

    LinearLayer<float>* output = new LinearLayer<float>(16, NUM_CLASSES);
    Softmax<float>* softmax = new Softmax<float>();

    ConfusionMatrixCollection<float>* metrics = new ConfusionMatrixCollection<float>(
        NUM_CLASSES,
        {
            new Accuracy<float>(),
            new F1Score<float>()
        }
    );

    NeuralNetwork<float> model(
        new Sequential<float>({
            hidden1, activation1, dropout,
            hidden2, activation2, 
            output, softmax
        }),
        new CrossEntropyLoss<float>(),
        new SGD<float>(LEARNING_RATE),
        metrics
    );

    std::cout << "Start training..." << std::endl;
    model.train(train_data, train_target, 1000, 256);

    std::cout << "Start testing..." << std::endl;
    model.evaluate(test_data, test_target);

    std::cout << "Test Accuracy:" << metrics->compute("accuracy") * 100 << "%" << std::endl;
    std::cout << "Test F1Score:" << metrics->compute("f1_score") * 100 << "%" << std::endl;

    return 0;
}
