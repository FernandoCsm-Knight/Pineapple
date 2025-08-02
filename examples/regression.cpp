#include <pineapple/pineapple.hpp>

int main() {
    omp_set_num_threads(16); // Change if your performance is not optimal

    const int NUM_SAMPLES = 1000;
    const int NUM_FEATURES = 2;
    const float LEARNING_RATE = 0.001f;

    auto [data, target] = pineapple::spread_function<float>(NUM_SAMPLES, NUM_FEATURES, [](const Tensor<float>& coef) {
        // y = ax2 + bw + c

        const float x = coef[0];
        const float w = coef[1];

        return 2 * x * x + 1.5 * w + 10;
    }, 0.0f, 1.0f);

    Partition<float> partition(data, target, true);
    auto [train_data, train_target, test_data, test_target] = partition.split(0.2f);

    std::cout << "Train samples: " << train_data.shape(0) << std::endl;
    std::cout << "Test samples: " << test_data.shape(0) << std::endl;

    LinearLayer<float>* hidden1 = new LinearLayer<float>(NUM_FEATURES, 16);
    LeakyReLU<float>* activation1 = new LeakyReLU<float>();
    Dropout<float>* dropout = new Dropout<float>(0.25f);

    LinearLayer<float>* hidden2 = new LinearLayer<float>(16, 8);
    LeakyReLU<float>* activation2 = new LeakyReLU<float>();

    LinearLayer<float>* output = new LinearLayer<float>(8, 1);

    RegressionCollection<float>* metrics = new RegressionCollection<float>({
        new MSE<float>(),
        new R2Score<float>()
    });
    
    NeuralNetwork<float> model(
        new Sequential<float>({
            hidden1, activation1, dropout,
            hidden2, activation2, 
            output
        }),
        new MSELoss<float>(),
        new Adam<float>(LEARNING_RATE),
        metrics
    );

    std::cout << "Start training..." << std::endl;
    model.train(train_data, train_target, 1000, 256);

    std::cout << "Start testing..." << std::endl;
    model.evaluate(test_data, test_target);

    std::cout << "Test MSE:" << metrics->compute("mse") << std::endl;
    std::cout << "Test R2:" << metrics->compute("r2") * 100 << "%" << std::endl;

    return 0;
}
