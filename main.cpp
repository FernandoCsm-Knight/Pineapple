#include "inc/pineapple.hpp"

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
    
    ConfusionMatrixCollection<float>* cm = new ConfusionMatrixCollection<float>(
        NUM_CLASSES,
        { new Accuracy<float>() }
    );

    NeuralNetwork<float> model(
        new Sequential<float>({
            layer1, relu1,
            layer2, relu2,
            layer3, softmax
        }),
        new CrossEntropyLoss<float>(),
        new GD<float>(LEARNING_RATE), 
        cm
    );

    std::cout << "Iniciando treinamento..." << std::endl;
    model.train(X_train, y_train, EPOCHS, BATCH_SIZE);
    
    model.evaluate(X_train, y_train);
    std::cout << "Acurácia de treinamento: " << cm->compute("accuracy") << std::endl;
    
    model.evaluate(X_test, y_test);
    std::cout << "Acurácia de teste: " <<  cm->compute("accuracy") << std::endl;

    return 0;
}

// int main() {
//     const int NUM_SAMPLES = 1000;
//     const int NUM_FEATURES = 2;
//     const float LEARNING_RATE = 0.01f;
//     const int EPOCHS = 1000;
//     const int BATCH_SIZE = 32;

//     // Gerar dados para regressão
//     Tensor<float> X(NUM_SAMPLES, NUM_FEATURES);
//     Tensor<float> y(NUM_SAMPLES, 1);
    
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::normal_distribution<float> noise(0.0f, 0.2f);
//     std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    
//     // Criar uma função não-linear: y = x1^2 + 2*x2 + ruído
//     for(int i = 0; i < NUM_SAMPLES; ++i) {
//         float x1 = uniform(gen);
//         float x2 = uniform(gen);
        
//         X(i, 0) = x1;
//         X(i, 1) = x2;
//         y(i, 0) = x1 * x1 + 2 * x2 + noise(gen);
//     }

//     Partition<float> partition = Partition<float>(X, y, false);
//     auto [X_train, y_train, X_test, y_test] = partition.split(0.2f);

//     // Criar coleção de métricas para regressão
//     RegressionCollection<float>* regression_metrics = new RegressionCollection<float>({
//         new MSE<float>(),
//         new MAE<float>(),
//         new R2Score<float>()
//     });

//     // Arquitetura da rede neural para regressão
//     LinearLayer<float>* layer1 = new LinearLayer<float>(NUM_FEATURES, 32);
//     ReLU<float>* relu1 = new ReLU<float>();
    
//     LinearLayer<float>* layer2 = new LinearLayer<float>(32, 16);
//     ReLU<float>* relu2 = new ReLU<float>();
    
//     LinearLayer<float>* layer3 = new LinearLayer<float>(16, 8);
//     ReLU<float>* relu3 = new ReLU<float>();
    
//     LinearLayer<float>* output_layer = new LinearLayer<float>(8, 1);
    
//     NeuralNetwork<float> model(
//         new Sequential<float>({
//             layer1, relu1,
//             layer2, relu2,
//             layer3, relu3,
//             output_layer 
//         }),
//         new MSELoss<float>(),
//         new GD<float>(LEARNING_RATE),
//         regression_metrics  // Passar as métricas para o modelo
//     );

//     std::cout << "Iniciando treinamento para regressão..." << std::endl;
//     model.train(X_train, y_train, EPOCHS, BATCH_SIZE);
    
//     std::cout << "\n=== AVALIAÇÃO DO MODELO ===" << std::endl;
    
//     // Avaliar no conjunto de treinamento
//     std::cout << "\nMétricas de Treinamento:" << std::endl;
//     model.evaluate(X_train, y_train);
//     std::cout << "MSE: " << regression_metrics->compute("mse") << std::endl;
//     std::cout << "MAE: " << regression_metrics->compute("mae") << std::endl;
//     std::cout << "R²: " << regression_metrics->compute("r2") << std::endl;
    
//     // Avaliar no conjunto de teste
//     std::cout << "\nMétricas de Teste:" << std::endl;
//     model.evaluate(X_test, y_test);
//     std::cout << "MSE: " << regression_metrics->compute("mse") << std::endl;
//     std::cout << "MAE: " << regression_metrics->compute("mae") << std::endl;
//     std::cout << "R²: " << regression_metrics->compute("r2") << std::endl;
    
//     // Mostrar algumas predições vs valores reais
//     Tensor<float> test_pred = model.forward(X_test);
//     std::cout << "\nAlgumas predições vs valores reais (teste):" << std::endl;
//     std::cout << "Predito\t\tReal\t\tErro Absoluto" << std::endl;
//     for(size_t i = 0; i < 10 && i < y_test.length(); ++i) {
//         float pred = test_pred[i];
//         float real = y_test[i];
//         float error = std::abs(pred - real);
//         std::cout << std::fixed << pred << "\t\t" << real << "\t\t" << error << std::endl;
//     }

//     return 0;
// }