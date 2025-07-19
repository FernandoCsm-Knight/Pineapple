#ifndef LAYERS_HPP
#define LAYERS_HPP

// Tensor library

#include "inc/tensor/tensor.hpp"

// Abstract classes

#include "inc/abstract/layer.hpp"
#include "inc/abstract/optimizer.hpp"
#include "inc/abstract/activation.hpp"
#include "inc/abstract/loss_function.hpp"
#include "inc/abstract/metric_collection.hpp"

// Layers

#include "inc/layer/flatten_layer.hpp"
#include "inc/layer/linear_layer.hpp"
#include "inc/layer/sequential.hpp"
#include "inc/layer/dropout_layer.hpp"

// Optimizers

#include "inc/optimizer/gd.hpp"
#include "inc/optimizer/sgd.hpp"
#include "inc/optimizer/adam.hpp"
#include "inc/optimizer/adamw.hpp"

// Activation functions

#include "inc/activation/elu.hpp"
#include "inc/activation/sigmoid.hpp"
#include "inc/activation/tanh.hpp"
#include "inc/activation/relu.hpp"
#include "inc/activation/leaky_relu.hpp"
#include "inc/activation/softmax.hpp"

// Loss functions

#include "inc/loss/cross_entropy_loss.hpp"
#include "inc/loss/binary_cross_entropy_loss.hpp"
#include "inc/loss/mse_loss.hpp"
#include "inc/loss/huber_loss.hpp"
#include "inc/loss/mae_loss.hpp"

// Metrics

#include "inc/metrics/confusion_matrix_collection.hpp"

#include "inc/metrics/accuracy.hpp"
#include "inc/metrics/f1score.hpp"
#include "inc/metrics/precision.hpp"
#include "inc/metrics/recall.hpp"
#include "inc/metrics/specificity.hpp"

#include "inc/metrics/regression_collection.hpp"

#include "inc/metrics/mae.hpp"
#include "inc/metrics/mse.hpp"
#include "inc/metrics/r2_score.hpp"

// Neural network

#include "inc/neural_network.hpp"

// Helpers

#include "inc/generator.hpp"
#include "inc/data/partition.hpp"

#endif