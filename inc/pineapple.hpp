#ifndef LAYERS_HPP
#define LAYERS_HPP

// Tensor library

#include "tensor/tensor.hpp"

// Abstract classes

#include "abstract/layer.hpp"
#include "abstract/optimizer.hpp"
#include "abstract/activation.hpp"
#include "abstract/loss_function.hpp"

// Layers

#include "layer/flatten_layer.hpp"
#include "layer/linear_layer.hpp"
#include "layer/sequential.hpp"

// Optimizers

#include "optimizer/gd.hpp"
#include "optimizer/sgd.hpp"
#include "optimizer/adam.hpp"
#include "optimizer/adamw.hpp"

// Activation functions

#include "activation/elu.hpp"
#include "activation/sigmoid.hpp"
#include "activation/tanh.hpp"
#include "activation/relu.hpp"
#include "activation/leaky_relu.hpp"
#include "activation/softmax.hpp"

// Loss functions

#include "loss/cross_entropy_loss.hpp"
#include "loss/binary_cross_entropy_loss.hpp"
#include "loss/mse_loss.hpp"
#include "loss/huber_loss.hpp"
#include "loss/mae_loss.hpp"

// Metrics
#include "metrics/metric_collection.hpp"
#include "metrics/accuracy.hpp"
#include "metrics/f1score.hpp"
#include "metrics/precision.hpp"
#include "metrics/recall.hpp"
#include "metrics/specificity.hpp"

// Neural network

#include "neural_network.hpp"

// Helpers

#include "generator.hpp"
#include "data/partition.hpp"

#endif