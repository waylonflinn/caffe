#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void EuclideanAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  batch_size_ = bottom[0]->shape(0);

  CHECK_EQ(batch_size_ , bottom[1]->count())
      << "Number of labels must match number of predictions; ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EuclideanAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / batch_size_;

  Dtype accuracy = 0;
  int count = 0;
  for (int i = 0; i < batch_size_; ++i) {

    const int label_value =
        static_cast<int>(bottom_label[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      continue;
    }
    DCHECK_GE(label_value, 0);

    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    // construct
    int predicted_label = round(bottom_data[i * dim]);
    if (predicted_label == label_value) {
        ++accuracy;
    }
    ++count;
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(EuclideanAccuracyLayer);
REGISTER_LAYER_CLASS(EuclideanAccuracy);

}  // namespace caffe
