#include <vector>
#include <iostream>
#include "caffe/kmeans.hpp"
#include "caffe/layers/cmp_conv_layer.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
void CmpConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*Additional part starts here*/
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int *mask_data = this->masks_.cpu_data();
  int count = this->blobs_[0]->count();
  /*mask weight value*/
  for (int i = 0; i < count; ++i)
    muweight[i] *= mask_data[i] ;
  /*use centroid data as weight value*/
  if(this->quantize_term_){
    const Dtype *cent_data = this->centroids_.cpu_data();
    const int *indice_data = this->indices_.cpu_data();
    for (int i = 0; i < count; ++i){
       if (mask_data[i])
         muweight[i] = cent_data[indice_data[i]];
    }
  }
  /*Additional parts ends here*/

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 //LOG(INFO) << "conv backward" << endl;
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int count = this->blobs_[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }

        /*Additional part starts here*/
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        const int *mask_data = this->masks_.cpu_data();
        /*mask diff if coresponding weight value is masked*/
        for (int j = 0; j < count; ++j)
          weight_diff[j] *=  mask_data[j];

        if(this->quantize_term_){
          vector<Dtype> tmpDiff(this->class_num_);
          vector<int> freq(this->class_num_);
          const int *indice_data = this->indices_.cpu_data();

          /*Add all diff in same centroid and get average value*/
          for (int j = 0; j < count; ++j){
            if (mask_data[j] == 1){
              tmpDiff[indice_data[j]] += weight_diff[j];
              freq[indice_data[j]] += 1;
            }
          }
          for (int j = 0; j < count; ++j)
          {
            if (mask_data[j] == 1)
              weight_diff[j] = tmpDiff[indice_data[j]] / freq[indice_data[j]];
          }
        }
        /*Addtional part ends here*/

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}


template <typename Dtype>
void CmpConvolutionLayer<Dtype>::PruneMask()
{
  /*get arg counts and sparse ratio*/
  int count = this->blobs_[0]->count();
  float ratio = this->sparse_ratio_;
  /*get weight matrix*/
  const Dtype* weight = this->blobs_[0]->cpu_data();

  /*get sorted weights to find threshold value*/
  vector<Dtype> sorted_weight(count);   
  for (int i = 0; i < count; ++i){
     sorted_weight[i] = fabs(weight[i]);
  }
  sort(sorted_weight.begin(), sorted_weight.end());
  
  /*index of threshold value in sorted array*/
  int index = int(count*ratio);
  Dtype threshold_val;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  int *mask_data = this->masks_.mutable_cpu_data();
  int pruned_num = 0;
  /*index < 0 means no threshold value*/
  if(index > 0){
    threshold_val = sorted_weight[index-1];
    for (int i = 0; i < count; ++i){
      /*make_data: 1 means valid, 0 means weight value is pruned to be 0*/
      mask_data[i] = ((weight[i] > threshold_val || weight[i] < -threshold_val) ? 1 : 0);
      muweight[i] *= mask_data[i];
      pruned_num += (1 - mask_data[i]) ;
     }
   
  }
  else {
      for(int i = 0; i < count; ++i){
        mask_data[i] = (weight[i]==0 ? 0 :1); //keep unchanged
        pruned_num += (1 - mask_data[i]);
      }
  }
  if(this->quantize_term_)
  {
    kmeans_cluster(this->indices_.mutable_cpu_data(), this->centroids_.mutable_cpu_data(),  
    muweight, count, mask_data, this->class_num_, 1000);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CmpConvolutionLayer);
#endif

INSTANTIATE_CLASS(CmpConvolutionLayer);

}  // namespace caffe
