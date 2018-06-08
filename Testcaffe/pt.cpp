#define CPU_ONLY
#include <caffe/caffe.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;
using google::protobuf::RepeatedPtrField;

#define IDX_DIF_BIT 8
#define FLOAT_BIT 32
#define INDICE_BIT 8

void printLayerInfo(string layernName, int blobnum, int datanum, float sparseRatio,
	float Prune_comp, float Cen_comp){
	printf("%s:\nBlobnum:%d\nTotal datanum:%d\nSparseRatio:%f\n",
		layernName.c_str(), blobnum, datanum, sparseRatio);
	printf("Prunning caused compression:%f\nAdditional Centroid caused compression:%f\n", 
		Prune_comp, Cen_comp);
}

int main()
{
	NetParameter proto;
	
	ReadProtoFromBinaryFile("/home/kaddxu/CMPcaffe/examples/mnist/models/lenet_finetune_stage4_iter_10000.caffemodel", &proto);
	WriteProtoToTextFile(proto, "/home/kaddxu/Testcaffe/netArgs.txt");

	/*the largest index diffenence between two weight value*/
	const int idx_dif_max = 1 << IDX_DIF_BIT;
	/*centroid numbers*/
	const int centroid_num = 1 << INDICE_BIT;

	/*get net layer number*/
	int layernum = proto.layer_size();
	printf("net name: %s\nnet-layer-num: %d\n", proto.name().c_str(), layernum);

	/*calculate convolution layer sparse ratio (= zero / total)*/
	long long total_conv_args = 0;
	long long zero_conv_args = 0;
	/*traverse all layers to find conv-layer*/
	for(int i = 0; i < layernum; i++){
		LayerParameter tmplayer = proto.layer(i);
		if(tmplayer.type() == "CmpConvolution"){
			/*Get this layer weight info*/
			long long layer_total_args = 0;
			long long layer_zero_args = 0;
			/*Use difference index. Need to charge zero when differences to large*/
			long long addtional_zero = 0;
			//cout << tmplayer.name() << endl;

			int blobnum = tmplayer.blobs_size();
			int datanum = 0;
			/*traverse all blob to get data*/
			for(int j = 0; j < blobnum; j++){
				BlobProto tmpblob = tmplayer.blobs(j);
				layer_total_args += tmpblob.data_size();
				datanum += tmpblob.data_size();

				/*record distance between two non-zero weight value*/
				int disBetweenNonZero = 0;

				/*traverse all data to get weight value*/
				for(int data_idx = 0; data_idx < tmpblob.data_size(); data_idx++){
					/*if data[i] == 0 then zero_conv_args ++*/
					float tmpdata = tmpblob.data(data_idx);
					if(fabs(tmpdata - 0) < 1e-6){
						layer_zero_args ++;
						disBetweenNonZero = 0;
					}
					/*Increase distance*/
					disBetweenNonZero += 1;
					if(disBetweenNonZero > idx_dif_max){
						addtional_zero += 1;
						disBetweenNonZero = 0;
					}

				}
			}

			total_conv_args += layer_total_args;
			zero_conv_args += layer_zero_args;
			/*actual save weights*/
			long long actual_args = layer_total_args - layer_zero_args + addtional_zero;
			/*prune caused compression*/
			float Prune_comp = (float)(layer_total_args * (FLOAT_BIT)) / 
				(actual_args * (FLOAT_BIT + IDX_DIF_BIT));
			/*centroid cause compression(with prune)*/
			float Cen_comp = (float)(layer_total_args * (FLOAT_BIT)) / 
				(actual_args * (INDICE_BIT + IDX_DIF_BIT) + centroid_num * FLOAT_BIT);

			printLayerInfo(tmplayer.name(), blobnum, 
				datanum, (float)layer_zero_args / (float)layer_total_args, Prune_comp, Cen_comp);
		}
	}
	return 0;
}

//Iteration 10000, accuracy = 0.9918

/*net name: LeNet
net-layer-num: 9
conv1:
Blobnum:2
Total datanum:520
SparseRatio:3
Prunning caused compression:1.171831
Additional Centroid caused compression:1.199539
conv2:
Blobnum:2
Total datanum:25050
SparseRatio:1
Prunning caused compression:3.968317
Additional Centroid caused compression:9.007551*/

