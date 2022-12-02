#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace nvinfer1;
using namespace std;

typedef struct _y_thres
{
	int x;
	int y;
}y_thres;


inline void debug_print(ITensor *input_tensor, string name) {
	cout << name << ";";
	for (int i = 0;i < input_tensor->getDimensions().nbDims;i++) {
		cout << input_tensor->getDimensions().d[i] << " ";
	}
	cout << endl;
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
inline std::map<std::string, Weights> loadWeights(const std::string file) {
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = nvinfer1::DataType::kFLOAT;

		// Load blob
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}
		wt.values = val;

		wt.count = size;
		weightMap[name] = wt;
	}

	return weightMap;
}

inline string get_bn_name(string lname) {
	string bn_name;
	if (lname[0] == 'e') {
		int idx = lname.find_last_of('.');
		string num = lname.substr(idx + 1);
		int bn_num = stoi(num) + 1;
		bn_name = lname.substr(0, idx + 1) + to_string(bn_num);
	}
	if (lname[0] == 'm') {
		char num = lname.back();
		string bn_lname = "merge.bn";
		bn_name = bn_lname + num;
	}
	return bn_name;
}

inline ITensor *MeanStd(INetworkDefinition *network, ITensor *input, float *mean, float *std, bool div255) {
	ITensor **cat = new ITensor*[3];
	if (div255) {
		Weights Div_225{ nvinfer1::DataType::kFLOAT, nullptr, 1 };
		float *wgt = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
		for (int i = 0; i < 1; ++i) {
			wgt[i] = 255.0f;
		}
		Div_225.values = wgt;
		IConstantLayer* d = network->addConstant(Dims3{ 1, 1, 1 }, Div_225);
		input = network->addElementWise(*input, *d->getOutput(0), ElementWiseOperation::kDIV)->getOutput(0);
	}
	for (int i = 0;i < 3;i++)
	{
		cat[i] = input;
	}
	ITensor *data = network->addConcatenation(cat, 3)->getOutput(0);
	int len = 3;
	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = 1.0 / std[i];
	}
	Weights scale{ nvinfer1::DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = -1.0* (mean[i] / std[i]);
	}
	Weights shift{ nvinfer1::DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ nvinfer1::DataType::kFLOAT, pval, len };

	IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kCHANNEL, shift, scale, power);
	debug_print(scale_1->getOutput(0), "preInput");
	return scale_1->getOutput(0);
}
inline IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;
	float *mean = (float*)weightMap[lname + ".running_mean"].values;
	float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ nvinfer1::DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ nvinfer1::DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ nvinfer1::DataType::kFLOAT, pval, len };

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	return scale_1;
}

inline ITensor* cbr(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor *input, string lname, int c_out, int k = 1, int s = 1, int p = 0, bool if_bias = false) {
	Weights bias{ nvinfer1::DataType::kFLOAT,nullptr,0 };
	if (if_bias)
	{
		bias = weightMap[lname + ".bias"];
	}
	auto conv = network->addConvolutionNd(*input, c_out, Dims2{ k, k }, weightMap[lname + ".weight"], bias);
	assert(conv);
	conv->setStrideNd(Dims2{ s,s });
	conv->setPaddingNd(Dims2{ p,p });
	debug_print(conv->getOutput(0), lname + ".weight");

	string bn_name = get_bn_name(lname);
	auto bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), bn_name, 1e-5);
	debug_print(bn->getOutput(0), bn_name + ".weight");
	auto relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
	assert(relu);
	return relu->getOutput(0);
}

inline ITensor *concat(INetworkDefinition *network, ITensor *low, ITensor *high) {
	int dim = low->getDimensions().d[0];
	int up_h = high->getDimensions().d[1];
	IResizeLayer *up = network->addResize(*low);
	up->setResizeMode(ResizeMode::kLINEAR);
	up->setAlignCorners(false);
	up->setOutputDimensions(Dims3{ dim,up_h,up_h });
	ITensor *cat_tensor[] = { up->getOutput(0),high };
	/*ITensor **cat_tensor = new ITensor*[2];
	cat_tensor[0] = up->getOutput(0);
	cat_tensor[1] = high;*/
	auto cat = network->addConcatenation(cat_tensor, 2);
	debug_print(cat->getOutput(0), "concat");
	return cat->getOutput(0);
}

inline ITensor *get_constant(INetworkDefinition *network, int c, int h, int w, float value_) {
	int count = c * h * w;
	Weights constant_matrix{ nvinfer1::DataType::kFLOAT,nullptr, count };
	float *value_array = new float[count];
	for (int i = 0;i < count;i++) {
		value_array[i] = value_;
	}
	constant_matrix.values = value_array;
	auto result = network->addConstant(Dims3{ c,w,h }, constant_matrix);
	return result->getOutput(0);
}

inline ITensor *get_output(INetworkDefinition *network, ITensor *input, map<string, Weights> &weightMap, int mark) {
	if (mark == 1) {
		auto score_conv = network->addConvolutionNd(*input, 1, Dims2{ 1,1 }, weightMap["output.conv1.weight"], weightMap["output.conv1.bias"]);
		score_conv->setStrideNd(Dims2{ 1,1 });
		score_conv->setPaddingNd(Dims2{ 0,0 });
		auto socre_activation = network->addActivation(*score_conv->getOutput(0), ActivationType::kSIGMOID);
		debug_print(socre_activation->getOutput(0), "output1");
		return socre_activation->getOutput(0);
	}
	if (mark == 2) {
		auto loc_conv = network->addConvolutionNd(*input, 4, Dims2{ 1,1 }, weightMap["output.conv2.weight"], weightMap["output.conv2.bias"]);
		loc_conv->setStrideNd(Dims2{ 1,1 });
		loc_conv->setPaddingNd(Dims2{ 0,0 });
		auto loc_activation = network->addActivation(*loc_conv->getOutput(0), ActivationType::kSIGMOID);
		int loc_c = loc_activation->getOutput(0)->getDimensions().d[0];
		int h = loc_activation->getOutput(0)->getDimensions().d[1];
		int w = loc_activation->getOutput(0)->getDimensions().d[2];
		auto scope = get_constant(network, loc_c, h, w, 512);
		auto loc_mul_scope = network->addElementWise(*loc_activation->getOutput(0), *scope, ElementWiseOperation::kPROD);

		auto angle_conv = network->addConvolutionNd(*input, 1, Dims2{ 1,1 }, weightMap["output.conv3.weight"], weightMap["output.conv3.bias"]);
		angle_conv->setStrideNd(Dims2{ 1,1 });
		angle_conv->setPaddingNd(Dims2{ 0,0 });
		auto angle_activation = network->addActivation(*angle_conv->getOutput(0), ActivationType::kSIGMOID);
		int angle_c = angle_activation->getOutput(0)->getDimensions().d[0];
		auto arg_05 = get_constant(network, angle_c, h, w, 0.5);
		auto angle_sub = network->addElementWise(*angle_activation->getOutput(0), *arg_05, ElementWiseOperation::kSUB);
		auto pi = get_constant(network, angle_c, h, w, atan(1) * 4);
		auto angle_mul_pi = network->addElementWise(*angle_sub->getOutput(0), *pi, ElementWiseOperation::kPROD);
		ITensor *cat[] = { loc_mul_scope->getOutput(0),angle_mul_pi->getOutput(0) };
		auto output2 = network->addConcatenation(cat, 2);
		debug_print(output2->getOutput(0), "output2");
		return output2->getOutput(0);
	}
}