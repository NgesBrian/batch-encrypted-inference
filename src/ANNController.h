
/*********************************************************************************************************************** 
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
********************************************************************************************************************/

/*******************************************************************************************************************
 * This ANN controller is used to define all ANN layers used in this project such as; 
 * Convolution, avgPool, fclinear
 *******************************************************************************************************************/

#ifndef FHEON_ANNCONCROLLER_H
#define FHEON_ANNCONCROLLER_H

#include <openfhe.h>
#include <thread>
// #include <cereal/types/polymorphic.hpp> // Include this header

#include "./FHEController.h"

#include "Utils.h"
#include "UtilsData.h"

using namespace lbcrypto;
using namespace std;

/** secure_anncontroller defined utils */
using namespace utils;
using namespace utilsdata;

class ANNController{

private:
    CryptoContext<DCRTPoly> context;

public:
    string public_data = "sskeys";
    int num_slots = 1 << 14;
    
    ANNController(CryptoContext<DCRTPoly>& ctx) : context(ctx) {}
    void setContext(CryptoContext<DCRTPoly>& in_context);
    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }
   
    vector<int> generate_convolution_rotation_positions(int inputWidth, int inputChannels, int outputChannels,
                                                    int kernelSize, int paddingSize, int StrideLen);
    vector<int> generate_fullyconnected_rotation_positions(int maxFCLayeroutputs, int rotationPosition);
    vector<int> generate_avgpool_rotation_positions(int inputWidth, int kernelSize, int StrideLen, int inputChannels);
    
    vector<int> generate_optimized_convolution_rotation_positions(int inputWidth,  int inputChannels, 
                                            int outputChannels, int StrideLen = 1, string stridingType="multi_channels");
    vector<int> generate_avgpool_optimized_rotation_positions(int inputWidth,  int inputChannels, 
                                            int kernelSize, int StrideLen, bool globalPooling=false, string stridingType="multi_channels", int rotationIndex=16);

    Ctext secure_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& baisInput,
                            int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int paddingSize=0, int stridingLen=1);
    Ctext secure_advanced_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& baisInput,
                            int inputWidth, int kernelWidth, int paddingSize, int stridingLen, int inputChannelsSize, int outputChannelsSize);
    Ctext secure_optimized_convolution(Ctext& encryptedInput,  vector<vector<Ptext>>& kernelData, Ptext& baisInput, 
                            int inputWidth, int inputChannelsSize, int outputChannelsSize, int StrideLen = 1, int index=0);
    Ctext  secure_optimized_convolution_multi_channels(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& baisInput,  
                            int inputWidth,  int inputChannels, int outputChannels);
    Ctext secure_shortcut_convolution(Ctext& encryptedInput,  vector<Ptext>& kernelData, Ptext& baisInput,  
                            int inputWith, int inputChannelsSize, int outputChannelsSize);
    vector<Ctext> secure_double_optimized_convolution(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, 
                            const vector<Ptext>& shortcutKernelData, Ptext& biasVector,  Ptext& shortcutBiasVector, 
                            int inputWidth, int inputChannels, int outputChannels);
    vector<Ctext>  secure_double_optimized_convolution_multi_channels(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, 
                            const vector<Ptext>& shortcutKernelData, Ptext& baisInput, Ptext& shortcutBiasInput,  
                            int inputWidth,  int inputChannels, int outputChannels);

    Ctext secure_avgPool(Ctext encryptedInput, int imgCols, int outputChannels, int kernelSize=2, int StrideLen=2);
    Ctext secure_advanced_avgPool(Ctext encryptedInput, int inputWidth, int outputChannels, int kernelSize, int stridingLen, int paddingSize);
    Ctext secure_globalAvgPool(Ctext& encryptedInput, int inputWidth, int outputChannels, int kernelSize, int rotatePositions);
    Ctext secure_optimzed_avgPool(Ctext& encryptedInput,  int inputWidth, int outputChannels, int kernelSize, int StrideLen);
    Ctext secure_optimzed_avgPool_multi_channels(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int strideLen);
    
    Ctext secure_flinear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int inputSize, int outputSize, int rotatePositions);
    Ctext secure_optimized_flinear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int inputSize, int outputSize);

    Ctext secure_relu(Ctext& encryptedInput, double scale, int vectorSize, int polyDegree = 59);
    Ctext secure_sumTwoCiphers(Ctext& firstCipher, Ctext& secondCipher); 

protected:
        Ctext generalized_downsample_with_channels(const Ctext& input, int inputWidth, int stride, int numChannels);
private:
    Ctext secure_private_striding(Ctext in_cipher, int inputWidth, int width_out,  int StrideLen);
    Ctext generalized_downsample(const Ctext& input, int inputWidth, int stride);
    // Ctext generalized_downsample_with_channels(const Ctext& input, int inputWidth, int stride, int numChannels);
    Ctext batchChannelConvolution(const vector<Ctext>& rotatedInputs, const vector<Ptext>& kernelData, int kernelSize, int inputSize,  int inputChannels);

     Ptext first_mask(int width, int inputSize, int stride, int level);
    Ptext first_mask_with_channels(int width, int inputSize, int stride, int num_channels, int level);
    
    Ptext gen_binary_mask(int pattern, int inputSize, int stride, int level);
    Ptext gen_binary_mask_with_channels(int pattern, int inputSize, int stride, int num_channels, int level);
   
    Ptext gen_row_mask(int row, int width, int inputSize, int stride, int level);
    Ptext gen_row_mask_with_channels(int row, int width, int inputSize, int stride, int numChannels,int level);

    Ptext gen_zero_mask(int size, int level);
    Ptext gen_zero_mask_channels(int size, int num_channels, int level);
    Ptext gen_channel_full_mask(int n, int in_elements, int out_elements, int num_channels, int level);
    Ptext gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level);

};

#endif // FHEON_ANNCONCROLLER_H