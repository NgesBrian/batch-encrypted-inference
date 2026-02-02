
#ifndef FHEON_ANNBATCHCONCROLLER_H
#define FHEON_ANNBATCHCONCROLLER_H

#include <openfhe.h>
#include <thread>
// #include <cereal/types/polymorphic.hpp> // Include this header

#include "./FHEController.h"
#include "./ANNController.h"

#include "Utils.h"
#include "UtilsData.h"
#include "UtilsBatchData.h"

using namespace lbcrypto;
using namespace std;

/** secure_anncontroller defined utils */
using namespace utils;
using namespace utilsdata;
using namespace utilsbatchdata;

class ANNBatchController : public ANNController {

private:
    CryptoContext<DCRTPoly> context;

public:
    int num_slots = 1 << 14;
    int baseIndex = 1024;

    // Constructor must forward CryptoContext to the base class
    ANNBatchController(CryptoContext<DCRTPoly> ctx)
        : ANNController(ctx), context(ctx) {}
    
    void setContext(CryptoContext<DCRTPoly>& in_context);
    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }

    vector <int> generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelSize, int paddingSize=0, int StrideLen=1);
    vector <int> generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelSize, int StrideLen=2, 
                            bool globalPooling=false, int rotationIndex=16);
    vector <int> generate_fullyconnected_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex=100);
    vector <int> generate_secure_convert_batch_input_rotation_positions(int batchSize, int inputChannels, int inputWidth);
   

    vector<Ctext>   secure_batch_convolution(vector<Ctext>& encryptedInput, vector<vector<vector<Ptext>>>& kernelData, vector<Ptext>& baisInputs,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int paddingSize=0, int stridingLen=1);
    vector<Ctext>   secure_optimized_batch_convolution(vector<Ctext>& encryptedInputs, vector<vector<vector<Ptext>>>& kernelData, vector<Ptext>& baisInputs,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stridingLen=1);
    vector<Ctext>   secure_optimized_batch_shortcut_convolution(vector<Ctext>& encryptedInputs, vector<vector<Ptext>>& kernelData, vector<Ptext>& baisInputs,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stridingLen);


    vector<Ctext>  secure_optimized_batch_convolution(FHEController &fheController, vector<Ctext>& encryptedInputs, vector<vector<vector<vector<vector<double>>>>>& rawKernel, vector<vector<double>>& rawBias,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stridingLen);
    vector<Ctext>  secure_optimized_batch_shortcut_convolution(FHEController &fheController, vector<Ctext>& encryptedInputs, vector<vector<vector<double>>>& rawKernel, vector<vector<double>>& rawbiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stridingLen);


    vector<Ctext> secure_optimzed_batch_avgPool(vector<Ctext>& encryptedInputs,  int batchSize, int inputWidth, int inputChannels, int kernelWidth, int strideLen=2);
    vector<Ctext> secure_batch_globalAvgPool(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, int inputChannels, int kernelSize, int rotatePositions);

    Ctext secure_batch_flinear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int batchSize, int inputSize, int outputSize, int rotatePositions=100);
        vector<Ctext> secure_batch_flinear_multiple_outputs(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int batchSize, int inputSize, int outputSize);
    vector<Ctext> secure_batch_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues, int inputChannels, int vectorSize, int polyDegree=59); 
    Ctext secure_convert_batch_input(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth);
    Ctext corrected_secure_convert_batch_input(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth);
    vector<Ctext> secure_batch_sumTwoCiphers(vector<Ctext>& first_encryptedInputs, vector<Ctext>& second_encryptedInputs, int inputChannels);
    
private:
    Ptext gen_zero_mask(int size, int level); 
    Ptext gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level);
    Ptext gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level );
    

};

#endif // FHEON_ANNBATCHCONCROLLER_H