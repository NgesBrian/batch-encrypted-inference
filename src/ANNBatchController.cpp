
/**
 * @brief ANN Batch Controller for managing homomorphic batched ANN operations.
 *
 * This class provides high-level functions for performing batched neural network
 * operations in the encrypted domain using homomorphic encryption (FHE). It 
 * supports operations such as:
 *   - Batched convolution layers
 *   - Batched average and global pooling
 *   - Batched fully connected (linear) layers
 *   - Batched activation functions (e.g., ReLU)
 *
 * The class handles multiple input channels and multiple batches simultaneously,
 * using optimized rotation and packing strategies to minimize the number of
 * homomorphic operations while preserving the correct layout of data.
 *
 * @note
 * - All operations assume ciphertext packing follows a contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Designed for high-throughput FHE-based ANN inference pipelines.
 */

#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
// #include <thread>
#include "ANNBatchController.h"

namespace fs = std::filesystem;

void ANNBatchController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
}

/**
 * @brief Perform a secure convolution operation on batched encrypted data.
 *
 * This function implements a convolutional layer in the encrypted domain
 * using homomorphic encryption for batched inputs. Given a set of encrypted
 * input feature maps, convolution kernels, and bias terms, it computes the
 * convolution across multiple channels and batches while respecting the
 * specified input dimensions, kernel size, padding, and stride.
 *
 * For each output channel, the function multiplies rotated input ciphertexts
 * by corresponding kernel weights, sums contributions from all input channels,
 * applies the bias, and aligns outputs across batches using rotations and
 * packing strategies. The result is one ciphertext per output channel,
 * encoding all batches contiguously.
 *
 * @param encryptedInputs   Vector of ciphertexts, one per input channel,
 *                          each encoding batched feature maps of size
 *                          (batchSize × inputWidth × inputWidth).
 * @param kernelData        Convolution kernels represented as a 3D vector of
 *                          plaintexts: [outputChannel][inputChannel][kernelWeights].
 * @param baisInputs        Bias terms for each output channel (plaintexts).
 * @param batchSize         Number of batches encoded in the input ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param inputChannels     Number of input channels.
 * @param outputChannels    Number of output channels.
 * @param kernelWidth       Width (and height) of the convolution kernel (assumed square).
 * @param paddingLen        Number of zero-padding slots applied around the input feature map.
 * @param stridingLen       Stride length used for the convolution operation.
 *
 * @return vector<Ctext>    Vector of ciphertexts containing the encrypted results
 *                          of the convolution layer, one ciphertext per output channel,
 *                          with batch data packed contiguously:
 *                          [batch0 output channel_i][batch1 output channel_i]...
 *
 * @note
 * - The function assumes ciphertext packing follows contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Optimized rotations and multi-channel processing are used to reduce
 *   the number of homomorphic operations, improving performance for FHE-based CNNs.
 */

vector <int> ANNBatchController::generate_convolution_batch_rotation_positions(int batchSize, int inputWidth,  int kernelSize, int paddingSize, int StrideLen){
        
    vector<int> keys_position;
    int inputWidth_sq = pow(inputWidth, 2);
    int padded_width = inputWidth+(2*paddingSize);
    int width_out = ((padded_width - (kernelSize - 1) - 1)/StrideLen)+1;
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(inputWidth);
    keys_position.push_back(-inputWidth);
    keys_position.push_back(-1);
     keys_position.push_back(1);

    /** Convolution rotations */
    // for(int i=1; i < kernelSize;i++){
    //     keys_position.push_back(i);
    // }

    // int shift = inputWidth - width_out;
    // keys_position.push_back(shift);

    // shift = (inputWidth_sq - width_out_sq);
    // keys_position.push_back(shift);

    // for(int i=1; i<width_out; i++){
    //     shift = (i*width_out);
    //     keys_position.push_back(-shift);
    // }

    // for(int b=0; b<batchSize; b++){
    //     shift = (b*width_out_sq);
    //     keys_position.push_back(-shift);
    // }

    if(StrideLen > 1){
        for (int s=1; s<log2(width_out); s++) {
            keys_position.push_back( pow(2, s-1));
        }
        keys_position.push_back(pow(2, log2(width_out)-1));
        int rotAmount = (StrideLen * inputWidth - width_out);
        keys_position.push_back(rotAmount);

        int shift = (inputWidth_sq - width_out_sq)* ((batchSize / StrideLen) - 1);
        keys_position.push_back(-shift);

        shift = -(inputWidth_sq - width_out_sq);
        keys_position.push_back(shift);

        shift = (inputWidth_sq - width_out_sq);
        keys_position.push_back(shift);

        int rotateAmount = - batchSize * width_out_sq;
        keys_position.push_back(rotateAmount);
    }

    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}


vector <int> ANNBatchController::generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelSize, int StrideLen, 
                                bool globalPooling, int rotationIndex){
    
    vector<int> keys_position;
    if(globalPooling){
        keys_position.push_back(kernelSize*kernelSize);
        keys_position.push_back((inputWidth*inputWidth));
        for(int pos=0; pos<batchSize; pos+=rotationIndex){
            keys_position.push_back(-pos);
        }
        for(int i=1; i<=rotationIndex; i++){
            keys_position.push_back(i);
            keys_position.push_back(-i);
        }
        
        return keys_position;
    }

    int width_avgpool_out = (inputWidth/StrideLen);
    int width_avgpool_sq = pow(width_avgpool_out, 2); 
    int width_sq = pow(inputWidth, 2);
    keys_position.push_back(inputWidth);
    keys_position.push_back(width_sq);

    for(int i=1; i < kernelSize;i++){
        keys_position.push_back(i);
    }

    if(inputWidth<=2){
        return keys_position;
    }

    for (int s=1; s<log2(width_avgpool_out); s++) {
        keys_position.push_back( pow(2, s-1));
    }

    keys_position.push_back(pow(2, log2(width_avgpool_out)-1));

    int shift = ((StrideLen * inputWidth) - width_avgpool_out);
    keys_position.push_back(shift);

    shift = width_sq - width_avgpool_sq;
    keys_position.push_back(shift);
    
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());

    return keys_position;
}

// vector <int> ANNBatchController::generate_fullyconnected_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationPositions){
//     vector<int> keys_position;
//     for(size_t i=0; i<inputSizes.size(); i++){
//         keys_position.push_back(inputSizes[i]);
//         keys_position.push_back(-inputSizes[i]);
//         for(int counter=0; counter<inputSizes[i]; counter+=rotationPositions){
//             keys_position.push_back(-counter);
//             // keys_position.push_back((counter));
//         }
//     }

//     for(int i=1; i<=rotationPositions; i++){
//         keys_position.push_back(i);
//         keys_position.push_back(-i);
//     }

//     for(size_t i=0; i<outputSizes.size(); i++){
//         keys_position.push_back(outputSizes[i]);
//         for(int b=1; b<batchSize; b++){
//             keys_position.push_back((-b*outputSizes[i]));
//         }
//     }
    
//     std::sort(keys_position.begin(), keys_position.end());
//     auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
//     new_end = std::unique(keys_position.begin(), keys_position.end());
//     unique(keys_position.begin(), keys_position.end());
//     keys_position.erase(new_end, keys_position.end());
//     std::sort(keys_position.begin(), keys_position.end());
//     return keys_position;
// }


vector <int> ANNBatchController::generate_fullyconnected_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex){
    
    vector<int> keys_position;
    // int chunkSize = 128;
    for(size_t k =0; k<inputSizes.size(); k++){
        keys_position.push_back(inputSizes[k]);

        // if(chunkSize < batchSize){
        //     for(int i=0; i<batchSize/chunkSize; i++){
        //         int rotIndex = chunkSize * inputSizes[k];
        //         keys_position.push_back(rotIndex);
        
        //     }
        // }
    }

    for(size_t i=0; i<outputSizes.size(); i++){
        for(int j=1; j<=outputSizes[i]; j++){
            keys_position.push_back(j);
            keys_position.push_back(-j);

            // int chunkSize = 128;
            // for(int b=0; b<batchSize; b+=chunkSize){
            //     int rot_idx = (b * outputSizes[i]); 
            //     keys_position.push_back(rot_idx);
            // }
        }
         for(int b=1; b<batchSize; b++){
            int rot_idx = (b * outputSizes[i]); 
            int base_idx = (rot_idx / rotationIndex) * rotationIndex;
            int mod_idx = rot_idx % rotationIndex; 
            keys_position.push_back(-base_idx);
            keys_position.push_back(-mod_idx);
        }

    }
   

    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}


vector <int> ANNBatchController::generate_secure_convert_batch_input_rotation_positions(int batchSize, int inputChannels, int inputSize){
    
    vector<int> keys_position;
    int batchKey = inputChannels * pow(inputSize, 2);
    keys_position.push_back(inputSize);
    keys_position.push_back(inputSize*inputSize);

    for(int b=1; b<batchSize; b++){
        int rot_idx = (b * batchKey); 
        int base_idx = (rot_idx / baseIndex) * baseIndex;
        int mod_idx = rot_idx % baseIndex; 
        keys_position.push_back(-base_idx);
        keys_position.push_back(-mod_idx);
    }

    for(int i=1; i<=inputChannels; i++){
        keys_position.push_back(-i);
        keys_position.push_back(i);
    }
    
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}

/**
 * @brief Perform an optimized secure convolution layer evaluation on batched encrypted data.
 *
 * This function computes a convolution layer over multiple input channels for batched
 * data under homomorphic encryption. It is optimized for the special case where
 * stride = 1, kernel size = 3, and padding = 1, but also supports larger strides
 * by applying striding across multiple channels simultaneously (multi-channel approach),
 * improving efficiency for deep FHE-based networks.
 *
 * For each output channel, the function multiplies input ciphertexts by the corresponding
 * plaintext convolution kernels, sums contributions from all input channels, applies
 * bias, and rotates/aggregates results across batches to produce the final output.
 *
 * @param encryptedInputs     Vector of ciphertexts, one per input channel, each encoding
 *                            batched feature maps of size (batchSize × inputWidth × inputWidth).
 * @param kernelData           Convolution kernels, represented as a 3D vector of plaintexts:
 *                             [outputChannel][inputChannel][kernelWeights].
 * @param baisInputs           Bias terms for each output channel (plaintexts).
 * @param batchSize            Number of batches encoded in the input ciphertexts.
 * @param inputWidth           Width (and height) of the input feature maps (assumed square).
 * @param inputChannels        Number of input channels.
 * @param outputChannels       Number of output channels.
 * @param stridingLen          Stride length used for the convolution operation.
 *
 * @return vector<Ctext>       Vector of ciphertexts containing the encrypted results
 *                            of the convolution layer, one ciphertext per output channel,
 *                            with batch data packed contiguously:
 *                            [batch0 output channel_i][batch1 output channel_i]...
 *
 * @note
 * - This function assumes ciphertext packing follows contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Multi-channel striding is used to reduce the number of rotations and multiplications,
 *   making it efficient for FHE-based CNNs with multiple input channels.
 */
vector<Ctext> ANNBatchController::secure_optimized_batch_convolution(
                FHEController &fheController,vector<Ctext> &encryptedInputs,
                vector<vector<vector<vector<vector<double>>>>> &rawKernel,
                vector<vector<double>> &rawBias,
                int batchSize, int inputWidth, int inputChannels,
                int outputChannels, int stridingLen) {

    constexpr int kernelSize = 9;
    int outputWidth = inputWidth / stridingLen;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec;
    if (stridingLen > 1) {
        cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    }

    int numThreads = std::min(inputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);

    // Precompute rotations (same as before)
    vector<vector<Ctext>> batched_rotated_ciphertexts(inputChannels);
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            vector<Ctext> rotated_ciphertexts;
            Ctext encryptedInput = encryptedInputs[i]->Clone();
            auto digits = context->EvalFastRotationPrecompute(encryptedInput);

            Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
            Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
            rotated_ciphertexts.push_back(context->EvalRotate(first_shot, -inputWidth));
            rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
            rotated_ciphertexts.push_back(context->EvalRotate(second_shot, -inputWidth));
            rotated_ciphertexts.push_back(first_shot);
            rotated_ciphertexts.push_back(encryptedInput);
            rotated_ciphertexts.push_back(second_shot);
            rotated_ciphertexts.push_back(context->EvalRotate(first_shot, inputWidth));
            rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
            rotated_ciphertexts.push_back(context->EvalRotate(second_shot, inputWidth));
            batched_rotated_ciphertexts[i] = rotated_ciphertexts;
            rotated_ciphertexts.clear();
        }
    };

    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = min(start + block, inputChannels);
        threads[t] = thread(in_worker, start, end);
    }
    for (auto &th : threads) th.join();

    // Allocate outputs
    int outnumThreads = std::min(outputChannels, (int)thread::hardware_concurrency());
    vector<thread> outThreads(outnumThreads);
    vector<Ctext> output_ciphertexts(outputChannels);
    auto worker = [&](int start, int end) {
        for (int outCh = start; outCh < end; outCh++) {
            vector<Ctext> batched_results(inputChannels);
            vector<Ctext> mult_results(kernelSize);

            for (int inCh = 0; inCh < inputChannels; inCh++) {
                auto encodeKernel = fheController.optimized_encode_kernel(rawKernel[outCh][inCh], inputSize);
                for (int k = 0; k < kernelSize; k++) {
                    mult_results[k] = context->EvalMult(batched_rotated_ciphertexts[inCh][k], encodeKernel[k]);
                }
                batched_results[inCh] = context->EvalAddMany(mult_results);
            }

            // Sum input-channel results
            Ctext result = context->EvalAddMany(batched_results);

            if (stridingLen > 1) {
                result = ANNController::generalized_downsample_with_channels(result, inputWidth, stridingLen, batchSize);
                result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }

            auto inbiasEncoded = context->MakeCKKSPackedPlaintext(rawBias[outCh], 1, result->GetLevel(), nullptr, outVecSize);
            output_ciphertexts[outCh] = context->EvalAdd(result, inbiasEncoded);
            batched_results.clear();
        }
    };

    // Divide output channels among threads
    block = (outputChannels + outnumThreads - 1) / outnumThreads;
    for (int t = 0; t < outnumThreads; t++) {
        int start = t * block;
        int end = min(start + block, outputChannels);
        outThreads[t] = thread(worker, start, end);
    }
    for (auto &thout : outThreads) thout.join();

    batched_rotated_ciphertexts.clear();
    return output_ciphertexts;
}

vector<Ctext>  ANNBatchController::secure_optimized_batch_shortcut_convolution(
                            FHEController &fheController, vector<Ctext>& encryptedInputs, 
                            vector<vector<vector<double>>>& rawKernel, vector<vector<double>>& rawbiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stridingLen){
    
    int outputWidth = inputWidth / stridingLen;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    // int encodeLevel = encryptedInputs[0]->GetLevel();
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    vector<Ctext> output_ciphertexts(outputChannels); 
 
    int numThreads = min(outputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);
           
    // Lambda for per-thread work
    auto worker = [&](int start, int end) {
        for (int outCh = start; outCh < end; outCh++) {
            vector<Ctext> batched_results(inputChannels);
            for(int inCh=0; inCh<inputChannels; inCh++){
                auto encodeWeights = fheController.encode_inputData(rawKernel[outCh][inCh], inVecSize, encryptedInputs[inCh]->GetLevel());
                batched_results[inCh] = context->EvalMult(encryptedInputs[inCh], encodeWeights);
            }
            
            Ctext result = context->EvalAddMany(batched_results);
            result = ANNController::generalized_downsample_with_channels(result, inputWidth, stridingLen, batchSize);
            result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));

            auto biasVectorEncoded = fheController.encode_inputData(rawbiasData[outCh], outVecSize, result->GetLevel());
            output_ciphertexts[outCh] = context->EvalAdd(result, biasVectorEncoded);
            batched_results.clear();
        }
    };

     // Divide output channels among threads
    int block = (outputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = min(start + block, outputChannels);
        threads[t] = thread(worker, start, end);
    }

    for (auto &th : threads) th.join();

    return output_ciphertexts;
}


/**
 * @brief Securely sum two sets of encrypted inputs channel-wise.
 *
 * This function takes two vectors of ciphertexts, each representing encrypted 
 * input data across multiple channels, and performs homomorphic addition on 
 * corresponding elements. The result is a new vector of ciphertexts where each 
 * ciphertext is the sum of the two inputs for that channel.
 *
 * @param first_encryptedInputs   Vector of ciphertexts for the first input, 
 *                                sized [inputChannels].
 * @param second_encryptedInputs  Vector of ciphertexts for the second input, 
 *                                sized [inputChannels].
 * @param inputChannels           Number of channels (i.e., number of ciphertexts 
 *                                in each input vector).
 *
 * @return A vector of ciphertexts where each element is the homomorphic sum of 
 *         the corresponding inputs from the two provided vectors.
 */

// vector<Ctext> ANNBatchController::secure_batch_sumTwoCiphers(
//     vector<Ctext>& first_encryptedInputs,
//     vector<Ctext>& second_encryptedInputs,
//     int inputChannels
// ) {
//     cout << "[DEBUG] first_encryptedInputs.size(): " << first_encryptedInputs.size()
//          << ", second_encryptedInputs.size(): " << second_encryptedInputs.size()
//          << ", inputChannels: " << inputChannels << endl;

//     size_t n = min(first_encryptedInputs.size(), second_encryptedInputs.size());
//     vector<Ctext> summed_ciphertexts(n);

//     for (size_t i = 0; i < n; i++) {
//         summed_ciphertexts[i] = context->EvalAdd(first_encryptedInputs[i], second_encryptedInputs[i]);
//     }

//     return summed_ciphertexts;
// }


vector<Ctext> ANNBatchController::secure_batch_sumTwoCiphers(
    vector<Ctext>& first_encryptedInputs,
    vector<Ctext>& second_encryptedInputs,
    int inputChannels
) {

    vector<Ctext> summed_ciphertexts(inputChannels);

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(inputChannels, hwThreads > 0 ? hwThreads : 4);
    int block = (inputChannels + numThreads - 1) / numThreads;

    vector<thread> threads;
    threads.reserve(numThreads);
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            summed_ciphertexts[i] = context->EvalAdd(first_encryptedInputs[i], second_encryptedInputs[i]);
        }
    };

    for (int t = 0; t < numThreads; ++t) {
        int start = t * block;
        int end = min(start + block, inputChannels);
        if (start < end){
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &th : threads) th.join();

    return summed_ciphertexts;
}


vector<Ctext> ANNBatchController::secure_batch_globalAvgPool(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, 
                        int inputChannels, int kernelSize, int rotatePositions){
    
    int inputSize = inputWidth*inputWidth;
    // auto reducingMask =  context->MakeCKKSPackedPlaintext(generate_scale_mask(inputSize, batchSize), 1, encryptedInputs[0]->GetLevel());
    vector<double> reducingMask = generate_scale_mask(inputSize, batchSize);
    vector<Ctext> pooled_ciphertexts(inputChannels);

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(inputChannels, hwThreads > 0 ? hwThreads : 4);
    // Only create real threads (never invalid placeholders)

    // threads.reserve(numThreads);
    auto worker = [&](int start, int end) {
        for(int i = start; i< end; i++){
            vector<Ctext> result_ciphers;
            vector<Ctext> batched_ciphertexts;
            int rotation_index = 0;
            int j = 0;
            Ctext encryptedInput = encryptedInputs[i]->Clone();
            for(int b=0; b<batchSize; b++){
                if(b != 0){
                    encryptedInput = context->EvalRotate(encryptedInput, inputSize);
                }
                result_ciphers.push_back(context->EvalSum(encryptedInput, inputSize));
                // cout << "inputchannels: " << i << " batchSize: "<< b << endl; 
                /** check whether is equal to imgcols, merge them and rotate by imgCols. 
                * If i is equal to the outputSize, merge and rotate by imgCols */
                if(j == (rotatePositions-1) || b == (batchSize-1)){
                    // cout << "rotatePositions: "<< rotation_index << endl; 
                    Ctext merged = context->EvalMerge(result_ciphers);
                    if(rotation_index > 0){
                        merged = context->EvalRotate(merged, -rotation_index);
                        batched_ciphertexts.push_back(merged);
                    }
                    else{
                        batched_ciphertexts.push_back(merged); 
                    }
                    rotation_index += rotatePositions;
                    result_ciphers.clear();
                    j=0;
                }
                else{
                    j+=1;
                }
            }
            pooled_ciphertexts[i] = context->EvalMult(context->EvalAddMany(batched_ciphertexts), context->MakeCKKSPackedPlaintext(reducingMask, 1, encryptedInput->GetLevel()));
            batched_ciphertexts.clear();
        }
    };

    vector<thread> threads(numThreads);
    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, inputChannels);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto &th : threads) th.join();
    return pooled_ciphertexts; 
}


//  Ctext ANNBatchController::secure_batch_flinear(
//     Ctext& encryptedInput, vector<Ptext>& weightMatrix,
//     Ptext& baisInput, int batchSize, int inputSize,
//     int outputSize, int rotatePositions){
    
//     int totalSize = batchSize*inputSize;
//     vector<Ctext> result_matrix;
//     vector<vector<Ctext>> batched_ciphertexts(batchSize);
//     vector<double> cleaningInVec = generate_mixed_mask(inputSize, totalSize);
    
//     int j = 0;
//     int rotation_index = 0;
//     for(int i=0; i<outputSize; i++){
//         // For each batch element, rotate base and sum -> push into that batch's vector
//         Ctext temp = context->EvalMult(encryptedInput, weightMatrix[i]);
//         for (int b = 0; b < batchSize; b++) {
//             if (b != 0) {
//                 temp = context->EvalRotate(temp, inputSize);
//             }
//             Ctext summed = context->EvalSum(temp, inputSize);
//             // Ctext summed = context->EvalSum(context->EvalMult(temp, context->MakeCKKSPackedPlaintext(cleaningInVec, 1, temp->GetLevel())), inputSize);
//             batched_ciphertexts[b].push_back(summed);
//         }
//         // cout << "outputSize: " << i << " -- Rotate: " << inputSize << endl;

//         if ((j == rotatePositions - 1) || (i == outputSize - 1)) {
//             for (int b = 0; b < batchSize; b++) {
//                 // cout << "batched_ciphertexts[" << b << "] size: " << batched_ciphertexts[b].size() << " --batch Rotate: " << -(b * outputSize) << " --rotation_index "<< rotation_index << endl;
//                 Ctext inn_ciphertext = context->EvalMerge(batched_ciphertexts[b]);
//                 // cout << "After Merge" << endl; 
//                 // Apply rotations to place merged block into the correct global slot
//                 if (b != 0) {
//                     // cout << "b " << b << ": " << (b * outputSize) << endl;
//                     inn_ciphertext = context->EvalRotate(inn_ciphertext, -(b * outputSize));
//                 }
//                 if (rotation_index != 0) {
//                     inn_ciphertext = context->EvalRotate(inn_ciphertext, -rotation_index);
//                 }
//                 result_matrix.push_back(inn_ciphertext);
//                 batched_ciphertexts[b].clear();
//             }
//             if(j == rotatePositions - 1){
//                 rotation_index += rotatePositions;
//             }
//             j = 0;
//         } else {
//             j++;
//         }
//     } 
//     // Combine results and add bias
//     Ctext with_bias = context->EvalAdd(context->EvalAddMany(result_matrix), baisInput);
//     result_matrix.clear();
//     batched_ciphertexts.clear();

//     return with_bias;
// }

Ctext ANNBatchController::secure_batch_flinear(
    Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize,
    int outputSize, int rotationIndex){
    
    vector<Ctext> result_matrix(batchSize);
    vector<vector<Ctext>> batched_ciphertexts(batchSize, vector<Ctext>(outputSize));

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(outputSize, hwThreads > 0 ? hwThreads : 4);
    auto worker = [&](int start, int end) {
        for(int i = start; i< end; i++){
        // For each batch element, rotate base and sum -> push into that batch's vector
            Ctext rotated_temp = context->EvalMult(encryptedInput, weightMatrix[i]);
            for (int b = 0; b < batchSize; b++) {
                if (b != 0) {
                    rotated_temp = context->EvalRotate(rotated_temp, inputSize);
                }
                batched_ciphertexts[b][i] = context->EvalSum(rotated_temp, inputSize);
            }
        }
    };
    
    vector<thread> threads(numThreads);
    int block = (outputSize + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, outputSize);
        threads[t] = std::thread(worker, start, end);
    }
    for (auto &th : threads) th.join();
    
    // for(int i = 0; i< outputSize; i++){
    // // For each batch element, rotate base and sum -> push into that batch's vector
    //     Ctext temp = context->EvalMult(encryptedInput, weightMatrix[i]);
    //     // cout << "Index: " << i << endl; 
    //     auto worker = [&](int start, int end) {
    //         for (int b = start; b < end; b++) {
    //             Ctext in_ciphertext = temp;
    //             if (b != 0) {
    //                 int rotPosition = b*inputSize;
    //                 int base_idx = (rotPosition / baseIndex) * baseIndex;
    //                 int mod_idx = rotPosition % baseIndex;
                    
    //                 if(base_idx > 0){
    //                     // cout << "Base rotation" << endl; 
    //                     in_ciphertext = context->EvalRotate(in_ciphertext, -base_idx);
    //                 }
    //                 if(mod_idx > 0){
    //                     // cout << "Mod Rotation" << endl; 
    //                     in_ciphertext = context->EvalRotate(in_ciphertext, -mod_idx);
    //                 }
    //             }
    //             batched_ciphertexts[b][i] = context->EvalSum(in_ciphertext, inputSize);
    //         }
    //     };
    //     vector<thread> threads(numThreads);
    //     int block = (batchSize + numThreads - 1) / numThreads;
    //     for (int t = 0; t < numThreads; t++) {
    //         int start = t * block;
    //         int end = std::min(start + block, batchSize);
    //         threads[t] = std::thread(worker, start, end);
    //     }
    //     for (auto &th : threads) th.join();
    // }

    // cout << "Inputs rows all calculated" << endl; 
    int numBatchThreads = min(batchSize, hwThreads > 0 ? hwThreads : 4);

    auto batchworker = [&](int bstart, int bend) {
        for(int b = bstart; b < bend; b++){
            // cout << "batched_ciphertexts[" << b << "] size: " << batched_ciphertexts[b].size() << " --batch Rotate: " << -(b * outputSize) << " --rotation_index "<< rotation_index << endl;
            Ctext inn_ciphertext = context->EvalMerge(batched_ciphertexts[b]);
            // cout << "After Merge" << endl; 
            // Apply rotations to place merged block into the correct global slot
            if (b != 0){ 
                int rot_idx = (b * outputSize); 
                int base_idx = (rot_idx / rotationIndex) * rotationIndex;
                int mod_idx = rot_idx % rotationIndex; 
                //  cout << "Rot Indx: " << rot_idx << " -- Base Indx: " << base_idx << " -- Mod Indx: " << mod_idx << endl; 
                if(base_idx > 0){
                    // cout << "Base rotation" << endl; 
                    inn_ciphertext = context->EvalRotate(inn_ciphertext, -base_idx);
                }
                if(mod_idx > 0){
                    // cout << "Mod Rotation" << endl; 
                    inn_ciphertext = context->EvalRotate(inn_ciphertext, -mod_idx);
                }
            }
            result_matrix[b] = inn_ciphertext;
            // batched_ciphertexts[b].clear();
        }
    };

    vector<thread> bthreads(numBatchThreads);
    int bblock = (batchSize + numBatchThreads - 1) / numBatchThreads;
    for (int t = 0; t < numBatchThreads; t++) {
        int bstart = t * bblock;
        int bend = std::min(bstart + bblock, batchSize);
        bthreads[t] = std::thread(batchworker, bstart, bend);
    }
    for (auto &th : bthreads) th.join();

    // Combine results and add bias
    Ctext fc_results = context->EvalAdd(context->EvalAddMany(result_matrix), baisInput);
    result_matrix.clear();
    batched_ciphertexts.clear();
    return fc_results;
}


vector<Ctext> ANNBatchController::secure_batch_flinear_multiple_outputs(
    Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize, int outputSize){
    
    vector<vector<Ctext>> batched_ciphertexts(batchSize, vector<Ctext>(outputSize));

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(outputSize, hwThreads > 0 ? hwThreads : 4);
    auto worker = [&](int start, int end) {
        for(int i = start; i< end; i++){
            Ctext temp = context->EvalMult(encryptedInput, weightMatrix[i]);
            for (int b = 0; b < batchSize; b++) {
                if (b != 0) {
                    temp = context->EvalRotate(temp, inputSize);
                }
                batched_ciphertexts[b][i] = context->EvalSum(temp, inputSize);
            }
        }
    };
    
    vector<thread> threads(numThreads);
    int block = (outputSize + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, outputSize);
        threads[t] = std::thread(worker, start, end);
    }
    for (auto &th : threads) th.join();


    weightMatrix.clear();
    weightMatrix.shrink_to_fit();

    
    vector<Ctext> result_matrix(batchSize);
    int numBatchThreads = min(batchSize, hwThreads > 0 ? hwThreads : 4);
    auto batchworker = [&](int bstart, int bend) {
        for(int b = bstart; b < bend; b++){
            result_matrix[b] = context->EvalAdd(context->EvalMerge(batched_ciphertexts[b]), baisInput);
        }
    };

    vector<thread> bthreads(numBatchThreads);
    int bblock = (batchSize + numBatchThreads - 1) / numBatchThreads;
    for (int t = 0; t < numBatchThreads; t++) {
        int bstart = t * bblock;
        int bend = std::min(bstart + bblock, batchSize);
        bthreads[t] = std::thread(batchworker, bstart, bend);
    }
    for (auto &th : bthreads) th.join();

    // Combine results and add bias
    batched_ciphertexts.clear();
    return result_matrix;
}



// vector<Ctext> ANNBatchController::secure_batch_flinear_multiple_outputs(
    
//     Ctext& encryptedInput, vector<Ptext>& weightMatrix,
//     Ptext& baisInput, int batchSize, 
//     int inputSize, int outputSize){
    
//     vector<Ctext> result_matrix(batchSize);

//     int hwThreads = thread::hardware_concurrency();
//     int numThreads = min(outputSize, hwThreads > 0 ? hwThreads : 4);

//     // Process batches in smaller chunks to reduce memory usage
//     int chunkSize = 128; // Process 128 batches at a time
//     for (int chunkStart = 0; chunkStart < batchSize; chunkStart += chunkSize) {
//         int chunkEnd = std::min(chunkStart + chunkSize, batchSize);
//         int currentChunkSize = chunkEnd - chunkStart;

//         vector<vector<Ctext>> batched_ciphertexts(currentChunkSize, vector<Ctext>(outputSize));
//         auto worker = [&](int start, int end) {
//             for (int i = start; i < end; i++) {
//                 int idxW = chunkStart +i;
//                 Ctext temp = context->EvalMult(encryptedInput, weightMatrix[idxW]);
//                 if( chunkStart != 0 ){
//                     // Rotate to the start of the current chunk
//                     temp = context->EvalRotate(temp, chunkStart * inputSize);
//                 }
//                 for (int b = 0; b < currentChunkSize; b++) {
//                     if (b != 0) {
//                         temp = context->EvalRotate(temp, inputSize);
//                     }
//                     batched_ciphertexts[b][i] = context->EvalSum(temp, inputSize);
//                 }
//             }
//         };

//         vector<thread> threads(numThreads);
//         int block = (outputSize + numThreads - 1) / numThreads;
//         for (int t = 0; t < numThreads; t++) {
//             int start = t * block;
//             int end = std::min(start + block, outputSize);
//             threads[t] = std::thread(worker, start, end);
//         }
//         for (auto &th : threads) th.join();

//         cout << "Inputs rows for chunk [" << chunkStart << ", " << chunkEnd << ") all calculated" << endl;

//         auto batchworker = [&](int bstart, int bend) {
//             for (int b = bstart; b < bend; b++) {
//                 int indexInResult = chunkStart + b;
//                 result_matrix[indexInResult] =  context->EvalAdd(context->EvalMerge(batched_ciphertexts[b]), baisInput);
//             }
//         };

//         int numBatchThreads = min(currentChunkSize, hwThreads > 0 ? hwThreads : 4);
//         vector<thread> bthreads(numBatchThreads);
//         int bblock = (currentChunkSize + numBatchThreads - 1) / numBatchThreads;
//         for (int t = 0; t < numBatchThreads; t++) {
//             int bstart = t * bblock + chunkStart;
//             int bend = std::min(bstart + bblock, chunkEnd);
//             bthreads[t] = std::thread(batchworker, bstart, bend);
//         }
//         for (auto &th : bthreads) th.join();

//         batched_ciphertexts.clear(); // Clear memory for this chunk
//     }

//     return result_matrix;
// }


vector<Ctext> ANNBatchController::secure_batch_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues,  int inputChannels, int vectorSize, int polyDegree) {
    double lowerBound = -1;
    double upperBound = 1;
    
    auto encryptInn = encryptedInputs;
    // int scaleVal = 110;
    vector<Ctext> relu_results(inputChannels);
    int totalElements =  nextPowerOf2(vectorSize);
    int numThreads = std::min(inputChannels, (int)std::thread::hardware_concurrency());
    std::vector<std::thread> threads(numThreads);

    // Lambda for per-thread work
    auto worker = [&](int start, int end) {
        for(int i=start; i<end; i++){
            int scaleVal = scaleValues[i];
            if(scaleVal <= 1){
                encryptInn[i] = encryptedInputs[i];
                scaleVal = 1;
            }
            else{
                scaleVal = 2*scaleVal;
                // auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleVal, vectorSize), 1, 1, nullptr, totalElements);
                auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleVal, vectorSize), 1, encryptedInputs[i]->GetLevel(), nullptr, totalElements);
                encryptInn[i] = context->EvalMult(encryptedInputs[i], mask_data);
            }
            relu_results[i] = context->EvalChebyshevFunction(
                            [scaleVal](double x) -> double { if (x < 0) return 0; else return scaleVal*x; }, 
                                                encryptInn[i],
                                                lowerBound,
                                                upperBound, 
                                                polyDegree);
        }
    };

     // Divide output channels among threads
    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, inputChannels);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto &th : threads) th.join();

    return relu_results;
}

/**
 * @brief Convert channel-separated encrypted inputs into a single batched ciphertext.
 *
 * This function takes encrypted inputs where each ciphertext corresponds to one input channel. 
 * It aligns all channel data together within a batch, producing a single ciphertext that 
 * represents all inputs across channels and batches. This transformation is necessary 
 * when passing data into fully connected layers, which expect a flat neuron-style input 
 * rather than channel-separated structures (as in convolutional layers).
 *
 * @param encryptedInputs A vector of ciphertexts, where each ciphertext corresponds to one input channel.
 * @param batchSize Number of batches to process.
 * @param inputChannels Number of input channels per batch.
 * @param inputSize Size of each input channel (number of slots used per channel).
 *
 * @return A single ciphertext that encodes the entire batch across all channels, 
 *         aligned in a flat neuron-compatible format.
 *
 * @note 
 * - Internally, the function:
 *   - Uses a cleaning mask to isolate values in each channel.
 *   - Applies rotations to align channel data.
 *   - Aggregates all channel ciphertexts for each batch.
 *   - Applies additional rotations to stack batches correctly.
 * - This ensures compatibility between convolutional outputs (channel-based) 
 *   and fully connected layers (neuron-based).
 */
Ctext ANNBatchController::secure_convert_batch_input(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth) {
    
    int inputSize = pow(inputWidth, 2); 
    int batchKey = (inputChannels * inputSize);
    // int inVecSize = batchSize * inputSize; // total packed vector size
    // auto cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, inVecSize), 1, encryptedInputs[0]->GetLevel());
    vector<Ctext> batch_ciphertexts(batchSize);
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < inputChannels; i++) {
            if(b != 0){
                encryptedInputs[i] = context->EvalRotate(encryptedInputs[i], inputSize);
            }
        }
        
        // Combine all channels for this batch into one ciphertext
        Ctext combinedChannels = context->EvalMerge(encryptedInputs);
        if (b != 0) {
            int rot_idx = (b * batchKey);
            int base_idx = (rot_idx / baseIndex) * baseIndex;
            int mod_idx = rot_idx % baseIndex; 
            if(base_idx != 0){
                combinedChannels = context->EvalRotate(combinedChannels, -base_idx);
            }
            if(mod_idx !=0){
                combinedChannels = context->EvalRotate(combinedChannels, -mod_idx);
            }
        }
        batch_ciphertexts[b] = combinedChannels;
    }

    // Combine all batch ciphertexts into the final output
    Ctext finalCipher = context->EvalAddMany(batch_ciphertexts);
    batch_ciphertexts.clear();
    return finalCipher;
}


/**
 * @brief Generate a zero mask of given size.
 *
 * @param size Number of elements in the mask.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with all zeros.
 */
Ptext ANNBatchController::gen_zero_mask(int size, int level) {
    vector<double> mask(size, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}


/**
 * @brief Generate a mask selecting a specific row in every channel.
 *
 * @param row Row index to select.
 * @param width Width of each channel.
 * @param inputSize Number of elements per channel.
 * @param stride Unused here but kept for consistency.
 * @param numChannels Total number of channels.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with the row selected in all channels.
 */
Ptext ANNBatchController::gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level) {
    
    vector<double> baseMask;
    for (int j = 0; j < (row * width); j++) {
        baseMask.push_back(0);
    }
    for (int j = 0; j < width; j++) {
        baseMask.push_back(1);
    }
    for (int j = 0; j < (inputSize - width - (row * width)); j++) {
        baseMask.push_back(0);
    }

    // repeat baseMask n times
    vector<double> mask;
    mask.reserve(baseMask.size() * batchSize);
    for (int i = 0; i < batchSize; i++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }

    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate a mask selecting a specific channel while zeroing all others.
 *
 * @param channel Channel index to select.
 * @param outputSize Number of elements per channel.
 * @param numChannels Total number of channels.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with the selected channel set to 1.
 */
Ptext ANNBatchController::gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level ){
    
    int totalSlots = outputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);

    int pos = channel * outputSize;
    for (int i = 0; i < outputSize; i++) {
        mask[pos + i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}
