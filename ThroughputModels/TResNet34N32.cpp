
#include <iostream>
#include <sys/stat.h>

#include "../src/FHEController.h"
#include "../src/ANNController.h"
#include "./../src/ANNBatchController.h"

using namespace std;
CryptoContext<DCRTPoly> context;
FHEController fheController(context);

#ifndef DEFAULT_ARG
#define DEFAULT_ARG 250
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

vector<Ctext> shortcut_convolution_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stridingLen);
vector<Ctext> convolution_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &inputdataWidth, int &inputdataSize, int inputChannels, int outputChannels, int stridinglen);
vector<Ctext> resnet_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, int &dataWidth, int &dataSize,
                 int inputChannels, int outputChannels, int reluScale, bool bootstrapState, bool shortcutState);
Ctext fc_layer_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions);

vector<int> measuringTime;
vector<int> intermTime;
auto startIn = get_current_time();
int batchSize = 32;
vector<int> slotsSizes = {15, 15, 15, 15, 15};
int scalingFact = 2; 

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    /*** Generate the context of the project in the FHEController and pass it to the SnnController */
     int ringDegree = 16;
    int numSlots = 15;
    int circuitDepth = 11;
    int dcrtBits = 46;
    int firstMod = 50;
    int digitSize = 4;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
    fheController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, levelBudget, serialize);
    context = fheController.getContext();
    ANNController annController(context);
    ANNBatchController annBatchController(context);
    printDuration(begin_time, "Context Generation and Keys Serialization", false);
    cout << "---------------------------------RESNET34-------------"<< to_string(DEFAULT_ARG) << "--------------------------" << endl; 
    
    /**** Read the CIFAR-10 Images and inference them */
    int img_cols = 32;
    int img_depth = 3;
    int kernelSize = 3; 
    int paddingLen = 1;
    int stridingLen = 1;
    int shortcutStridingLen = 2;
    int avgpoolSize = 4;
    vector<int> channels = {3, 16, 32, 64, 128, 100};
    vector<int> dataWidths = {32, 16, 8, 4, 1};
    vector<int> dataSizes = {1024, 256, 64, 16};
    vector<int> batchSizes = {32, 32, 32};
    int rotPositions = 32;
    int fcRotIndex = 1000;

    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[0], dataWidths[0],  kernelSize, paddingLen, stridingLen);
    auto conv2_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[0], dataWidths[0],  kernelSize, paddingLen, shortcutStridingLen);
    auto conv3_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[1],  kernelSize, paddingLen, stridingLen);
    auto conv4_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[1],  kernelSize, paddingLen, shortcutStridingLen);
    auto conv5_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[2], dataWidths[2],  kernelSize, paddingLen, stridingLen);
    auto conv6_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[2], dataWidths[2],  kernelSize, paddingLen, shortcutStridingLen);
    auto conv7_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[2], dataWidths[3],  kernelSize, paddingLen, stridingLen);
    auto conv8_keys = annBatchController.generate_convolution_batch_rotation_positions(batchSizes[2], dataWidths[3],  kernelSize, paddingLen, shortcutStridingLen);

    auto avg_keys = annBatchController.generate_avgpool_batch_optimized_rotation_positions(batchSizes[2], dataWidths[3], avgpoolSize, avgpoolSize, true, rotPositions);
    auto convert_keys = annBatchController.generate_secure_convert_batch_input_rotation_positions(batchSizes[2],  channels[4], dataWidths[4]);
    auto fc_keys = annBatchController.generate_fullyconnected_batch_rotation_positions(batchSizes[2], {channels[5]}, {channels[4]}, fcRotIndex);
    /************************************************************************************************ */

    vector<vector<int>> rkeys_layer1, rkeys_layer2, rkeys_layer3, rkeys_layer4, convert_layer, fc_layer;
    rkeys_layer1.push_back(conv1_keys);
    rkeys_layer1.push_back(conv2_keys);
    rkeys_layer1.push_back(conv3_keys);
    
    rkeys_layer2.push_back(conv3_keys);
    rkeys_layer2.push_back(conv4_keys);
    rkeys_layer2.push_back(conv5_keys);

    rkeys_layer3.push_back(conv5_keys);
    rkeys_layer3.push_back(conv6_keys);
    rkeys_layer3.push_back(conv7_keys);
    rkeys_layer3.push_back(conv8_keys);

    // rkeys_layer3.push_back(conv5_keys);
    rkeys_layer4.push_back(avg_keys);
    convert_layer.push_back(convert_keys);
    fc_layer.push_back(fc_keys);

    /*** join all keys and generate unique values only */
    /*********************************************** Key Generation ******************************************************************************/
    vector<int> serkeys_layer1 = serialize_rotation_keys(rkeys_layer1); 
    vector<int> serkeys_layer2 = serialize_rotation_keys(rkeys_layer2);
    vector<int> serkeys_layer3 = serialize_rotation_keys(rkeys_layer3);
    vector<int> serkeys_layer4 = serialize_rotation_keys(rkeys_layer4);
    vector<int> serkeys_convert_layer = serialize_rotation_keys(convert_layer);
    vector<int> serkeys_fc_layer = serialize_rotation_keys(fc_layer);
    // /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygen_time = startTime();
    // cout << "This is the rotation positions (" << serkeys_block1.size() <<"+" << serkeys_block2.size() << "+" << serkeys_block3.size() << " = " << total_rkeys << "): " << endl;
    cout << "Layer 1 keys (" << serkeys_layer1.size() << ") " << serkeys_layer1 << endl;
    cout << "Layer 2 keys (" << serkeys_layer2.size() << ") " << serkeys_layer2 << endl;
    cout << "Layer 3 keys (" << serkeys_layer3.size() << ") " << serkeys_layer3 << endl;
    cout << "Layer 4 keys (" << serkeys_layer4.size() << ") " << serkeys_layer4 << endl;
    cout << "Converter keys (" << serkeys_convert_layer.size() << ") " << serkeys_convert_layer << endl;
    cout << "FC Layer keys (" << serkeys_fc_layer.size() << ") " << serkeys_fc_layer << endl;

    fheController.generate_bootstrapping_and_rotation_keys(serkeys_layer1, slotsSizes[0], "layer1.bin", true);
    fheController.clear_context(slotsSizes[0]);
    
    fheController.generate_bootstrapping_and_rotation_keys(serkeys_layer2, slotsSizes[1], "layer2.bin",  true);
    fheController.clear_context(slotsSizes[1]);
    
    fheController.generate_bootstrapping_and_rotation_keys(serkeys_layer3, slotsSizes[2], "layer3.bin", true);
    fheController.clear_context(slotsSizes[2]);

    fheController.generate_bootstrapping_and_rotation_keys(serkeys_layer4, slotsSizes[2], "layer4.bin", true);
    fheController.clear_context(slotsSizes[3]);

    fheController.generate_bootstrapping_and_rotation_keys(serkeys_convert_layer, slotsSizes[2], "convert_layer.bin", true);
    fheController.clear_context(slotsSizes[2]);

    fheController.generate_bootstrapping_and_rotation_keys(serkeys_fc_layer, slotsSizes[3], "fc_layer.bin", true);
    fheController.clear_context(slotsSizes[4]);
    printDuration(begin_rotkeygen_time, "Rotation KeyGen Time", false);
   
    int numImages = DEFAULT_ARG+INDEX_VALUE;
    int dataSize = img_depth*pow(img_cols, 2);
    string cifar100tPath = "./../images/cifar-100-binary/test.bin";
    vector<vector<double>> imagesData = read_images(cifar100tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/TresNet34/fhepredictions.txt", ios_base::app);
    Ctext convData;
    int polyDegee = 59;
    int reluScale = 10;
    int bootstrap_level = 2;

    for (int idx = 0; idx < 1; idx++) {
        
        fheController.clear_context(slotsSizes[4]);
        fheController.load_bootstrapping_and_rotation_keys(slotsSizes[0], "layer1.bin", false);
        int imgIdx = idx*batchSize; 
        vector<vector<double>> batchedImages; 
        for (int b = 0; b < batchSize; b++) {
            auto img = imagesData[imgIdx + b];
            batchedImages.push_back(img);
        }

        /** adjust images */
        auto inputDatas = convert_inputData(batchedImages, batchSize, channels[0], dataSizes[0]);
        vector<Ctext>encryptedInputs;
        for(int i=0; i<channels[0]; i++){
            encryptedInputs.push_back(fheController.encrypt_inputData(inputDatas[i]));
        }
        cout << endl << imgIdx+1  << " to " << imgIdx+batchSize << " - (" << encryptedInputs.size() << " input channes) images Read, Normalized and Encrypt"<< endl;
        
        /************************************************************************************************ */
        auto inference_time = startTime();
        cout<< "Layer 0" << endl;
        auto convData = convolution_block(fheController, annBatchController,  "layer0_conv1", encryptedInputs, batchSize, dataWidths[0], dataSizes[0], channels[0], channels[1], stridingLen);
        int totalSize = batchSize * dataSizes[0];
        auto scalingVals = fheController.read_batch_scalingValues(convData, channels[1], totalSize);
    
        startIn = get_current_time();
        convData = annBatchController.secure_batch_relu(convData, scalingVals, channels[1], totalSize, polyDegee);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        printDuration(inference_time, "run time", false);

        cout<< endl<<  "Layer 1" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer1_block1", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, false, false);
        cout <<"Block 2 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer1_block2", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
        cout <<"Block 3 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer1_block3", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
        // fheController.read_batch_minmaxValue(convData, channels[1], totalSize);
        printDuration(inference_time, "run time", false);

        cout<< endl<< "Layer 2" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer2_block1", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[2], reluScale, true, true);
        totalSize = batchSize * dataSizes[1];
        
        
        fheController.clear_context(slotsSizes[0]);
        fheController.load_bootstrapping_and_rotation_keys(slotsSizes[1], "layer2.bin", false);
        
        
        cout <<"Block 2 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer2_block2", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        cout <<"Block 3 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer2_block3", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        cout <<"Block 4 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer2_block4", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        // fheController.read_batch_minmaxValue(convData, channels[2], totalSize);
        printDuration(inference_time, "run time", false);
        
        cout<< endl<<  "Layer 3" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer3_block1", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[3], reluScale, true, true);
        totalSize = batchSize * dataSizes[2];

        
        fheController.clear_context(slotsSizes[0]);
        fheController.load_bootstrapping_and_rotation_keys(slotsSizes[1], "layer3.bin", false);

        
        cout <<"Block 2 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer3_block2", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 3" << endl;
        convData = resnet_block(fheController, annBatchController, "layer3_block3", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 4" << endl;
        convData = resnet_block(fheController, annBatchController, "layer3_block4", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 5" << endl;
        convData = resnet_block(fheController, annBatchController, "layer3_block5", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 6" << endl;
        convData = resnet_block(fheController, annBatchController, "layer3_block6", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        // fheController.read_batch_minmaxValue(convData, channels[3], totalSize);
        printDuration(inference_time, "run time", false);


        cout<< endl<<  "Layer 4" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer4_block1", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[4], reluScale, true, true);
        totalSize = batchSize * dataSizes[3];
        cout <<"Block 2 " << endl;
        convData = resnet_block(fheController, annBatchController, "layer4_block2", convData, batchSize, dataWidths[3], dataSizes[3], channels[4], channels[4], reluScale, true, false);
        cout <<"Block 3" << endl;
        convData = resnet_block(fheController, annBatchController, "layer4_block3", convData, batchSize, dataWidths[3], dataSizes[3], channels[4], channels[4], reluScale, true, false);
        // fheController.read_batch_minmaxValue(convData, channels[4], totalSize);
        printDuration(inference_time, "run time", false);

        // cout<< "Classification" << endl;
        startIn = get_current_time();
        convData = fheController.batch_bootstrap_function(convData, channels[4], bootstrap_level);
        intermTime.push_back(measureTime(startIn, get_current_time()));

        
        fheController.clear_context(slotsSizes[2]);
        fheController.load_bootstrapping_and_rotation_keys(slotsSizes[3], "layer4.bin", false);

        cout << "Global Pooling" << endl;
        startIn = get_current_time();
        convData = annBatchController.secure_batch_globalAvgPool(convData, batchSize, dataWidths[3], channels[4], avgpoolSize, rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        cout << "Ciphertext Converter..." << endl;
        fheController.clear_context(slotsSizes[2]);
        fheController.load_bootstrapping_and_rotation_keys(slotsSizes[3], "convert_layer.bin", false);
        auto fcData = annBatchController.secure_convert_batch_input(convData, batchSize, channels[4], dataWidths[4]);
        cout << "Converter done" << endl; 


        convData.clear();
        convData.shrink_to_fit();

        cout << "Fully Connected Layer..." << endl;
        fheController.clear_context(slotsSizes[2]);
        fheController.load_bootstrapping_and_rotation_keys(slotsSizes[3], "fc_layer.bin", false);
        fcData = fc_layer_block(fheController, annBatchController, "layer_fc", fcData, batchSize, channels[4], channels[5], fcRotIndex);


        printTimeWithMessage("ResNet34 Circuit : ", measuringTime);
        printTimeWithMessage("ResNet34 Bootsrapping: ", intermTime);
        measuringTime.clear();
        intermTime.clear();

        string infereMessage = "Batch Size ("+ to_string(batchSize) +") -- Total Run Time for Images " + to_string(imgIdx + 1) + " - " +  to_string(imgIdx+1+batchSize);  
        printDuration(inference_time, infereMessage, false);
        auto predictions = fheController.read_batch_inferencedLabel(fcData, batchSize, channels[5], outFile);
        cout << "Batch Predictions: " << predictions << endl;

    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
   return 0;
}

vector<Ctext> shortcut_convolution_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stridingLen){
    
    int outputWidth = dataWidth/stridingLen;
    int outputSize = pow(outputWidth, 2);
    
    /******** DOUBLE CHECK THIS KERNEL ENCODING */
    string dataPath = "./../weights/resnet34/"+layer;
    auto rawKernel = load_shortcut_batch_weights(dataPath+"_shortcut_weight.csv", batchSize, outputChannels, inputChannels, dataSize); 
    auto rawbiasData = load_batch_bias(dataPath+"_shortcut_bias.csv", outputChannels, batchSize, outputSize);

    startIn = get_current_time();
    auto conv_data = annBatchController.secure_optimized_batch_shortcut_convolution(fheController, encrytedInputs, rawKernel, rawbiasData, batchSize, dataWidth, inputChannels, outputChannels, stridingLen);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    rawKernel.clear();
    rawKernel.shrink_to_fit();
    rawbiasData.clear();
    rawbiasData.shrink_to_fit();
    return conv_data;
}

vector<Ctext> convolution_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stridinglen){
     
    int kernelWidth = 3;
    int outputWidth = dataWidth/stridinglen;
    int outputSize = pow(outputWidth, 2);
    string dataPath = "./../weights/resnet34/"+layer;
    auto rawKernel = load_batch_weights(dataPath+"_weight.csv", outputChannels, inputChannels, batchSize, kernelWidth, kernelWidth);
    auto rawbiasData = load_batch_bias(dataPath+"_bias.csv", outputChannels, batchSize, outputSize);

    startIn = get_current_time();
    auto conv_data = annBatchController.secure_optimized_batch_convolution(fheController, encrytedInputs, rawKernel, rawbiasData, batchSize, dataWidth, inputChannels, outputChannels, stridinglen);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
        
    rawKernel.clear();
    rawKernel.shrink_to_fit();
    rawbiasData.clear();
    rawbiasData.shrink_to_fit();

    return conv_data;
}


vector<Ctext> resnet_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, int &inputdataWidth, int &inputdataSize,
                 int inputChannels, int outputChannels, int reluScale, bool bootstrapState, bool shortcutState){

    int stridingLen = 1;
    int shortcutStridindLen = 2; 
    int polyDegee = 59; 
    int bootstrap_level= 2;
    int dataWidth = inputdataWidth;
    int dataSize = inputdataSize;

    vector<Ctext> shortcuts = encrytedInputs;
    vector<Ctext> convData;

    if(shortcutState){
        startIn = get_current_time();
        encrytedInputs = fheController.batch_bootstrap_function(encrytedInputs, inputChannels, bootstrap_level);
        intermTime.push_back(measureTime(startIn, get_current_time()));

        convData = convolution_block(fheController, annBatchController, layer+"_conv1", encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, shortcutStridindLen);
        shortcuts = shortcut_convolution_block(fheController, annBatchController, layer, encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, shortcutStridindLen);
        
        dataWidth = dataWidth/2;
        dataSize = pow(dataWidth, 2);
        auto short_scalingVals = fheController.read_batch_scalingValues(shortcuts, outputChannels, (batchSize*dataSize));
        cout << "Shortcut SumScaling Values----: " << short_scalingVals << endl;
    }
    else{
        convData = convolution_block(fheController, annBatchController, layer+"_conv1", encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, stridingLen);
    }
    if(bootstrapState){
        startIn = get_current_time();
        convData = fheController.batch_bootstrap_function(convData, outputChannels, bootstrap_level);
        intermTime.push_back(measureTime(startIn, get_current_time()));
    }

    int totalSize = (batchSize*dataSize); 
    auto scalingVals = fheController.read_batch_scalingValues(convData, outputChannels, totalSize);
    cout << "1st Convolution Scaling Values: " << scalingVals << endl;

    
    if(layer == "layer4_block2"){
        for(int i=0; i<outputChannels; i++){
            scalingVals[i] = scalingFact*scalingVals[i]; 
        }
        cout << "SumScaling Values * ScaleFact -------: " << scalingVals << endl;
    }


    startIn = get_current_time();
    convData = annBatchController.secure_batch_relu(convData, scalingVals, outputChannels, totalSize, polyDegee);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    auto second_convData = convolution_block(fheController, annBatchController, layer+"_conv2", convData, batchSize, dataWidth, dataSize, outputChannels, outputChannels, stridingLen);
    scalingVals = fheController.read_batch_scalingValues(second_convData, outputChannels, totalSize);
    cout << "2nd Convolution Scaling Values: " << scalingVals << endl;

    vector<Ctext> sum_convData = annBatchController.secure_batch_sumTwoCiphers(second_convData, shortcuts, outputChannels);

    startIn = get_current_time();
    sum_convData = fheController.batch_bootstrap_function(sum_convData, outputChannels, bootstrap_level);
    intermTime.push_back(measureTime(startIn, get_current_time()));

    scalingVals = fheController.read_batch_scalingValues(sum_convData, outputChannels, totalSize);
    cout << "After SumScaling Values-------: " << scalingVals << endl;
    
    startIn = get_current_time();
    sum_convData = annBatchController.secure_batch_relu(sum_convData, scalingVals, outputChannels, totalSize, polyDegee);
    measuringTime.push_back(measureTime(startIn, get_current_time()));

    shortcuts.clear();
    convData.clear();
    return sum_convData;
}

Ctext fc_layer_block(FHEController &fheController, ANNBatchController &annBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions){
   
    string dataPath = "./../weights/resnet34/"+layer;
     cout << "kernel In: "<< endl;
    auto fc_rawKernel = load_batch_fc_weights(dataPath+"_weight.csv", outputChannels, batchSize, inputChannels);
    cout << "kernel Inputed: "<< endl;
    auto fc_rawBiasData = load_batch_fc_bias(dataPath+"_bias.csv", outputChannels, batchSize);
    int encodeLevel = encrytedInput->GetLevel();
    cout << "Bais Inputed: " << fc_rawBiasData.size() << endl;
    vector<Ptext> fc_kernelData;
    for(int i=0; i < outputChannels; i++){
        // cout << "fc kernel size: " << fc_rawKernel[i].size() << endl;
        auto encodeWeights = fheController.encode_inputData(fc_rawKernel[i], encodeLevel);
        fc_kernelData.push_back(encodeWeights);
    }
    Ptext fc_BiasData = fheController.encode_inputData(fc_rawBiasData, encodeLevel);

    startIn = get_current_time();
    Ctext fcData = annBatchController.secure_batch_flinear(encrytedInput, fc_kernelData, fc_BiasData, batchSize, inputChannels, outputChannels, rotPositions);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    fc_kernelData.clear();
    fc_kernelData.shrink_to_fit();
    return fcData;
}