#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace MNN;
using namespace std;

// read the csv file(which has already been normalized)
vector<vector<float>> readCSV(const string& path) {
    vector<vector<float>> data;
    ifstream fin(path);
    if (!fin.is_open()) { cerr << "Failed to open " << path << endl; return data; }
    
    // skip the headline
    string line; getline(fin, line);               

    while (getline(fin, line)) {
        stringstream ss(line); string item; vector<float> row;
        while (getline(ss, item, ','))
            row.push_back(stof(item));
        data.push_back(row);
    }
    return data;
}

// read the scaler file (which was preserved through training)
bool loadScaler(const string& jsonPath, float& mean, float& scale) {
    ifstream fin(jsonPath);
    if (!fin.is_open()) { cerr << "Open scaler JSON failed: " << jsonPath << endl; return false; }
    json j; fin >> j;
    try {
        mean  = j.at("mean" ).get<vector<float>>()[0];
        scale = j.at("scale").get<vector<float>>()[0];
    } catch (const exception& e) {
        cerr << "Parse scaler JSON error: " << e.what() << endl; return false;
    }
    return true;
}
// -------------------------------------------------------

int main() {

    //load GRU MNN model
    const string modelPath = "/home/gary/Capstone/Neural/NN/My_NN/models/gru_model.mnn";
    auto net = shared_ptr<Interpreter>(Interpreter::createFromFile(modelPath.c_str()));
    if (!net) { cerr << "Failed to load model\n"; return -1; }

    //set CUDA, in order to use GPU

    // cfg.type      = MNN_FORWARD_CUDA; 
    // cfg.type      = MNN_FORWARD_OPENCL;

    ScheduleConfig cfg;  
    cfg.type = MNN_FORWARD_CPU;      
    cfg.numThread = 4;                       

    // //some setting for MNN using GPU(selective)
    // BackendConfig bCfg;
    // bCfg.precision = BackendConfig::Precision_High; // FP32
    // bCfg.power     = BackendConfig::Power_High;
    // cfg.backendConfig = &bCfg;

    // create session
    Session* session = net->createSession(cfg);

    //read the test.csv file
    string csvFile = "/home/gary/Capstone/Neural/NN/My_NN/data_for_train/test_data.csv";
    auto allData   = readCSV(csvFile);
    if (allData.empty()) { cerr << "CSV empty\n"; return -1; }

    const int totalCols = allData[0].size();

    //the last column is the target
    const int inputDim  = totalCols - 1;    
    const int seqLen    = 50;

    auto* inTensor = net->getSessionInput(session, "input");
    net->resizeTensor(inTensor, {1, seqLen, inputDim});
    net->resizeSession(session);

    //main loop
    vector<float> predsNorm, targNorm;
    Tensor hostInput(inTensor, Tensor::CAFFE);

    const int numSamples = allData.size() - seqLen;
    for (int i = 0; i < numSamples; ++i) {

        float* dst = hostInput.host<float>();

        for (int t = 0; t < seqLen; ++t)
            memcpy(dst + t * inputDim,
                   allData[i + t].data(), sizeof(float) * inputDim);

        inTensor->copyFromHostTensor(&hostInput);

        if (net->runSession(session) != NO_ERROR) {
            cerr << "Inference failed @" << i << endl; continue;
        }
        

        auto* outTensor = net->getSessionOutput(session, nullptr);

        //Tensor::CAFFE is for pytorch originally trainned, and Tensor::TENSORFLOW is for TensorFlow 
        Tensor hostOut(outTensor, Tensor::CAFFE);

        outTensor->copyToHostTensor(&hostOut);
        
        //get the pred and norm data
        predsNorm.push_back(hostOut.host<float>()[0]);
        targNorm.push_back(allData[i + seqLen][inputDim]);

    }

    //calculate the evaluation metrics
    size_t N = predsNorm.size();
    double sumAbs = 0, sumSq = 0;
    for (size_t i = 0; i < N; ++i) {
        double err = predsNorm[i] - targNorm[i];
        sumAbs += fabs(err);
        sumSq  += err * err;
    }
    cout << "Samples   : " << N                        << '\n'
         << "MAE (norm): " << sumAbs / N              << '\n'
         << "MSE (norm): " << sumSq  / N              << '\n'
         << "RMSE(norm): " << sqrt(sumSq / N)         << endl;

    //denormalize the target, in order to calculate the true energy consumption
    float mu = 0.0f, sig = 1.0f;
    if (!loadScaler("/home/gary/Capstone/Neural/NN/My_NN/scalers/target_scaler.json",
                    mu, sig)) return -1;

    vector<float> predsRaw(N), targRaw(N);
    for (size_t i = 0; i < N; ++i) {
        predsRaw[i] = predsNorm[i] * sig + mu;
        targRaw [i] = targNorm [i] * sig + mu;
    }
    

    //calculate the energy, here I asssume that dt is 0.01
    const double dt = 0.01;
    double totalPred = 0, totalActual = 0;
    for (size_t i = 0; i < N; ++i) {
        totalPred   += predsRaw[i] * dt;
        totalActual += targRaw[i]  * dt;
    }
    cout << "\n--- Energy (original unit) ---\n"
         << "Total Predicted: " << totalPred   << '\n'
         << "Total Actual   : " << totalActual << endl;


    //print 10 sample
    for (size_t i = 10000; i < 10010 && i < N; ++i)
    cout << "idx " << i
         << "  pred_Raw=" << predsRaw[i] * dt
         << "  true_Raw=" << targRaw[i] * dt << '\n';

    return 0;

}
