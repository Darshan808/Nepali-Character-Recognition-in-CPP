#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include "NeuralNetwork.h"
#include "Danfe.h"

std::queue<std::pair<std::vector<std::vector<float>>, std::vector<int>>> batch_queue;
std::mutex mtx;
std::condition_variable cv;
bool done_loading = false;
std::vector<std::vector<float>> train_data;
std::vector<std::vector<float>> X_train;
std::vector<int> Y_train;
std::vector<std::vector<float>> X_val;
std::vector<int> Y_val;
std::vector<std::vector<float>> X_val_2;
std::vector<int> Y_val_2;
int i_batch = 1;


void load_val_data(){
    std::vector<std::vector<float>> val_data;
    std::vector<std::vector<float>> val_2_data;
    Danfe::loadData(val_data, "dataset/cv_data/cv.txt", true);
    Danfe::loadData(val_2_data, "dataset/cv_data/cv_2.txt", true);
    Danfe::xySplit(val_data, X_val, Y_val, true);
    Danfe::xySplit(val_2_data, X_val_2, Y_val_2, true);
}

void load_batch(std::pair<std::vector<std::vector<float>>, std::vector<int>> &batch, int batch_number)
{
    Danfe::loadData(train_data, "dataset/train_batches/train_batch_" + std::to_string(batch_number) + ".txt");
    Danfe::xySplit(train_data, X_train, Y_train);
    batch = {X_train, Y_train};
    if(batch_number % 100 == 0){
        std::cout << "Batch " << batch_number << " loaded.\n";
    }
    X_train.clear();
    train_data.clear();
    Y_train.clear();
}

void batch_loader(int max_batches)
{
    for (int i = 0; i < max_batches; ++i)
    {
        std::pair<std::vector<std::vector<float>>, std::vector<int>> batch;
        load_batch(batch, i);
        {
            std::lock_guard<std::mutex> lock(mtx);
            batch_queue.push(batch);
        }
        cv.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        done_loading = true;
    }
    cv.notify_one();
}

void train_model(NeuralNetwork *NN, const std::pair<std::vector<std::vector<float>>, std::vector<int>> &batch)
{
    if((i_batch+1)%50 == 0)
        cout<<"Training on Batch "<<i_batch+1<<endl;
    i_batch++;
    NN->train(batch.first,batch.second,X_val,Y_val,X_val_2,Y_val_2,1,i_batch%50==0);
}

int main()
{
    int max_batches = 1221;
    load_val_data();
    NeuralNetwork* NN = new NeuralNetwork({1024,256,46},0.01);
    std::thread loader_thread(batch_loader, max_batches);

    // Main thread starts training
    while (true)
    {
        std::pair<std::vector<std::vector<float>>, std::vector<int>> batch;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, []
                    { return !batch_queue.empty() || done_loading; });

            if (batch_queue.empty() && done_loading){
                NN->saveNetwork("model/512_256.txt");
                break;
            }

            batch = batch_queue.front();
            batch_queue.pop();
        }
        train_model(NN,batch);
    }

    loader_thread.join();

    std::cout << "All batches loaded and trained.\n";

    return 0;
}