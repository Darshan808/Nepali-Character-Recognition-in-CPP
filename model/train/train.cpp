#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include "../../include/NeuralNetwork.h"
#include "../../include/Danfe.h"

std::queue<std::pair<std::vector<std::vector<double>>, std::vector<int>>> batch_queue;
std::mutex mtx;
std::condition_variable cv;
bool done_loading = false;
std::vector<std::vector<double>> train_data;
std::vector<std::vector<double>> X_train;
std::vector<int> Y_train;
std::vector<std::vector<double>> X_val;
std::vector<int> Y_val;
std::vector<std::vector<double>> X_val_2;
std::vector<int> Y_val_2;
int i_batch = 1;

// Always remember to change one hot encoding!! (not required any more)

const string TYPE = "alpha";

void load_val_data()
{
    std::vector<std::vector<double>> val_data;
    std::vector<std::vector<double>> val_2_data;
    Danfe::loadData(val_data, "C:\\Users\\Darshan\\Desktop\\OOP project\\dataset\\val\\"+TYPE+"\\cv_1.txt", true);
    Danfe::loadData(val_2_data, "C:\\Users\\Darshan\\Desktop\\OOP project\\dataset\\val\\"+TYPE+"\\cv_2.txt", true);
    Danfe::xySplit(val_data, X_val, Y_val, 0, true);
    Danfe::xySplit(val_2_data, X_val_2, Y_val_2, 0, true);
}

void load_batch(std::pair<std::vector<std::vector<double>>, std::vector<int>> &batch, int batch_number)
{
    batch_number = 6;
    Danfe::loadData(train_data, "C:\\Users\\Darshan\\Desktop\\OOP project\\dataset\\train\\" + TYPE + "\\train_batch_" + std::to_string(batch_number) + ".txt");
    Danfe::xySplit(train_data, X_train, Y_train, 0);
    batch = {X_train, Y_train};
    if (batch_number % 20 == 0)
    {
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
        std::pair<std::vector<std::vector<double>>, std::vector<int>> batch;
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

void train_model(NeuralNetwork *NN, const std::pair<std::vector<std::vector<double>>, std::vector<int>> &batch)
{
    // cout<<"Reached here!"<<endl;
    if ((i_batch + 1) % 20 == 0)
        cout << "Training on Batch " << i_batch + 1 << endl;
    NN->train(batch.first, batch.second, X_val, Y_val, X_val_2, Y_val_2, 1, i_batch % 10 == 0 | i_batch == 476);
    i_batch++;
}

int main()
{
    int max_batches = 477; //477(128) 264(64)
    load_val_data();
    NeuralNetwork *NN = new NeuralNetwork({1024, 10, 36}, 0.01, "alpha.txt");
    std::thread loader_thread(batch_loader, max_batches);

    // Main thread starts training
    while (true)
    {
        std::pair<std::vector<std::vector<double>>, std::vector<int>> batch;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, []
                    { return !batch_queue.empty() || done_loading; });

            if (batch_queue.empty() && done_loading)
            {
                NN->saveNetwork("C:\\Users\\Darshan\\Desktop\\OOP project\\model\\saved\\alpha.txt");
                break;
            }

            batch = batch_queue.front();
            batch_queue.pop();
        }
        train_model(NN, batch);
    }

    loader_thread.join();
    delete NN;
    std::cout << "All batches loaded and trained.\n";

    return 0;
}