#include "include/MaiInterface.h"
#include "include/Type.h"
#include "source/core/SimpleNeuralNetwork.h"
#include <pthread.h>

using namespace MAI;
void* thread_run(void* _val) { return NULL;}

class NAME{
public:
    NAME(){
        printf("NAME_____________________\n");
    }
};

static NAME name;

int main(int argc, char **argv) {
    ALOGI("NeuralNetwork Test");
    std::unique_ptr<NeuralNetwork> networkPtr(new SimpleNeuralNetwork());
    MAI_STATUS status = networkPtr->init();
    ALOGI("NeuralNetwork init ret:%d", status);
    pthread_t tid;
    int tret = pthread_create(&tid, NULL, thread_run, NULL);

}
