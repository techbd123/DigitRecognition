/*
 * backprop.h
 * Backpropagation neural network library.
 */
#ifndef BACKPROPHDR
#define BACKPROPHDR

#define BACKPROP_TYPE_NORMAL 1

/*
 * bkp_network_t
 */
typedef struct {
   int NumInTrainSet;
   int NumInputs;
   int NumHidden;
   int NumOutputs;
   int NumBias;
   float *TrainSetDesiredOutputVals; /* values set by bkp_set_training_set() */
   float *GivenDesiredOutputVals;    /* values set by bkp_set_output() */
   float *DesiredOutputVals;         /* pointer used by bkp_forward() and bkp_backward() */
   float *TrainSetInputVals;         /* values set by bkp_set_training_set() */
   float *GivenInputVals;            /* values set by bkp_set_input() */
   float *InputVals;                 /* pointer used by bkp_forward() and bkp_backward() */
   float *IHWeights;                 /* NumInputs x NumHidden */
   float *PrevDeltaIH;
   float *PrevDeltaHO;
   float *PrevDeltaBH;
   float *PrevDeltaBO;
   float *HiddenVals;
   float *HiddenBetas;
   float *HOWeights;                 /* NumHidden x NumOutputs */
   float *BiasVals;
   float *BHWeights;                 /* NumBias x NumHidden */
   float *BOWeights;                 /* NumBias x NumOutputs */
   float *OutputVals;
   float *OutputBetas;

   float RMSSquareOfOutputBetas;
   float PrevRMSError;
   float LastRMSError;
   float LearningError;
   float StepSize;
   float HStepSize;
   float Momentum;
   float Cost;
   int Epoch;
   int NumConsecConverged;

   int InputReady;
   int DesiredOutputReady;
   int Learned;
} bkp_network_t;

/*
 * bkp_config_t
 */
typedef struct {
   short Type;           /* see BACKPROP_TYPE_* above,
                            currently only BACKPROP_TYPE_NORMAL exists */
   int NumInputs;        /* number of input units */
   int NumHidden;        /* number of hidden units */
   int NumOutputs;       /* number of output units */
   float StepSize;       /* step size (aka learning rate) for changing 
                            the weights between units, it is > 0 and
                            < 1 (defaults to 0.5 if given 0) */
   float Momentum;       /* helps prevent getting stuck in local
                            minimas, it is a value betwene 0 - 1,
                            use a small StepSize when using a large
                            Momentum (defaults to 0.5 if given -1) */
   float Cost;           /* fraction of the weight's own value to 
                            subtract from itself each time the weight
                            is modified, use 0 if not desired */
} bkp_config_t;

int bkp_create_network(bkp_network_t **n, bkp_config_t *config);
void bkp_destroy_network(bkp_network_t *n);
int bkp_set_training_set(bkp_network_t *n, int ntrainset, float *tinputvals, float *targetvals);
void bkp_clear_training_set(bkp_network_t *n);
int bkp_learn(bkp_network_t *n, int ntimes);
int bkp_evaluate(bkp_network_t *n, float *eoutputvals, int sizeofoutputvals);
int bkp_query(bkp_network_t *n, 
      float *qlastlearningerror, float *qlastrmserror,
	float *qinputvals, float *qihweights, float *qhiddenvals, 
	float *qhoweights, float *qoutputvals,
      float *qbhweights, float *qbiasvals, float *qboweights);
int bkp_set_input(bkp_network_t *n, int setall, float val, float *sinputvals);
int bkp_set_output(bkp_network_t *n, int setall, float val, float *soutputvals);
int bkp_loadfromfile(bkp_network_t **n, char *fname);
int bkp_savetofile(bkp_network_t *n, char *fname);

#include "backprop.c"

#endif


