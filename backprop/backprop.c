/*
 * backprop.c
 * Backpropagation neural network library.
 *
 * 2016, December 13 - fixed bkp_loadfromfile. Changed the file format
 * to include a file type, 'A', and the network type. Updated 
 * bkp_savetofile to match.
 * 2016, April 7 - made bkp_query return BiasVals, BHWeights and BIWeights
 * 2016, April 3 - cleaned up version for website publication
 * 1992 - originally written around this time
 * A note of credit:
 * This code had its origins as code obtained back around 1992 by sending
 * a floppy disk to The Amateur Scientist, Scientific American magazine.
 * I've since modified and added to it a great deal, and it's even on 
 * its 3rd OS (MS-DOS -> QNX -> Windows). As I no longer have the 
 * original I can't know how much is left to give credit for.
 */
#define CMDIFFSTEPSIZE   1 /* set to 1 for Chen & Mars differential step size */
#define DYNAMIC_LEARNING 0 /* set to 1 for Dynamic Learning */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "backprop.h"

static void bkp_setup_all(bkp_network_t *n);
static void bkp_forward(bkp_network_t *n);
static void bkp_backward(bkp_network_t *n);

/* The following sigmoid returns values from 0.0 to 1.0 */
#define sigmoid(x)           (1.0 / (1.0 + (float)exp(-(double)(x))))
#define sigmoidDerivative(x) ((float)(x)*(1.0-(x))) 
/* random() for -1 to +1 */
#define random()             ((((float)rand()/(RAND_MAX)) * 2.0) - 1.0)
/* random() for -0.5 to +0.5
#define random()             (((float)rand()/(RAND_MAX)) - 0.5)
*/

/*
 * bkp_create_network - Create a new network with the given configuration.
 * Returns a pointer to the new network in 'n'.
 *
 * Return Value:
 * int  0: Success
 *     -1: Error, errno is set to:
 *         ENOMEM - out of memory
 *         EINVAL - config.Type is one which this server does not handle
 */
int bkp_create_network(bkp_network_t **n, bkp_config_t *config)
{
   if (config->Type != BACKPROP_TYPE_NORMAL) {
      errno = EINVAL;
      return -1;
   }

   if ((*n = (bkp_network_t *) malloc(sizeof(bkp_network_t))) == NULL) {
      errno = ENOMEM;
      return -1;
   }
	
   (*n)->NumInputs = config->NumInputs;
   (*n)->NumHidden = config->NumHidden;
   (*n)->NumOutputs = config->NumOutputs;
   (*n)->NumConsecConverged = 0;
   (*n)->Epoch = (*n)->LastRMSError = (*n)->RMSSquareOfOutputBetas = 0.0;
   (*n)->NumBias = 1;
   if (config->StepSize == 0)
      (*n)->StepSize = 0.5;
   else
      (*n)->StepSize = config->StepSize;
#if CMDIFFSTEPSIZE
   (*n)->HStepSize = 0.1 * (*n)->StepSize;
#endif
   if (config->Momentum == -1)
      (*n)->Momentum = 0.5;
   else
      (*n)->Momentum = config->Momentum;
   (*n)->Cost = config->Cost;
   if (((*n)->GivenInputVals = (float *) malloc((*n)->NumInputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->GivenDesiredOutputVals = (float *) malloc((*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->IHWeights = (float *) malloc((*n)->NumInputs * (*n)->NumHidden * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->PrevDeltaIH = (float *) malloc((*n)->NumInputs * (*n)->NumHidden * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->PrevDeltaHO = (float *) malloc((*n)->NumHidden * (*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->PrevDeltaBH = (float *) malloc((*n)->NumBias * (*n)->NumHidden * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->PrevDeltaBO = (float *) malloc((*n)->NumBias * (*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->HiddenVals = (float *) malloc((*n)->NumHidden * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->HiddenBetas = (float *) malloc((*n)->NumHidden * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->HOWeights = (float *) malloc((*n)->NumHidden * (*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->BiasVals = (float *) malloc((*n)->NumBias * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->BHWeights = (float *) malloc((*n)->NumBias * (*n)->NumHidden * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->BOWeights = (float *) malloc((*n)->NumBias * (*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->OutputVals = (float *) malloc((*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   if (((*n)->OutputBetas = (float *) malloc((*n)->NumOutputs * sizeof(float))) == NULL)
      goto memerrorout;
   bkp_setup_all(*n);
   return 0;
	
memerrorout:
   bkp_destroy_network(*n);
   errno = ENOMEM;
   return -1;
}

/*
 * bkp_destroy_network - Frees up any resources allocated for the
 * given neural network.
 *
 * Return Values:
 *    (none)
 */
void bkp_destroy_network(bkp_network_t *n)
{
   if (n == NULL)
      return;

   if (n->GivenInputVals == NULL) return;
   bkp_clear_training_set(n);
   free(n->GivenInputVals);
   if (n->GivenDesiredOutputVals != NULL) { free(n->GivenDesiredOutputVals); n->GivenDesiredOutputVals = NULL; }
   if (n->IHWeights != NULL) { free(n->IHWeights); n->IHWeights = NULL; }
   if (n->PrevDeltaIH != NULL) { free(n->PrevDeltaIH); n->PrevDeltaIH = NULL; }
   if (n->PrevDeltaHO != NULL) { free(n->PrevDeltaHO); n->PrevDeltaHO = NULL; }
   if (n->PrevDeltaBH != NULL) { free(n->PrevDeltaBH); n->PrevDeltaBH = NULL; }
   if (n->PrevDeltaBO != NULL) { free(n->PrevDeltaBO); n->PrevDeltaBO = NULL; }
   if (n->HiddenVals != NULL) { free(n->HiddenVals); n->HiddenVals = NULL; }
   if (n->HiddenBetas != NULL) { free(n->HiddenBetas); n->HiddenBetas = NULL; }
   if (n->HOWeights != NULL) { free(n->HOWeights); n->HOWeights = NULL; }
   if (n->BiasVals != NULL) { free(n->BiasVals); n->BiasVals = NULL; }
   if (n->BHWeights != NULL) { free(n->BHWeights); n->BHWeights = NULL; }
   if (n->BOWeights != NULL) { free(n->BOWeights); n->BOWeights = NULL; }
   if (n->OutputVals != NULL) { free(n->OutputVals); n->OutputVals = NULL; }
   if (n->OutputBetas != NULL) { free(n->OutputBetas); n->OutputBetas = NULL; }
   n->GivenInputVals = NULL;
   free(n);
}

/*
 * bkp_set_training_set - Gives addresses of the input and target data 
 * in the form of a input values and output values. No data is copied
 * so do not destroy the originals until you call 
 * bkp_clear_training_set(), or bkp_destroy_network().
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 */
int bkp_set_training_set(bkp_network_t *n, int ntrainset, float *tinputvals, float *targetvals)
{
   if (!n) {
      errno = ENOENT;
      return -1;
   }

   bkp_clear_training_set(n);

   n->NumInTrainSet = ntrainset;
   n->TrainSetInputVals = tinputvals;
   n->TrainSetDesiredOutputVals = targetvals;
	
   return 0;
}

/*
 * bkp_clear_training_set - Invalidates the training set such that
 * you can no longer use it for training. After this you can free
 * up any memory associated with the training data you'd passed to
 * bkp_set_training_set(). It has not been modified in any way.
 *
 * Return Values:
 *    (none)
 */
void bkp_clear_training_set(bkp_network_t *n)
{
   if (n->NumInTrainSet > 0) {
      n->TrainSetInputVals = NULL;
      n->TrainSetDesiredOutputVals = NULL;
      n->NumInTrainSet = 0;
   }
}

static void bkp_setup_all(bkp_network_t *n)
{
   int i, h, o, b;
	
   n->InputReady = n->DesiredOutputReady = n->Learned = 0;

   n->LearningError = 0.0;
	
   for (i = 0;  i < n->NumInputs;  i++)
      n->GivenInputVals[i] = 0.0;
	
   for(h = 0;  h < n->NumHidden;  h++) {
      n->HiddenVals[h] = 0.0;
      for (i = 0;  i < n->NumInputs;  i++) {
         n->IHWeights[i+(h*n->NumInputs)] = random();
         n->PrevDeltaIH[i+(h*n->NumInputs)] = 0.0;
      }
      for (b = 0;  b < n->NumBias;  b++) {
         n->BHWeights[b+(h*n->NumBias)] = random();
         n->PrevDeltaBH[b+(h*n->NumBias)] = 0.0;
      }
   }

   for(o = 0;  o < n->NumOutputs;  o++) {
      n->OutputVals[o] = 0.0;
      for (h = 0;  h < n->NumHidden;  h++) {
         n->HOWeights[h+(o*n->NumHidden)] = random();
         n->PrevDeltaHO[h+(o*n->NumHidden)] = 0.0;
      }
      for (b = 0;  b < n->NumBias;  b++) {
         n->BOWeights[b+(o*n->NumBias)] = random();
         n->PrevDeltaBO[b+(o*n->NumBias)] = 0.0;
      }
   }
	
   for (b = 0;  b < n->NumBias;  b++)
      n->BiasVals[b] = 1.0;
}

/*
 * bkp_learn - Tells backprop to learn the current training set ntimes.
 * The training set must already have been set by calling 
 * bkp_set_training_set(). This does not return until the training
 * has been completed. You can call bkp_query() after this to find out 
 * the results of the learning.
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 *        ESRCH if no bkp_set_training_set() has been done yet.
 */
int bkp_learn(bkp_network_t *n, int ntimes)
{
   int item, run;
	
   if (!n) {
      errno = ENOENT;
      return -1;
   }
   if (n->NumInTrainSet == 0) {
      errno = ESRCH;
      return -1;
   }

   for (run = 0;  run < ntimes;  run++) {
      for (item = 0;  item < n->NumInTrainSet;  item++) {
         /* set up for the given set item */
         n->InputVals = &(n->TrainSetInputVals[item*n->NumInputs]);
         n->DesiredOutputVals = &(n->TrainSetDesiredOutputVals[item*n->NumOutputs]);

         /* now do the learning */	
         bkp_forward(n);
         bkp_backward(n);
      }
	
      /* now that we have gone through the entire training set, calculate the
         RMS to see how well we have learned */
		   
      n->Epoch++;

      /* calculate the RMS error for the epoch just completed */
      n->LastRMSError = sqrt(n->RMSSquareOfOutputBetas / (n->NumInTrainSet * n->NumOutputs));
      n->RMSSquareOfOutputBetas = 0.0;
		
#if DYNAMIC_LEARNING
      if (n->Epoch > 1) {
         if (n->PrevRMSError < n->LastRMSError) {
            /* diverging */
            n->NumConsecConverged = 0;
            n->StepSize *= 0.95; /* make step size smaller */
#if CMDIFFSTEPSIZE
            n->HStepSize = 0.1 * n->StepSize;
#endif
#ifdef DISPLAYMSGS
            printf("Epoch: %d Diverging:  Prev %f, New %f, Step size %f\n",
               n->Epoch, n->PrevRMSError, n->LastRMSError, n->StepSize);
#endif
         } else if (n->PrevRMSError > n->LastRMSError) {
            /* converging */
            n->NumConsecConverged++;
            if (n->NumConsecConverged == 5) {
               n->StepSize += 0.04; /* make step size bigger */
#if CMDIFFSTEPSIZE
               n->HStepSize = 0.1 * n->StepSize;
#endif
#ifdef DISPLAYMSGS
               printf("Epoch: %d Converging: Prev %f, New %f, Step size %f\n",
                  n->Epoch, n->PrevRMSError, n->LastRMSError, n->StepSize);
#endif
               n->NumConsecConverged = 0;
            }
         } else {
            n->NumConsecConverged = 0;
         }
      }
      n->PrevRMSError = n->LastRMSError;
#endif
   }
   n->Learned = 1;
   return 0;
}

/*
 * bkp_evaluate - Evaluate but don't learn the current input set.
 * This is usually preceded by a call to bkp_set_input() and is
 * typically called after the training set (epoch) has been learned.
 *
 * If you give eoutputvals as NULL then you can do a bkp_query() to 
 * get the results.
 *
 * If you give the address of a buffer to return the results of the
 * evaluation (eoutputvals != NULL) then the results will copied to the 
 * eoutputvals buffer.
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 *        ESRCH if no bkp_set_input() has been done yet.
 *        ENODEV if both bkp_create_network() and bkp_set_input()
 *               have been done but bkp_earn() has not been done
 *               yet (ie; neural net has not had any training).
 *        EINVAL if sizeofoutputvals is not the same as the
 *               size understood according to n. This is to help
 *               prevent buffer overflow during copying.
 */
int bkp_evaluate(bkp_network_t *n, float *eoutputvals, int sizeofoutputvals)
{
   if (!n) {
      errno = ENOENT;
      return -1;
   }
   if (!n->InputReady) {
      errno = ESRCH;
      return -1;
   }
   if (!n->Learned) {
      errno = ENODEV;
      return -1;
   }

   n->InputVals = n->GivenInputVals;
   n->DesiredOutputVals = n->GivenDesiredOutputVals;

   bkp_forward(n);

   if (eoutputvals) {
      if (sizeofoutputvals != n->NumOutputs*sizeof(float)) {
         errno = EINVAL;
         return -1;
      }
      memcpy(eoutputvals, n->OutputVals, n->NumOutputs*sizeof(float));
   }
   return 0;
}

/*
 * bkp_forward - This makes a pass from the input units to the hidden 
 * units to the output units, updating the hidden units, output units and 
 * other components. This is how the neural network is run in order to
 * evaluate a set of input values to get output values.
 * When training the neural network, this is the first step in the
 * backpropagation algorithm.
 */
static void bkp_forward(bkp_network_t *n)
{
   int i, h, o, b;
	
   n->LearningError = 0.0;

   /*
    * Apply input unit values to weights between input units and hidden units
    * Apply bias unit values to weights between bias units and hidden units
    */
   for (h = 0;  h < n->NumHidden;  h++) {
      n->HiddenVals[h] = 0.0;
      n->HiddenBetas[h] = 0.0; /* needed if doing a backward pass later */
      for (i = 0;  i < n->NumInputs;  i++)
         n->HiddenVals[h] = n->HiddenVals[h] + (n->InputVals[i] * n->IHWeights[i+(h*n->NumInputs)]);
      for (b = 0;  b < n->NumBias;  b++)
         n->HiddenVals[h] = n->HiddenVals[h] + (n->BiasVals[b] * n->BHWeights[b+(h*n->NumBias)]);
      n->HiddenVals[h] = sigmoid(n->HiddenVals[h]);
   }
	
   /*
    * Apply hidden unit values to weights between hidden units and outputs
    * Apply bias unit values to weights between bias units and outputs
    */
   for (o = 0;  o < n->NumOutputs;  o++) {
      n->OutputVals[o] = 0.0;
      for (h = 0;  h < n->NumHidden;  h++)
         n->OutputVals[o] = n->OutputVals[o] + (n->HiddenVals[h] * n->HOWeights[h+(o*n->NumHidden)]);
      for (b = 0;  b < n->NumBias;  b++)
         n->OutputVals[o] = n->OutputVals[o] + (n->BiasVals[b] * n->BOWeights[b+(o*n->NumBias)]);
      n->OutputVals[o] = sigmoid(n->OutputVals[o]);
      n->LearningError = n->LearningError + 
         ((n->OutputVals[o] - n->DesiredOutputVals[o]) * (n->OutputVals[o] - n->DesiredOutputVals[o]));
   }
   n->LearningError = n->LearningError / 2.0;
}

/*
 * bkp_backward - This is the 2nd half of the backpropagation algorithm
 * which is carried out immediately after bkp_forward() has done its
 * step of calculating the outputs. This does the reverse, comparing
 * those output values to those given as targets in the training set 
 * and updating the weights and other components appropriately, which
 * is essentially the training of the neural network.
 */
static void bkp_backward(bkp_network_t *n)
{
   float deltaweight;
   int i, h, o, b;

   for (o = 0;  o < n->NumOutputs;  o++) {
      /* calculate beta for output units */
      n->OutputBetas[o] = n->DesiredOutputVals[o] - n->OutputVals[o];

      /* update for RMS error */
      n->RMSSquareOfOutputBetas += (n->OutputBetas[o] * n->OutputBetas[o]);

      /* update hidden to output weights */
      for (h = 0;  h < n->NumHidden;  h++) {
         /* calculate beta for hidden units for later */
         n->HiddenBetas[h] = n->HiddenBetas[h] +
            (n->HOWeights[h+(o*n->NumHidden)] * sigmoidDerivative(n->OutputVals[o]) * n->OutputBetas[o]);

#if CMDIFFSTEPSIZE
         deltaweight = n->HiddenVals[h] * n->OutputBetas[o];
#else
         deltaweight = n->HiddenVals[h] * n->OutputBetas[o] *
            sigmoidDerivative(n->OutputVals[o]);
#endif
         n->HOWeights[h+(o*n->NumHidden)] = n->HOWeights[h+(o*n->NumHidden)] + 
            (n->StepSize * deltaweight) +
            (n->Momentum * n->PrevDeltaHO[h+(o*n->NumHidden)]);
         n->PrevDeltaHO[h+(o*n->NumHidden)] = deltaweight;
      }
      /* update bias to output weights */
      for (b = 0;  b < n->NumBias;  b++) {
#if CMDIFFSTEPSIZE
         deltaweight = n->BiasVals[b] * n->OutputBetas[o];
#else
         deltaweight = n->BiasVals[b] * n->OutputBetas[o] +
            sigmoidDerivative(n->OutputVals[o]);
#endif
         n->BOWeights[b+(o*n->NumBias)] = n->BOWeights[b+(o*n->NumBias)] +
            (n->StepSize * deltaweight) +
            (n->Momentum * n->PrevDeltaBO[b+(o*n->NumBias)]);
         n->PrevDeltaBO[b+(o*n->NumBias)] = deltaweight;
      }
   }

   for (h = 0;  h < n->NumHidden;  h++) {
      /* update input to hidden weights */
      for (i = 0;  i < n->NumInputs;  i++) {
         deltaweight = n->InputVals[i] * sigmoidDerivative(n->HiddenVals[h]) *
            n->HiddenBetas[h];
         n->IHWeights[i+(h*n->NumInputs)] = n->IHWeights[i+(h*n->NumInputs)] + 
#if CMDIFFSTEPSIZE
            (n->HStepSize * deltaweight) +
#else
            (n->StepSize * deltaweight) +
#endif
            (n->Momentum * n->PrevDeltaIH[i+(h*n->NumInputs)]);
         n->PrevDeltaIH[i+(h*n->NumInputs)] = deltaweight;
         if (n->Cost)
            n->IHWeights[i+(h*n->NumInputs)] = n->IHWeights[i+(h*n->NumInputs)] - 
               (n->Cost * n->IHWeights[i+(h*n->NumInputs)]);
      }
      /* update bias to hidden weights */
      for (b = 0;  b < n->NumBias;  b++) {
         deltaweight = n->BiasVals[b] * n->HiddenBetas[h] *
            sigmoidDerivative(n->HiddenVals[h]);
         n->BHWeights[b+(h*n->NumBias)] = n->BHWeights[b+(h*n->NumBias)] +
#if CMDIFFSTEPSIZE
            (n->HStepSize * deltaweight) +
#else
            (n->StepSize * deltaweight) +
#endif
            (n->Momentum * n->PrevDeltaBH[b+(h*n->NumBias)]);
         n->PrevDeltaBH[b+(h*n->NumBias)] = deltaweight;
         if (n->Cost)
            n->BHWeights[b+(h*n->NumBias)] = n->BHWeights[b+(h*n->NumBias)] - 
               (n->Cost * n->BHWeights[b+(h*n->NumBias)]);
      }
   }
}

/*
 * bkp_query - Get the current state of the neural network.
 *
 * Parameters (all parameters return information unless given as NULL):
 * float *qlastlearningerror: The error for the last set of inputs
 *                            and outputs learned by bkp_learn()
 *                            or evaluated by bkp_evaluate().
 *                            It is the sum of the squares
 *                            of the difference between the actual
 *                            outputs and the target or desired outputs,
 *                            all divided by 2 
 * float *qlastrmserror:      The RMS error for the last epoch learned
 *                            i.e. the learning of the training set.
 * float *qinputvals:         An array to fill with the current input 
 *                            values (must be at least 
 *                            bkp_config_t.NumInputs * sizeof(float))
 * float *qihweights:         An array to fill with the current input
 *                            units to hidden units weights (must be at 
 *                            least bkp_config_t.NumInputs * 
 *                            bkp_config_t.NumHidden * sizeof(float)
 * float *qhiddenvals:        An array to fill with the current hidden
 *                            unit values (must be at least 
 *                            bkp_config_t.NumHidden * sizeof(float))
 * float *qhoweights:         An array to fill with the current hidden
 *                            units to output units weights (must be at 
 *                            least bkp_config_t.NumHidden * 
 *                            bkp_config_t.NumOutputs * sizeof(float))
 * float *qoutputvals:        An array to fill with the current output
 *                            values (must be at least 
 *                            bkp_config_t.NumOutputs * sizeof(float))
 * Note that for the following three, the size required is 1 * ...
 * The reason for the 1 is because there is only one bias unit for
 * everything. Theoretically there could be more though.
 * float *qbhweights:         An array to fill with the current bias
 *                            units to hidden units weights (must be at 
 *                            least 1 * bkp_config_t->NumHidden * 
 *                            sizeof(float))
 * float *qbiasvals:          An array to fill with the current bias
 *                            values (must be at least 1 * sizeof(float))
 * float *qboweights:         An array to fill with the current bias
 *                            units to output units weights (must be at
 *                            least 1 * (*n)->NumOutputs * sizeof(float))
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 *        ENODEV if bkp_create_network() has been done
 *               but bkp_learn() has not been done yet (ie; neural 
 *               net has not had any training).
 */
int bkp_query(bkp_network_t *n, 
      float *qlastlearningerror, float *qlastrmserror,
	float *qinputvals, float *qihweights, float *qhiddenvals, 
	float *qhoweights, float *qoutputvals,
      float *qbhweights, float *qbiasvals, float *qboweights)
{
   if (!n) {
      errno = ENOENT;
      return -1;
   }
   if (!n->Learned) {
      errno  = ENODEV;
      return -1;
   }
   if (qlastlearningerror)
      *qlastlearningerror = n->LearningError;
   if (qlastrmserror)
      *qlastrmserror = n->LastRMSError;
   if (qinputvals)
      memcpy(qinputvals, n->InputVals, n->NumInputs*sizeof(float));
   if (qihweights)
      memcpy(qihweights, n->IHWeights, (n->NumInputs*n->NumHidden)*sizeof(float));
   if (qhiddenvals)
      memcpy(qhiddenvals, n->HiddenVals, n->NumHidden*sizeof(float));
   if (qhoweights)
      memcpy(qhoweights, n->HOWeights, (n->NumHidden*n->NumOutputs)*sizeof(float));
   if (qoutputvals)
      memcpy(qoutputvals, n->OutputVals, n->NumOutputs*sizeof(float));
   if (qbhweights)
      memcpy(qbhweights, n->BHWeights, n->NumBias*n->NumHidden*sizeof(float));
   if (qbiasvals)
      memcpy(qbiasvals, n->BiasVals, n->NumBias*sizeof(float));
   if (qboweights)
      memcpy(qboweights, n->BOWeights, n->NumBias*n->NumOutputs*sizeof(float));
   return 0;
}

/*
 * bkp_set_input - Use this to set the current input values of the neural
 * network. Nothing is done with the values until bkp_learn() is called.
 *
 * Parameters:
 * int setall: If 1: Set all inputs to Val. Any sinputvals are ignored so 
 *                   you may as well give sinputvals as NULL.
 * float val: See SetAll.
 * float sinputvals: An array of input values.  The array should contain
 *                   bkp_config_t.NumInputs elements.
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 */
int bkp_set_input(bkp_network_t *n, int setall, float val, float *sinputvals)
{
   int i;

   if (!n) {
      errno = ENOENT;
      return -1;
   }

   if (setall) {
      for (i = 0;  i < n->NumInputs;  i++)
         n->GivenInputVals[i] = val;
   } else {
      memcpy(n->GivenInputVals, sinputvals, n->NumInputs*sizeof(float));
   }

   n->InputReady = 1;
   return 0;
}

/*
 * bkp_set_output - Use this so that bkp_evaluate() can calculate the
 * error between the output values you passs to bkp_set_output() and 
 * the output it gets by evaulating the network using the input values
 * you passed to the last call to bkp_set_input(). The purpose is so
 * that you can find out what that error is using bkp_query()'s
 * qlastlearningerror argument. Typically bkp_set_output() will have been
 * accompanied by a call to bkp_set_input().
 *
 * Parameters:
 * int setall: If 1: Set all outputs to val. Any soutputvals 
 *                   are ignored so you may as well give 
 *                   soutputvals as NULL.
 *             If 0: val is ignored. You must provide soutputvals.
 * float val:  See setall.
 * float sonputvals: An array of input values.  The array should contain
 *                   bkp_config_t.NumInputs elements.
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 */
int bkp_set_output(bkp_network_t *n, int setall, float val, float *soutputvals)
{
   int i;

   if (!n) {
      errno = ENOENT;
      return -1;
   }

   if (setall) {
      for (i = 0;  i < n->NumOutputs;  i++)
         n->GivenDesiredOutputVals[i] = val;
   } else {
      memcpy(n->GivenDesiredOutputVals, soutputvals, n->NumOutputs*sizeof(float));
   }

   n->DesiredOutputReady = 1;
   return 0;
}

/*
 * bkp_loadfromfile - Creates a neural network using the information
 * loaded from the given file and returns a pointer to it in n.
 * If successful, the end result of this will be a neural network
 * for which bkp_create_network() will effectively have been done.
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        EOK or any applicable errors from the open() or read() functions.
 *        ENOMEM if no memory.
 *        EINVAL if the file is not in the correct format.
 */
int bkp_loadfromfile(bkp_network_t **n, char *fname)
{
   char file_format;
   int fd, returncode;
   bkp_config_t config;
	
   returncode = -1;
	
   if ((fd = open(fname, O_RDONLY)) == -1)
      return returncode;

   if (read(fd, &file_format, sizeof(char)) == -1)
      goto cleanupandret;
   if (file_format != 'A') {
      errno = EINVAL;
      goto cleanupandret;
   }
   if (read(fd, &config.Type, sizeof(short)) == -1)
      goto cleanupandret;
   if (read(fd, &config.NumInputs, sizeof(int)) == -1)
      goto cleanupandret;
   if (read(fd, &config.NumHidden, sizeof(int)) == -1)
      goto cleanupandret;
   if (read(fd, &config.NumOutputs, sizeof(int)) == -1)
      goto cleanupandret;
   if (read(fd, &config.StepSize, sizeof(float)) == -1)
      goto cleanupandret;
   if (read(fd, &config.Momentum, sizeof(float)) == -1)
      goto cleanupandret;
   if (read(fd, &config.Cost, sizeof(float)) == -1)
      goto cleanupandret;

   if (bkp_create_network(n, &config) == -1) {
      goto cleanupandret;
   }

   (*n)->InputVals = (*n)->GivenInputVals;
   (*n)->DesiredOutputVals = (*n)->GivenDesiredOutputVals;

   if (read(fd, (int *) &(*n)->NumBias, sizeof(int)) == -1)
      goto errandret;
		
   if (read(fd, (int *) &(*n)->InputReady, sizeof(int)) == -1)
      goto errandret;
   if (read(fd, (int *) &(*n)->DesiredOutputReady, sizeof(int)) == -1)
      goto errandret;
   if (read(fd, (int *) &(*n)->Learned, sizeof(int)) == -1)
      goto errandret;
		
   if (read(fd, (*n)->InputVals, (*n)->NumInputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->DesiredOutputVals, (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->IHWeights, (*n)->NumInputs * (*n)->NumHidden * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->PrevDeltaIH, (*n)->NumInputs * (*n)->NumHidden * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->PrevDeltaHO, (*n)->NumHidden * (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->PrevDeltaBH, (*n)->NumBias * (*n)->NumHidden * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->PrevDeltaBO, (*n)->NumBias * (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->HiddenVals, (*n)->NumHidden * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->HiddenBetas, (*n)->NumHidden * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->HOWeights, (*n)->NumHidden * (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->BiasVals, (*n)->NumBias * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->BHWeights, (*n)->NumBias * (*n)->NumHidden * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->BOWeights, (*n)->NumBias * (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->OutputVals, (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
   if (read(fd, (*n)->OutputBetas, (*n)->NumOutputs * sizeof(float)) == -1)
      goto errandret;
		
   returncode = 0;
   goto cleanupandret;

errandret:
   bkp_destroy_network(*n);

cleanupandret:
   close(fd);
	
   return returncode;
}	

/*
 * bkp_savetofile
 *
 * The format of the file is:
 *
 *  1. File format version e.g. 'A' (sizeof(char))
 *  2. Network type BACKPROP_TYPE_* (sizeof(short))
 *  3. Number of inputs (sizeof(int))
 *  4. Number of hidden units (sizeof(int))
 *  5. Number of outputs (sizeof(int))
 *  6. StepSize (sizeof(float))
 *  7. Momentum (sizeof(float))
 *  8. Cost (sizeof(float))
 *  9. Number of bias units (sizeof(int))
 * 10. Is input ready? 0 = no, 1 = yes (sizeof(int))
 * 11. Is desired output ready? 0 = no, 1 = yes (sizeof(int))
 * 12. Has some learning been done? 0 = no, 1 = yes (sizeof(int))
 * 13. Current input values (InputVals) (NumInputs * sizeof(float))
 * 14. Current desired output values (DesiredOutputVals) (NumOutputs * sizeof(float))
 * 15. Current input-hidden weights (IHWeights) (NumInputs * NumHidden * sizeof(float))
 * 16. Previous input-hidden weight deltas (PrevDeltaIH) (NumInputs * NumHidden * sizeof(float))
 * 17. Previous output-hidden weight deltas (PrevDeltaHO) (NumHidden * NumOutputs * sizeof(float))
 * 18. Previous bias-hidden weight deltas (PrevDeltaBH) (NumBias * NumHidden * sizeof(float))
 * 19. Previous bias-output weight deltas (PrevDeltaBO) (NumBias * NumOutputs * sizeof(float))
 * 20. Current hidden unit values (HiddenVals) (NumHidden * sizeof(float))
 * 21. Current hidden unit beta values (HiddenBetas) (NumHidden * sizeof(float))
 * 22. Current hidden-output weights (HOWeights) (NumHidden * NumOutputs * sizeof(float))
 * 23. Current bias unit values (BiasVals) (NumBias * sizeof(float))
 * 24. Current bias-hidden weights (BHWeights) (NumBias * NumHidden * sizeof(float))
 * 25. Current bias-output weights (BOWeights) (NumBias * NumOutputs * sizeof(float))
 * 26. Current output values (OutputVals) (NumOutputs * sizeof(float))
 * 27. Current output unit betas (OutputBetas) (NumOutputs * sizeof(float))
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 *        EOK or any applicable errors from the open() or write() 
 *        functions.
 */
int bkp_savetofile(bkp_network_t *n, char *fname)
{
   int fd, returncode;
   short type = BACKPROP_TYPE_NORMAL;
	
   returncode = -1;
	
   fd = open(fname, O_WRONLY | O_CREAT | O_TRUNC,
         S_IRUSR | S_IWUSR);
         // For Unix/Linux-like environments the following can also be used
         // | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
   if (fd == -1)
      return returncode;
	
   if (write(fd, (char *) "A", sizeof(char)) == -1) // file format version A
      goto cleanupandret;
   if (write(fd, (short *) &type, sizeof(short)) == -1) // BACKPROP_TYPE_*
      goto cleanupandret;
   if (write(fd, (int *) &n->NumInputs, sizeof(int)) == -1)
      goto cleanupandret;
   if (write(fd, (int *) &n->NumHidden, sizeof(int)) == -1)
      goto cleanupandret;
   if (write(fd, (int *) &n->NumOutputs, sizeof(int)) == -1)
      goto cleanupandret;
   if (write(fd, (float *) &n->StepSize, sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, (float *) &n->Momentum, sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, (float *) &n->Cost, sizeof(float)) == -1)
      goto cleanupandret;

   if (write(fd, (int *) &n->NumBias, sizeof(int)) == -1)
      goto cleanupandret;
		
   if (write(fd, (int *) &n->InputReady, sizeof(int)) == -1)
      goto cleanupandret;
   if (write(fd, (int *) &n->DesiredOutputReady, sizeof(int)) == -1)
      goto cleanupandret;
   if (write(fd, (int *) &n->Learned, sizeof(int)) == -1)
      goto cleanupandret;
		
   if (write(fd, n->InputVals, n->NumInputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->DesiredOutputVals, n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->IHWeights, n->NumInputs * n->NumHidden * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->PrevDeltaIH, n->NumInputs * n->NumHidden * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->PrevDeltaHO, n->NumHidden * n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->PrevDeltaBH, n->NumBias * n->NumHidden * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->PrevDeltaBO, n->NumBias * n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->HiddenVals, n->NumHidden * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->HiddenBetas, n->NumHidden * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->HOWeights, n->NumHidden * n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->BiasVals, n->NumBias * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->BHWeights, n->NumBias * n->NumHidden * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->BOWeights, n->NumBias * n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->OutputVals, n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
   if (write(fd, n->OutputBetas, n->NumOutputs * sizeof(float)) == -1)
      goto cleanupandret;
		
   returncode = 0;
		
cleanupandret:
   close(fd);
	
   return returncode;
}
