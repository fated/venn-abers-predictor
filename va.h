#ifndef VA_VM_H_
#define VA_VM_H_

#include "utilities.h"
#include "svm.h"

struct Calibrator {
  int num_ex;
  double *scores;
  double *labels;
};

struct Parameter {
  struct SVMParameter *svm_param;
  int save_model;
  int load_model;
  int num_folds;
  int probability;
  int calibrated;
  double ratio;
};

struct Model {
  struct Parameter param;
  struct SVMModel *svm_model;
  struct Calibrator *cali;
  int num_ex;
  int num_classes;
  int *labels;
};

Model *TrainVA(const struct Problem *train, const struct Parameter *param);
double PredictVA(const struct Model *model, const struct Node *x, double &lower, double &upper, double **avg_prob);
void CrossValidation(const struct Problem *prob, const struct Parameter *param, double *predict_labels, double *lower_bounds, double *upper_bounds, double *brier, double *logloss);
void OnlinePredict(const struct Problem *prob, const struct Parameter *param, double *predict_labels, int *indices, double *lower_bounds, double *upper_bounds, double *brier, double *logloss);

int SaveModel(const char *model_file_name, const struct Model *model);
Model *LoadModel(const char *model_file_name);
void FreeModel(struct Model *model);

void FreeParam(struct Parameter *param);
const char *CheckParameter(const struct Parameter *param);

#endif  // VA_VM_H_