#include "va.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

struct IsoNode
{
  int num;
  double value;
  struct IsoNode *next;
};

double IsotonicRegression(const struct Calibrator *cali, const double label, const double score) {
  int num_ex = cali->num_ex + 1;
  int i, j;

  // create linked list
  IsoNode *p, *h, *c;
  h = new IsoNode;
  h->next = NULL;
  p = h;
  for (i = 1; i < num_ex; ++i) {
    c = new IsoNode;
    p->next = c;
    c->next = NULL;
    p = c;
  }

  // insert label
  i = 0;
  p = h;
  int flag = 0;
  while (i < num_ex-1) {
    if (score <= cali->scores[i] && !flag) {
      p->num = 1;
      p->value = label;
      p = p->next;
      j = i;
      flag = 1;
    }
    p->num = 1;
    p->value = cali->labels[i++];
    p = p->next;
  }
  if (p != NULL) {
    p->num = 1;
    p->value = label;
    j = num_ex-1;
  }

  // pava
  p = h;
  while (1) {
    int inc = 0;
    for (p = h; p->next != NULL; p = p->next) {
      c = p->next;
      if ((p->value) > (c->value)) {
        double value = ((p->num * p->value) + (c->num * c->value)) / (p->num + c->num);
        p->value = value;
        p->num = p->num + c->num;
        p->next = c->next;
        inc = 1;
        delete c;
        break;
      }
    }
    if (inc == 0) {
      break;
    }
  }

  // get g(s(x))
  double value = 0;
  for (p = h; p != NULL; p = p->next) {
    j -= p->num;
    if (j < 0) {
      value = p->value;
      break;
    }
  }

  // free
  for (p = h; p != NULL; ) {
    c = p;
    p = p->next;
    delete c;
  }

  return value;
}

Model *TrainVA(const struct Problem *train, const struct Parameter *param) {
  Model *model = new Model;
  model->param = *param;
  double ratio = param->ratio;
  int num_classes;
  int num_ex = train->num_ex;
  int num_train = static_cast<int>(num_ex*ratio);
  int num_cali = num_ex - num_train;

  int *perm = new int[num_ex];
  int *start = NULL;
  int *label = NULL;
  int *count = NULL;
  GroupClasses(train, &num_classes, &label, &start, &count, perm);

  if (num_classes == 1) {
    std::cerr << "WARNING: training set only has one class. See README for details." << std::endl;
  }

  int *index;
  int *selected = new int[num_ex];
  clone(index, perm, num_ex);

  std::random_device rd;
  std::mt19937 g(rd());
  for (int i = 0; i < num_classes; ++i) {
    std::shuffle(index+start[i], index+start[i]+count[i], g);
  }
  for (int i = 0; i < num_ex; ++i) {
    selected[i] = 0;
  }

  int fold_start = 0;
  double curr_label = train->y[index[0]];
  for (int i = 1; i < num_ex; ++i) {
    double new_label = train->y[index[i]];
    if (new_label != curr_label || i == num_ex-1) {
      if (i == num_ex-1) {
        ++i;
      }
      int num_class = i - fold_start;
      int num_selected = i*num_train/num_ex - fold_start*num_train/num_ex;
      if (num_selected == 0) {
        num_selected = 1;
        std::cerr << "WARNING: at least one instance per class." << std::endl;
      }
      for (int j = 0; j < num_class; ++j) {
        std::uniform_int_distribution<int> ui(0, num_class-j-1);
        if (ui(g) < num_selected) {
          selected[fold_start+j] = 1;
          num_selected = num_selected - 1;
        }
      }
      fold_start = i;
      curr_label = new_label;
    }
  }

  struct Problem proper;
  proper.num_ex = num_train;
  proper.x = new Node*[proper.num_ex];
  proper.y = new double[proper.num_ex];
  struct Problem calibrator;
  calibrator.num_ex = num_cali;
  calibrator.x = new Node*[calibrator.num_ex];
  calibrator.y = new double[calibrator.num_ex];

  int index1 = 0, index2 = 0;
  for (int i = 0; i < num_ex; ++i) {
    if (selected[i] == 1) {
      proper.x[index1] = train->x[index[i]];
      proper.y[index1] = train->y[index[i]];
      ++index1;
    } else {
      calibrator.x[index2] = train->x[index[i]];
      calibrator.y[index2] = train->y[index[i]];
      ++index2;
    }
  }

  model->svm_model = TrainSVM(&proper, param->svm_param);

  Calibrator *cali = new Calibrator;
  cali->num_ex = num_cali;
  cali->scores = new double[num_cali];
  cali->labels = new double[num_cali];

  for(int i = 0; i < num_cali; ++i) {
    double *decision_values = new double[num_classes*(num_classes-1)/2];
    PredictSVMValues(model->svm_model, calibrator.x[i], decision_values);
    cali->scores[i] = decision_values[0];
    cali->labels[i] = (calibrator.y[i]==label[0]) ? 1 : 0;
    delete[] decision_values;
  }
  QuickSortIndex(cali->scores, cali->labels, 0, static_cast<size_t>(num_cali)-1);
  model->cali = cali;
  model->num_classes = num_classes;
  model->num_ex = num_ex;
  clone(model->labels, model->svm_model->labels, num_classes);

  delete[] selected;
  delete[] index;
  delete[] perm;
  delete[] start;
  delete[] label;
  delete[] count;
  delete[] proper.x;
  delete[] proper.y;
  delete[] calibrator.x;
  delete[] calibrator.y;

  return model;
}

double PredictVA(const struct Model *model, const struct Node *x, double &lower, double &upper, double **avg_prob) {
  int num_classes = model->num_classes;
  double *decision_values;

  decision_values = new double[num_classes*(num_classes-1)/2];
  double predicted_label = PredictSVMValues(model->svm_model, x, decision_values);

  lower = IsotonicRegression(model->cali, 0, decision_values[0]);
  upper = IsotonicRegression(model->cali, 1, decision_values[0]);

  *avg_prob = new double[num_classes];
  (*avg_prob)[0] = (lower + upper) / 2;
  (*avg_prob)[1] = 1 - (*avg_prob)[0];
  if (model->param.calibrated == 1) {
    if ((*avg_prob)[0] > (*avg_prob)[1]) {
      predicted_label = model->labels[0];
    } else {
      predicted_label = model->labels[1];
    }
  }
  if (predicted_label == model->labels[1]) {
    double tmp = lower;
    lower = 1 - upper;
    upper = 1 - tmp;
  }

  delete[] decision_values;

  return predicted_label;
}

void CrossValidation(const struct Problem *prob, const struct Parameter *param,
    double *predicted_labels, double *lower_bounds, double *upper_bounds,
    double *brier, double *logloss) {
  int num_folds = param->num_folds;
  int num_ex = prob->num_ex;
  int num_classes;

  int *fold_start;
  int *perm = new int[num_ex];

  if (num_folds > num_ex) {
    num_folds = num_ex;
    std::cerr << "WARNING: number of folds > number of data. Will use number of folds = number of data instead (i.e., leave-one-out cross validation)" << std::endl;
  }
  fold_start = new int[num_folds+1];

  if (num_folds < num_ex) {
    int *start = NULL;
    int *label = NULL;
    int *count = NULL;
    GroupClasses(prob, &num_classes, &label, &start, &count, perm);

    int *fold_count = new int[num_folds];
    int *index = new int[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      index[i] = perm[i];
    }
    std::random_device rd;
    std::mt19937 g(rd());
    for (int i = 0; i < num_classes; ++i) {
      std::shuffle(index+start[i], index+start[i]+count[i], g);
    }

    for (int i = 0; i < num_folds; ++i) {
      fold_count[i] = 0;
      for (int c = 0; c < num_classes; ++c) {
        fold_count[i] += (i+1)*count[c]/num_folds - i*count[c]/num_folds;
      }
    }

    fold_start[0] = 0;
    for (int i = 1; i <= num_folds; ++i) {
      fold_start[i] = fold_start[i-1] + fold_count[i-1];
    }
    for (int c = 0; c < num_classes; ++c) {
      for (int i = 0; i < num_folds; ++i) {
        int begin = start[c] + i*count[c]/num_folds;
        int end = start[c] + (i+1)*count[c]/num_folds;
        for (int j = begin; j < end; ++j) {
          perm[fold_start[i]] = index[j];
          fold_start[i]++;
        }
      }
    }
    fold_start[0] = 0;
    for (int i = 1; i <= num_folds; ++i) {
      fold_start[i] = fold_start[i-1] + fold_count[i-1];
    }
    delete[] start;
    delete[] label;
    delete[] count;
    delete[] index;
    delete[] fold_count;

  } else {

    for (int i = 0; i < num_ex; ++i) {
      perm[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm, perm+num_ex, g);
    fold_start[0] = 0;
    for (int i = 1; i <= num_folds; ++i) {
      fold_start[i] = fold_start[i-1] + (i+1)*num_ex/num_folds - i*num_ex/num_folds;
    }
  }

  for (int i = 0; i < num_folds; ++i) {
    int begin = fold_start[i];
    int end = fold_start[i+1];
    int k = 0;
    struct Problem subprob;

    subprob.num_ex = num_ex - (end-begin);
    subprob.x = new Node*[subprob.num_ex];
    subprob.y = new double[subprob.num_ex];

    for (int j = 0; j < begin; ++j) {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    for (int j = end; j < num_ex; ++j) {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }

    struct Model *submodel = TrainVA(&subprob, param);

    if (param->probability == 1) {
      for (int j = 0; j < submodel->num_classes; ++j) {
        std::cout << submodel->labels[j] << "        ";
      }
      std::cout << '\n';
    }

    for (int j = begin; j < end; ++j) {
      double *avg_prob = NULL;
      brier[perm[j]] = 0;

      predicted_labels[perm[j]] = PredictVA(submodel, prob->x[perm[j]], lower_bounds[perm[j]], upper_bounds[perm[j]], &avg_prob);

      for (k = 0; k < submodel->num_classes; ++k) {
        if (submodel->labels[k] == prob->y[perm[j]]) {
          brier[perm[j]] += (1-avg_prob[k]) * (1-avg_prob[k]);
          double tmp = std::fmax(std::fmin(avg_prob[k], 1-kEpsilon), kEpsilon);
          logloss[perm[j]] = - std::log(tmp);
        } else {
          brier[perm[j]] += avg_prob[k] * avg_prob[k];
        }
      }
      if (param->probability == 1) {
        for (k = 0; k < submodel->num_classes; ++k) {
          std::cout << avg_prob[k] << ' ';
        }
        std::cout << '\n';
      }
      delete[] avg_prob;
    }
    FreeModel(submodel);
    delete[] subprob.x;
    delete[] subprob.y;
  }
  delete[] fold_start;
  delete[] perm;

  return;
}

void OnlinePredict(const struct Problem *prob, const struct Parameter *param,
    double *predicted_labels, int *indices,
    double *lower_bounds, double *upper_bounds,
    double *brier, double *logloss) {
  int num_ex = prob->num_ex;

  for (int i = 0; i < num_ex; ++i) {
    indices[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices, indices+num_ex, g);

  Problem subprob;
  subprob.x = new Node*[num_ex];
  subprob.y = new double[num_ex];

  for (int i = 0; i < num_ex; ++i) {
    subprob.x[i] = prob->x[indices[i]];
    subprob.y[i] = prob->y[indices[i]];
  }

  for (int i = 4; i < num_ex; ++i) {
    double *avg_prob = NULL;
    brier[i] = 0;
    subprob.num_ex = i;
    Model *submodel = TrainVA(&subprob, param);
    predicted_labels[i] = PredictVA(submodel, subprob.x[i], lower_bounds[i], upper_bounds[i], &avg_prob);
    for (int j = 0; j < submodel->num_classes; ++j) {
      if (submodel->labels[j] == subprob.y[i]) {
        brier[i] += (1-avg_prob[j]) * (1-avg_prob[j]);
        double tmp = std::fmax(std::fmin(avg_prob[j], 1-kEpsilon), kEpsilon);
        logloss[i] = - std::log(tmp);
      } else {
        brier[i] += avg_prob[j] * avg_prob[j];
      }
    }
    if (param->probability == 1) {
      for (int j = 0; j < submodel->num_classes; ++j) {
        std::cout << avg_prob[j] << ' ';
      }
      std::cout << '\n';
    }
    FreeModel(submodel);
    delete[] avg_prob;
  }
  delete[] subprob.x;
  delete[] subprob.y;

  return;
}

int SaveModel(const char *model_file_name, const struct Model *model) {
  std::ofstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return -1;
  }

  const Parameter &param = model->param;

  model_file << "ratio " << param.ratio << '\n';
  model_file << "probability " << param.probability << '\n';
  model_file << "calibrated " << param.calibrated << '\n';

  if (model->cali != NULL) {
    model_file << "num_cali " << model->cali->num_ex << '\n';
  }

  if (model->svm_model != NULL) {
    SaveSVMModel(model_file, model->svm_model);
  }

  if (model->cali != NULL) {
    int num_cali = model->cali->num_ex;
    model_file << "cali_scores\n";
    for (int i = 0; i < num_cali; ++i) {
      model_file << model->cali->scores[i] << ' ';
    }
    model_file << '\n';
    model_file << "cali_labels\n";
    for (int i = 0; i < num_cali; ++i) {
      model_file << model->cali->labels[i] << ' ';
    }
    model_file << '\n';
  }

  if (model_file.bad() || model_file.fail()) {
    model_file.close();
    return -1;
  }

  model_file.close();

  return 0;
}

Model *LoadModel(const char *model_file_name) {
  std::ifstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return NULL;
  }

  Model *model = new Model;

  Parameter &param = model->param;
  param.load_model = 1;
  model->labels = NULL;
  model->cali = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "ratio") == 0) {
      model_file >> param.ratio;
    } else
    if (std::strcmp(cmd, "probability") == 0) {
      model_file >> param.probability;
    } else
    if (std::strcmp(cmd, "calibrated") == 0) {
      model_file >> param.calibrated;
    } else
    if (std::strcmp(cmd, "num_cali") == 0) {
      model->cali = new Calibrator;
      model_file >> model->cali->num_ex;
    } else
    if (std::strcmp(cmd, "svm_model") == 0) {
      model->svm_model = LoadSVMModel(model_file);
      if (model->svm_model == NULL) {
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
      model->num_ex = model->svm_model->num_ex;
      model->num_classes = model->svm_model->num_classes;
      clone(model->labels, model->svm_model->labels, model->num_classes);
      model->param.svm_param = &model->svm_model->param;
    } else
    if (std::strcmp(cmd, "cali_scores") == 0) {
      int num_cali = model->cali->num_ex;
      model->cali->scores = new double[num_cali];
      for (int i = 0; i < num_cali; ++i) {
        model_file >> model->cali->scores[i];
      }
    } else
    if (std::strcmp(cmd, "cali_labels") == 0) {
      int num_cali = model->cali->num_ex;
      model->cali->labels = new double[num_cali];
      for (int i = 0; i < num_cali; ++i) {
        model_file >> model->cali->labels[i];
      }
      break;
    } else {
      std::cerr << "Unknown text in model file: " << cmd << std::endl;
      FreeModel(model);
      delete model;
      model_file.close();
      return NULL;
    }
  }
  model_file.close();

  return model;
}

void FreeModel(struct Model *model) {
  if (model->svm_model != NULL) {
    FreeSVMModel(&(model->svm_model));
    delete model->svm_model;
    model->svm_model = NULL;
  }

  if (model->cali != NULL) {
    if (model->cali->scores != NULL) {
      delete[] model->cali->scores;
    }
    if (model->cali->labels != NULL) {
      delete[] model->cali->labels;
    }
    delete model->cali;
    model->cali = NULL;
  }

  if (model->labels != NULL) {
    delete[] model->labels;
    model->labels = NULL;
  }

  delete model;
  model = NULL;

  return;
}

void FreeParam(struct Parameter *param) {
  if (param->svm_param != NULL) {
    FreeSVMParam(param->svm_param);
    param->svm_param = NULL;
  }

  return;
}

const char *CheckParameter(const struct Parameter *param) {
  if (param->save_model == 1 && param->load_model == 1) {
    return "cannot save and load model at the same time";
  }

  if (param->ratio <= 0 ||
      param->ratio >= 1) {
    return "ratio should be > 0 and < 1";
  }

  if (param->svm_param == NULL) {
    return "no svm parameter";
  }

  return CheckSVMParameter(param->svm_param);
}