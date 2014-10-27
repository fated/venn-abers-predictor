#include "va.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *data_file_name, char *output_file_name);

struct Parameter param;

int main(int argc, char *argv[]) {
  char data_file_name[256];
  char output_file_name[256];
  struct Problem *prob;
  int num_correct = 0;
  double avg_lower_bound = 0, avg_upper_bound = 0, avg_brier = 0, avg_logloss = 0;
  double *predict_labels = NULL, *lower_bounds = NULL, *upper_bounds = NULL, *brier = NULL, *logloss = NULL;
  const char *error_message;

  ParseCommandLine(argc, argv, data_file_name, output_file_name);
  error_message = CheckParameter(&param);

  if (error_message != NULL) {
    std::cerr << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  prob = ReadProblem(data_file_name);

  if (param.svm_param->gamma == 0) {
    param.svm_param->gamma = 1.0 / prob->max_index;
  }

  std::ofstream output_file(output_file_name);
  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  predict_labels = new double[prob->num_ex];
  lower_bounds = new double[prob->num_ex];
  upper_bounds = new double[prob->num_ex];
  brier = new double[prob->num_ex];
  logloss = new double[prob->num_ex];

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();

  CrossValidation(prob, &param, predict_labels, lower_bounds, upper_bounds, brier, logloss);

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < prob->num_ex; ++i) {
    avg_lower_bound += lower_bounds[i];
    avg_upper_bound += upper_bounds[i];
    avg_brier += brier[i];
    avg_logloss += logloss[i];

    output_file << predict_labels[i] << ' ' << lower_bounds[i] << ' ' << upper_bounds[i] << '\n';
    if (predict_labels[i] == prob->y[i]) {
      ++num_correct;
    }
  }
  avg_lower_bound /= prob->num_ex;
  avg_upper_bound /= prob->num_ex;
  avg_brier /= prob->num_ex;
  avg_logloss /= prob->num_ex;

  std::cout << "CV Accuracy: " << 100.0*num_correct/(prob->num_ex) << '%'
            << " (" << num_correct << '/' << prob->num_ex << ") "
            << "Probabilities: [" << std::fixed << std::setprecision(4) << 100*avg_lower_bound << "%, "
            << 100*avg_upper_bound << "%] "
            << "Brier Score: " << avg_brier << ' '
            << "Logarithmic Loss: " << avg_logloss << '\n';
  output_file.close();

  std::cout << "Time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0 << " s\n";

  FreeProblem(prob);
  FreeParam(&param);
  delete[] predict_labels;
  delete[] lower_bounds;
  delete[] upper_bounds;
  delete[] brier;
  delete[] logloss;

  return 0;
}

void ExitWithHelp() {
  std::cout << "Usage: va-cv [options] data_file [output_file]\n"
            << "options:\n"
            << "  -a ratio : set ratio of proper training set takes of all training set in Venn-ABERS (default 0.7)\n"
            << "  -b probability estimates : whether to output probability estimates for all labels, 0 or 1 (default 0)\n"
            << "  -v num_folds : set number of folders in cross validation (default 5)\n"
            << "  -t svm_type : set type of SVM (default 0)\n"
            << "    0 -- C-SVC    (multi-class classification)\n"
            << "    1 -- nu-SVC   (multi-class classification)\n"
            << "  -k kernel_type : set type of kernel function (default 2)\n"
            << "    0 -- linear: u'*v\n"
            << "    1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
            << "    2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            << "    3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
            << "    4 -- precomputed kernel (kernel values in training_set_file)\n"
            << "  -d degree : set degree in kernel function (default 3)\n"
            << "  -g gamma : set gamma in kernel function (default 1/num_features)\n"
            << "  -r coef0 : set coef0 in kernel function (default 0)\n"
            << "  -c cost : set the parameter C of C-SVC (default 1)\n"
            << "  -n nu : set the parameter nu of nu-SVC (default 0.5)\n"
            << "  -m cachesize : set cache memory size in MB (default 100)\n"
            << "  -e epsilon : set tolerance of termination criterion (default 0.001)\n"
            << "  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
            << "  -wi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
            << "  -q : quiet mode (no outputs)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *data_file_name, char *output_file_name) {
  int i;
  param.save_model = 0;
  param.load_model = 0;
  param.num_folds = 5;
  param.probability = 0;
  param.svm_param = new SVMParameter;
  InitSVMParam(param.svm_param);

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+1) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 'a': {
        ++i;
        param.ratio = std::atof(argv[i]);
        break;
      }
      case 'b': {
        ++i;
        param.probability = std::atoi(argv[i]);
        break;
      }
      case 'v': {
        ++i;
        param.num_folds = std::atoi(argv[i]);
        if (param.num_folds < 2) {
          std::cerr << "n-fold cross validation: n must >= 2" << std::endl;
          exit(EXIT_FAILURE);
        }
        break;
      }
      case 't': {
        ++i;
        param.svm_param->svm_type = std::atoi(argv[i]);
        break;
      }
      case 'k': {
        ++i;
        param.svm_param->kernel_type = std::atoi(argv[i]);
        break;
      }
      case 'd': {
        ++i;
        param.svm_param->degree = std::atoi(argv[i]);
        break;
      }
      case 'g': {
        ++i;
        param.svm_param->gamma = std::atof(argv[i]);
        break;
      }
      case 'r': {
        ++i;
        param.svm_param->coef0 = std::atof(argv[i]);
        break;
      }
      case 'n': {
        ++i;
        param.svm_param->nu = std::atof(argv[i]);
        break;
      }
      case 'm': {
        ++i;
        param.svm_param->cache_size = std::atof(argv[i]);
        break;
      }
      case 'c': {
        ++i;
        param.svm_param->C = std::atof(argv[i]);
        break;
      }
      case 'e': {
        ++i;
        param.svm_param->eps = std::atof(argv[i]);
        break;
      }
      case 'h': {
        ++i;
        param.svm_param->shrinking = std::atoi(argv[i]);
        break;
      }
      case 'q': {
        SetPrintNull();
        break;
      }
      case 'w': {  // weights [option]: '-w1' means weight of '1'
        ++i;
        ++param.svm_param->num_weights;
        param.svm_param->weight_labels = (int *)realloc(param.svm_param->weight_labels, sizeof(int)*static_cast<unsigned long int>(param.svm_param->num_weights));
        param.svm_param->weights = (double *)realloc(param.svm_param->weights, sizeof(double)*static_cast<unsigned long int>(param.svm_param->num_weights));
        param.svm_param->weight_labels[param.svm_param->num_weights-1] = std::atoi(&argv[i-1][3]); // get and convert 'i' to int
        param.svm_param->weights[param.svm_param->num_weights-1] = std::atof(argv[i]);
        break;
        // TODO: change realloc function
      }
      default: {
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
      }
    }
  }

  if (i >= argc)
    ExitWithHelp();
  strcpy(data_file_name, argv[i]);
  if ((i+1) < argc) {
    std::strcpy(output_file_name, argv[i+1]);
  } else {
    char *p = std::strrchr(argv[i],'/');
    if (p == NULL) {
      p = argv[i];
    } else {
      ++p;
    }
    std::sprintf(output_file_name, "%s_output", p);
  }

  return;
}