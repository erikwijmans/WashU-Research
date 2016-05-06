#include <omp.h>
#include <math.h>

int main() {

  #pragma omp target teams num_teams(4) thread_limit(6)
  {
    double newPi = 0.0;
    #pragma omp distribute parallel for
    for (int j = 0; j < 1000; ++j) {
      newPi = 0.0;
      // #pragma omp parallel for reduction(+: newPi)
      for (int i = 0; i < 1000000; ++i) {
        newPi += sqrt(pow(-1, i)/((i+1)*(i+1)));
      }
    }
  }
}