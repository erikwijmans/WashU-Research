#include "highOrder.h"
#include "gurobi_c++.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

void place::createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
  const std::vector<place::node> & nodes, std::unordered_map<std::vector<int>, double> &
    highOrder) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node * prevNode = &nodes[0];
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
        prevNode = &n;
        ++i;
      } else {
        numberOfLabels.push_back(i);
        i = 1;
        prevNode = &n;
      }
    }
    numberOfLabels.push_back(i);
  }

  Eigen::ArrayXH hMap (floorPlan.rows, floorPlan.cols);
  for (int a = 0, offset = 0; a < numberOfLabels.size(); ++a) {
    for (int b = 0; b < std::min(numberOfLabels[a], 3); ++b) {
      auto & currentNode = nodes[b + offset];
      auto & currentScan = scans[currentNode.color][currentNode.s.rotation];
      auto & zeroZero = zeroZeros[currentNode.color][currentNode.s.rotation];
      const int xOffset = currentNode.s.x - zeroZero[0],
        yOffset = currentNode.s.y - zeroZero[1];

      for (int j = 0; j < currentScan.rows(); ++j) {
        if (j + yOffset < 0 || j + yOffset >= floorPlan.rows)
          continue;
        const uchar * src = floorPlan.ptr<uchar>(j + yOffset);
        for (int i = 0; i < currentScan.cols(); ++i) {
          if (i + xOffset < 0 || i + xOffset >= floorPlan.cols)
            continue;
          if (src[i + xOffset] != 255) {
            if (localGroup(currentScan, j, i, 2)) {
              hMap(j+yOffset, i + xOffset).incident.push_back(b + offset);
              hMap(j+yOffset, i + xOffset).weight += currentNode.w;
            }
          }
        }
      }
    }
    offset += numberOfLabels[a];
  }

  place::hOrder * data = hMap.data();
  for (int i = 0; i < hMap.size(); ++i) {
    if ((data + i)->incident.size() != 0) {
      // const double scale = harmonic((data + i)->incident.size(), 0.0);
      (data + i)->weight /= (data + i)->incident.size();
      // (data + i)->weight *= scale;
    }
  }

  for (int i = 0; i < hMap.size(); ++i) {
    std::vector<int> & key = (data + i)->incident;
    if (key.size() != 0 && (data + i)->weight > 0.0) {
      auto it = highOrder.find(key);
      if (it != highOrder.end())
        it->second += (data + i)->weight;
      else
        highOrder.emplace(key, (data + i)->weight);
    }
  }

  double average = 0.0, aveTerms = 0.0;
  for (auto & it : highOrder) {
    average += it.second;
    aveTerms += it.first.size();
  }
  average /= highOrder.size();
  aveTerms /= highOrder.size();

  double sigma = 0.0, sigTerms = 0.0;
  for (auto & it : highOrder) {
    sigma += (it.second - average)*(it.second -average);
    sigTerms += (it.first.size() - aveTerms) * (it.first.size() - aveTerms);
  }
  sigma /= (highOrder.size() - 1);
  sigma = sqrt(sigma);

  sigTerms /= (highOrder.size() - 1);
  sigTerms = sqrt(sigTerms);

  std::cout << "average: " << average << "   sigma: " << sigma << std::endl;

  for (auto & it : highOrder) {
    it.second = std::max(0.0,(((it.second - average)/(sigma) + 1.0)/2.0));
    const double significance = (it.first.size() - aveTerms)/sigTerms;
    if(significance < 10000)
      highOrder.erase(it.first);
  }
}

void place::displayHighOrder(const std::unordered_map<std::vector<int>, double> highOrder,
  const std::vector<place::node> & nodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  for (auto & it : highOrder) {
    auto & key = it.first;
    cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    for (auto & i : key) {
      const place::node & nodeA = nodes[i];

      auto & aScan = scans[nodeA.color][nodeA.s.rotation];

      auto & zeroZeroA = zeroZeros[nodeA.color][nodeA.s.rotation];

      int yOffset = nodeA.s.y - zeroZeroA[1];
      int xOffset = nodeA.s.x - zeroZeroA[0];
      for (int k = 0; k < aScan.cols(); ++k) {
        for (int l = 0; l < aScan.rows(); ++l) {
          if (l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if (k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

          if (aScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=0;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=255;
          }
        }
      }
    }
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", output);
    if (!FLAGS_quiteMode) {
      std::cout << it.second << std::endl;
      for (auto & i : key)
        std::cout << i << "_";
      std::cout << std::endl;
    }
    cv::waitKey(0);
    ~output;
  }
}

static void condenseStack(std::vector<GRBVar> & stacked,
  GRBModel & model) {
  if (stacked.size() == 2) {
    GRBVar first = stacked.back();
    stacked.pop_back();
    GRBVar second = stacked.back();
    stacked.pop_back();

    GRBVar newStack = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    model.update();
    model.addQConstr(first * second,
      GRB_EQUAL, newStack);
    stacked.push_back(newStack);

  } else if (stacked.size() == 1) return;
  else {
    std::vector<GRBVar> firstHalf (stacked.begin(),
      stacked.begin() + stacked.size()/2);
    std::vector<GRBVar> secondHalf(stacked.begin() + stacked.size()/2,
      stacked.end());

    condenseStack(firstHalf, model);
    condenseStack(secondHalf, model);
    stacked.clear();
    stacked.insert(stacked.end(), firstHalf.begin(), firstHalf.end());
    stacked.insert(stacked.end(), secondHalf.begin(), secondHalf.end());
  }
}

static void stackTerms(const std::vector<int> & toStack,
  const GRBVar * varList, GRBModel & model,
  std::map<std::pair<int,int>, GRBVar > & preStacked,
  std::vector<GRBVar> & stacked) {
  int i = 0;
  for (; i < toStack.size() - 1; i+=2) {
    std::pair<int, int> key (toStack[i], toStack[i+1]);
    auto it = preStacked.find(key);
    if (it == preStacked.end()) {
      GRBVar newStack = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
      model.update();
      model.addQConstr(varList[toStack[i]] * varList[toStack[i+1]],
        GRB_EQUAL, newStack);
      preStacked.emplace(key, newStack);
      stacked.push_back(newStack);
    } else {
      stacked.push_back(it->second);
    }
  }
  for (; i < toStack.size(); ++i) {
    if (stacked.size() > 1) {
      GRBVar newStack = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
      model.update();
      model.addQConstr(varList[toStack[i]] * stacked.back(),
        GRB_EQUAL, newStack);
      stacked.pop_back();
      stacked.push_back(newStack);
    } else {
      stacked.push_back(varList[toStack[i]]);
    }
  }
  while (stacked.size() > 2)
    condenseStack(stacked, model);
}

void place::MIPSolver(const Eigen::MatrixXE & adjacencyMatrix,
  const std::unordered_map<std::vector<int>, double> & highOrder, const std::vector<place::node> & nodes,
  std::vector<const place::node *> & bestNodes) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node * prevNode = &nodes[0];
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
        prevNode = &n;
        ++i;
      } else {
        numberOfLabels.push_back(i);
        i = 1;
        prevNode = &n;
      }
    }
    numberOfLabels.push_back(i);
  }

  const int numVars = numberOfLabels.size();
  const int numOpts = nodes.size();
  try {
    GRBEnv env = GRBEnv();
    env.set("TimeLimit", "600");

    GRBModel model = GRBModel(env);

    double * upperBound = new double [numOpts];
    char * type = new char [numOpts];
    for (int i = 0; i < numOpts; ++i) {
      upperBound[i] = 1.0;
      type[i] = GRB_BINARY;
    }

    GRBVar * varList = model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
    GRBVar * inverseVarList = model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
    delete [] upperBound;
    delete [] type;
    // Integrate new variables
    model.update();
    for (int i = 0; i < numOpts; ++i) {
      model.addConstr(varList[i] + inverseVarList[i], GRB_EQUAL, 1.0);
    }

    GRBQuadExpr objective = 0.0;
    for (int i = 0; i < numOpts; ++i) {
      for (int j = i + 1; j < numOpts; ++j) {
        if (adjacencyMatrix(j,i).w == 0.0)
          continue;

        objective += (adjacencyMatrix(j,i).w + adjacencyMatrix(j,i).shotW)*varList[i]*varList[j];
      }
      const place::posInfo & currentScore = nodes[i].s;
      double scanExplained =
        (currentScore.scanPixels - currentScore.scanFP)/(currentScore.scanPixels);
      double fpExplained =
      (currentScore.fpPixels - currentScore.fpScan)/(currentScore.fpPixels);

      objective += varList[i]*(fpExplained + scanExplained)/2.0;
    }

    for (int i = 0, offset = 0; i < numVars; ++i) {
      GRBLinExpr constr = 0.0;
      double * coeff = new double [numberOfLabels[i]];
      for (int a = 0; a < numberOfLabels[i]; ++ a)
        coeff[a] = 1.0;

      constr.addTerms(coeff, varList + offset, numberOfLabels[i]);
      model.addConstr(constr, GRB_LESS_EQUAL, 1.0);
      offset += numberOfLabels[i];
      delete [] coeff;
    }


    /*for (auto & it : highOrder) {
      auto & incident = it.first;
      for (auto & i : incident)
        objective += varList[i]*it.second;
    }
*/
    std::map<std::pair<int, int>, GRBVar > termCondense;
    for (auto & it : highOrder) {
      auto & incident = it.first;
      /*if (incident.size() == 2) {
        objective -= inverseVarList[incident[0]]*inverseVarList[incident[1]]*it.second;
      } else if (incident.size() == 1) {

      }else*/ if(incident.size() > 3) {
        std::vector<GRBVar> final;
        stackTerms(incident, inverseVarList, model, termCondense, final);
        objective -= final[0]*final[1]*it.second;
      }
    }
    model.update();
    model.setObjective(objective, GRB_MAXIMIZE);
    model.optimize();

    for (int i = 0, offset = 0, k = 0; i < numOpts; ++i) {
      if (varList[i].get(GRB_DoubleAttr_X) == 1.0) {
        bestNodes.push_back(&(nodes[i]));
        std::cout << i - offset << "_";
      }
      if (numberOfLabels[k] == i + 1 - offset)
        offset += numberOfLabels[k++];
    }
    std::cout << std::endl;
    std::cout << "Labeling found for " << bestNodes.size() << " out of " << numVars << " options" << std::endl;
  } catch(GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch(...) {
    std::cout << "Exception during optimization" << std::endl;
  }
}