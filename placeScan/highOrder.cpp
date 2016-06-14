#include "highOrder.h"
#include "gurobi_c++.h"

#include <iostream>
#include <list>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static void dispHMap(const Eigen::ArrayXH &hMap,
                     const multi::Labeler::map &highOrder) {
  double average = 0;
  size_t count = 0;
  auto dataPtr = hMap.data();
  for (int i = 0; i < hMap.size(); ++i) {
    if (!(dataPtr + i)->count)
      continue;
    auto it = highOrder.find((dataPtr + i)->incident);
    if (it != highOrder.cend()) {
      assert(it->second.c != 0);
      average += it->second.w;
      ++count;
    }
  }
  average /= count;

  double sigma = 0;
  for (int i = 0; i < hMap.size(); ++i) {
    if (!(dataPtr + i)->count)
      continue;
    auto it = highOrder.find((dataPtr + i)->incident);
    if (it != highOrder.cend())
      sigma += (it->second.w - average) * (it->second.w - average);
  }
  sigma /= count - 1;
  sigma = sqrt(sigma);

  std::cout << average << "  " << sigma << std::endl;

  cv::Mat heatMap(hMap.rows(), hMap.cols(), CV_8UC3);
  fpColor.copyTo(heatMap);
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar *dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if (!hMap(j, i).count)
        continue;
      auto it = highOrder.find(hMap(j, i).incident);
      if (it != highOrder.cend() && hMap(j, i).incident.size()) {
        const int gray = cv::saturate_cast<uchar>(
            255.0 * ((it->second.w - average) / sigma + 1.0) / 2.0);
        int r, g, b;
        if (gray < 128) {
          r = 0;
          g = 2 * gray;
          b = 255 - g;
        } else {
          r = 2 * (gray - 128);
          g = 255 - r;
          b = 0;
        }
        dst[3 * i + 0] = b;
        dst[3 * i + 1] = g;
        dst[3 * i + 2] = r;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

static double pseries(const int n, const double p = 1.0) {
  double sum = 1.0;
  for (int i = 2; i <= n; ++i) {
    sum += 1.0 / std::pow(i, p);
  }
  return sum;
}

static void dispHMap(Eigen::ArrayXH &hMap) {
  double average = 0;
  int count = 0;
  auto const dataPtr = hMap.data();
  for (int i = 0; i < hMap.size(); ++i) {
    const double weight = (dataPtr + i)->weight;
    if (weight) {
      average += weight;
      ++count;
    }
  }
  average /= count;

  double sigma = 0;
  for (int i = 0; i < hMap.size(); ++i) {
    const double weight = (dataPtr + i)->weight;
    if (weight) {
      sigma += (weight - average) * (weight - average);
    }
  }
  sigma /= count - 1;
  sigma = std::sqrt(sigma);

  cv::Mat heatMap(hMap.rows(), hMap.cols(), CV_8UC3);
  fpColor.copyTo(heatMap);
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar *dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if (hMap(j, i).weight) {
        const int gray = cv::saturate_cast<uchar>(
            255.0 * ((hMap(j, i).weight - average) / sigma + 1.0) / 2.0);
        int r, g, b;
        if (gray < 128) {
          r = 0;
          g = 2 * gray;
          b = 255 - g;
        } else {
          r = 2 * (gray - 128);
          g = 255 - r;
          b = 0;
        }
        dst[3 * i + 0] = b;
        dst[3 * i + 1] = g;
        dst[3 * i + 2] = r;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

void place::createHigherOrderTerms(
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros,
    const std::vector<place::R2Node> &nodes,
    const std::unordered_map<int, std::unordered_set<int>> &unwantedNeighbors,
    multi::Labeler::map &highOrder) {

  Eigen::ArrayXH hMap(floorPlan.rows, floorPlan.cols);
  for (int a = 0; a < nodes.size(); ++a) {
    auto &currentNode = nodes[a];
    auto &currentScan = scans[currentNode.color][currentNode.s.rotation];
    auto &zeroZero = zeroZeros[currentNode.color][currentNode.s.rotation];
    const int xOffset = currentNode.s.x - zeroZero[0],
              yOffset = currentNode.s.y - zeroZero[1];

    cv::Mat_<uchar> _fp = floorPlan;
    for (int j = 0; j < currentScan.rows(); ++j) {
      if (j + yOffset < 0 || j + yOffset >= floorPlan.rows)
        continue;
      for (int i = 0; i < currentScan.cols(); ++i) {
        if (i + xOffset < 0 || i + xOffset >= floorPlan.cols)
          continue;
        if (_fp(j + yOffset, i + xOffset) != 255 &&
            localGroup(currentScan, j, i, 10)) {
          auto &h = hMap(j + yOffset, i + xOffset);
          if (currentNode.locked) {
            h.weight += currentNode.w;
            h.owners.push_back(a);
          } else
            h.incident.push_back(a);
        }
      }
    }
  }

  auto const data = hMap.data();

  for (int i = 0; i < hMap.size(); ++i) {
    auto &owners = (data + i)->owners;
    std::vector<int> &incident = (data + i)->incident;
    for (auto &o : owners) {
      auto pair = unwantedNeighbors.find(o);
      if (pair == unwantedNeighbors.cend())
        continue;
      auto &exclusionSet = pair->second;
      for (auto it = incident.cbegin(); it != incident.cend();) {
        auto match = exclusionSet.find(*it);
        if (match != exclusionSet.cend())
          it = incident.erase(it);
        else
          ++it;
      }
    }
  }

  for (int i = 0; i < hMap.size(); ++i) {
    auto h = data + i;
    h->count = h->owners.size();
    if (h->count) {
      h->weight /= std::pow(h->count, 1.1);
    }
  }

  // dispHMap(hMap);

  for (int i = 0; i < hMap.size(); ++i) {
    std::vector<int> &key = (data + i)->incident;
    const double weight = (data + i)->weight;
    if (key.size() && (data + i)->count) {
      auto it = highOrder.find(key);
      if (it != highOrder.cend()) {
        it->second.w += weight;
        ++it->second.c;
      } else
        highOrder.emplace(key, multi::Labeler::s(weight, 1));
    }
  }

  for (auto it = highOrder.begin(); it != highOrder.end();) {
    it->second.w /= it->second.c;
    if (std::abs(it->second.w) <= 0.001 || it->first.size() == 0)
      it = highOrder.erase(it);
    else
      ++it;
  }

  double average = 0.0, aveTerms = 0.0;
  for (auto &it : highOrder) {
    average += it.second.w;
    aveTerms += it.first.size();
  }
  average /= highOrder.size();
  aveTerms /= highOrder.size();

  double sigma = 0.0, sigTerms = 0.0;
  for (auto &it : highOrder) {
    sigma += (it.second.w - average) * (it.second.w - average);
    sigTerms += (it.first.size() - aveTerms) * (it.first.size() - aveTerms);
  }
  sigma /= (highOrder.size() - 1);
  sigma = sqrt(sigma);

  sigTerms /= (highOrder.size() - 1);
  sigTerms = sqrt(sigTerms);

  std::cout << "average: " << average << "   sigma: " << sigma << std::endl;

  for (auto &pair : highOrder)
    pair.second.w = (pair.second.w - average) / (sigma);

  for (auto it = highOrder.cbegin(); it != highOrder.cend();)
    if (it->second.w <= -0.5)
      it = highOrder.erase(it);
    else
      ++it;

  // dispHMap(hMap, highOrder);
}

void place::displayHighOrder(
    const multi::Labeler::map highOrder,
    const std::vector<place::R2Node> &nodes,
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros) {

  for (auto &it : highOrder) {
    auto &key = it.first;
    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    for (auto &i : key) {
      const place::node &nodeA = nodes[i];

      auto &aScan = scans[nodeA.color][nodeA.s.rotation];

      auto &zeroZeroA = zeroZeros[nodeA.color][nodeA.s.rotation];

      int yOffset = nodeA.s.y - zeroZeroA[1];
      int xOffset = nodeA.s.x - zeroZeroA[0];
      for (int k = 0; k < aScan.cols(); ++k) {
        for (int l = 0; l < aScan.rows(); ++l) {
          if (l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if (k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

          if (aScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0] = 0;
            _output(l + yOffset, k + xOffset)[1] = 0;
            _output(l + yOffset, k + xOffset)[2] = 255;
          }
        }
      }
    }
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", output);
    if (!FLAGS_quietMode) {
      std::cout << it.second.w << ":  ";
      for (auto &i : key)
        std::cout << i << "_";
      std::cout << std::endl;
    }
    cv::waitKey(0);
    ~output;
  }
}

namespace std {
template <> struct hash<std::pair<GRBVar *, GRBVar *>> {
  std::hash<GRBVar *> h;
  std::size_t operator()(const std::pair<GRBVar *, GRBVar *> &k) const {
    size_t seed = h(k.first);
    seed ^= h(k.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // std

static std::vector<GRBVar *> condenseStack(
    const std::vector<GRBVar *> &toStack, GRBModel &model,
    std::unordered_map<std::pair<GRBVar *, GRBVar *>, GRBVar *> &H2toH,
    std::list<GRBVar> &hOrderVars,
    std::list<std::pair<GRBQuadExpr, GRBQuadExpr>> &hOrderQs) {
  std::vector<GRBVar *> stacked;
  int i = 0;
  for (; i + 1 < toStack.size(); i += 2) {
    std::pair<GRBVar *, GRBVar *> key(toStack[i], toStack[i + 1]);
    auto it = H2toH.find(key);
    if (it == H2toH.end()) {
      hOrderVars.emplace_back(model.addVar(0.0, 1.0, 0.0, GRB_BINARY));
      GRBVar *newStack = &hOrderVars.back();
      hOrderQs.emplace_back((*key.first) * (*key.second), *newStack);
      H2toH.emplace(key, newStack);
      stacked.push_back(newStack);
    } else {
      stacked.push_back(it->second);
    }
  }
  for (; i < toStack.size(); ++i) {
    stacked.push_back(toStack[i]);
  }

  return stacked;
}

static void
stackTerms(const std::vector<int> &toStack, GRBVar *varList, GRBModel &model,
           std::unordered_map<std::pair<GRBVar *, GRBVar *>, GRBVar *> &H2toH,
           std::list<GRBVar> &hOrderVars,
           std::list<std::pair<GRBQuadExpr, GRBQuadExpr>> &hOrderQs,
           std::vector<GRBVar *> &stacked) {

  for (auto &i : toStack)
    stacked.push_back(varList + i);

  while (stacked.size() > 2) {
    stacked = condenseStack(stacked, model, H2toH, hOrderVars, hOrderQs);
  }
}

void place::MIPSolver(const Eigen::MatrixXE &adjacencyMatrix,
                      const std::vector<place::R2Node> &nodes,
                      std::vector<place::SelectedNode> &bestNodes) {
  multi::Labeler::map tmp;
  MIPSolver(adjacencyMatrix, tmp, nodes, bestNodes);
}

static void populateModel(const Eigen::MatrixXE &adjacencyMatrix,
                          const multi::Labeler::map &highOrder,
                          const std::vector<place::R2Node> &nodes,
                          const std::vector<int> &numberOfLabels,
                          GRBVar *&varList, GRBModel &model) {

  const int numVars = numberOfLabels.size();
  const int numOpts = nodes.size();

  // Create variables that represent the labels
  double *upperBound = new double[numOpts];
  char *type = new char[numOpts];
  for (int i = 0; i < numOpts; ++i) {
    upperBound[i] = 1.0;
    type[i] = GRB_BINARY;
  }

  varList = model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
  GRBVar *inverseVarList =
      model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
  delete[] upperBound;
  delete[] type;
  // Integrate new variables
  model.update();
  // Add the inverse variables for the highorder terms
  for (int i = 0; i < numOpts; ++i) {
    model.addConstr(varList[i] + inverseVarList[i], GRB_EQUAL, 1.0);
  }

  // Set up the contraints to make this MRF
  for (int i = 0, offset = 0; i < numVars; ++i) {
    GRBLinExpr constr = 0.0;
    double *coeff = new double[numberOfLabels[i]];
    for (int a = 0; a < numberOfLabels[i]; ++a)
      coeff[a] = 1.0;

    constr.addTerms(coeff, varList + offset, numberOfLabels[i]);
    model.addConstr(constr, GRB_EQUAL, 1.0);
    offset += numberOfLabels[i];
    delete[] coeff;
  }

  // Add in pairwise and urnary weights to the model
  GRBQuadExpr objective = 0.0;
  for (int a = 0; a < numOpts; ++a) {
    // Pairwise
    for (int b = a + 1; b < numOpts; ++b) {
      const int i = nodes[a].pos;
      const int j = nodes[b].pos;
      const double weight = adjacencyMatrix(j, i).getWeight();
      if (weight == 0.0)
        continue;
      objective += weight * varList[a] * varList[b];
    }
    // Urnary
    objective += 0.5 * varList[a] * nodes[a].w;
  }

  std::list<std::pair<GRBQuadExpr, GRBQuadExpr>> hOrderQs;
  std::list<GRBVar> hOrderVars;
  std::unordered_map<std::pair<GRBVar *, GRBVar *>, GRBVar *> H2toH;
  for (auto &pair : highOrder) {
    auto &incident = pair.first;
    const double weight = 0.5 * pair.second.w;
    std::vector<GRBVar *> final;
    stackTerms(incident, inverseVarList, model, H2toH, hOrderVars, hOrderQs,
               final);
    if (final.size() == 1)
      objective -= (* final[0]) * weight;
    else
      objective -= (* final[0]) * (* final[1]) * weight;
  }
  model.update();
  for (auto &q : hOrderQs)
    model.addQConstr(q.first, GRB_EQUAL, q.second);

  model.setObjective(objective, GRB_MAXIMIZE);
}

void place::MIPSolver(const Eigen::MatrixXE &adjacencyMatrix,
                      const multi::Labeler::map &highOrder,
                      const std::vector<place::R2Node> &nodes,
                      std::vector<place::SelectedNode> &bestNodes) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node *prevNode = &nodes[0];
    for (auto &n : nodes) {
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
  std::cout << "Number of labels: ";
  for (auto &i : numberOfLabels)
    std::cout << i << "_";
  std::cout << std::endl;

  try {
    GRBEnv env;
    env.set("TimeLimit", "600");
    GRBModel model(env);

    GRBVar *varList;
    populateModel(adjacencyMatrix, highOrder, nodes, numberOfLabels, varList,
                  model);
    model.optimize();

    const int numVars = numberOfLabels.size();
    const int numOpts = nodes.size();
    std::cout << "Labels: ";
    for (int i = 0, offset = 0, k = 0; i < numOpts; ++i) {
      if (varList[i].get(GRB_DoubleAttr_X) == 1.0) {
        bestNodes.emplace_back(nodes[i], 1.0, i - offset, nodes[i].locked);
        std::cout << i - offset << "_";
      }
      if (numberOfLabels[k] == i + 1 - offset)
        offset += numberOfLabels[k++];
    }
    std::cout << std::endl;
    assert(bestNodes.size() == numVars);
  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
  }
}
