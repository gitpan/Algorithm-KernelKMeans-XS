#ifndef WKKM
#define WKKM

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace WKKM {

using std::vector;

typedef vector<vector<double> > KernelMatrix;
typedef vector<double> Vertex;
typedef vector<Vertex> Cluster;
typedef vector<size_t> Indices;

class PolynominalKernel {
private:
  double l;
  double p;
public:
  PolynominalKernel(): l(1), p(2) {}
  PolynominalKernel(double l, double p) : l(l), p(p) {}
  PolynominalKernel(const PolynominalKernel &kernel)
    : l(kernel.l), p(kernel.p) {}
  ~PolynominalKernel() {}
  PolynominalKernel &operator=(const PolynominalKernel &kernel) {
    this->l = kernel.l;
    this->p = kernel.p;
    return *this;
  }
  double operator()(const vector<double> &x1, const vector<double> &x2) const {
    double product = std::inner_product(x1.begin(), x1.end(), x2.begin(), 0);
    return pow((this->l + product), this->p);
  }
};

class NumberOfClusterError : public std::runtime_error {
public:
  NumberOfClusterError(const std::string &what) : std::runtime_error(what) {}
};

class Clusterer {
private:
  vector<Vertex> vertices;
  vector<double> weights;
  KernelMatrix kernel_matrix;

  Clusterer(const Clusterer &);
  Clusterer &operator=(const Clusterer &);

  // generates lower triangle matrix
  template <typename T>
  KernelMatrix compute_kernel_matrix(T kernel) const {
    KernelMatrix matrix(this->vertices.size());
    for (size_t i = 0; i < this->vertices.size(); i++) {
      matrix[i].resize(i + 1);
      for (size_t j = 0; j <= i; j++) {
        matrix[i][j] = kernel(this->vertices[i], this->vertices[j]);
      }
    }
    return matrix;
  }

  // vertex index -> cluster index -> norm
  vector<vector<double> > compute_norms(const vector<Indices> &clusters) const {
    vector<double> term3s(clusters.size());
    vector<double> total_weights(clusters.size());

    for (size_t i = 0; i < clusters.size(); i++) {
      const Indices &cluster = clusters[i];
      double term3 = 0, total_weight = 0;

      for (size_t j = 0; j < cluster.size(); j++) {
        size_t idx_v = cluster[j];
        for (size_t k = 0; k < cluster.size(); k++) {
          size_t idx_u = cluster[k];
          double kernel = (idx_v < idx_u)
            ? this->kernel_matrix[idx_u][idx_v]
            : this->kernel_matrix[idx_v][idx_u];
          term3 += this->weights[idx_v] * this->weights[idx_u] * kernel;
        }
        total_weight += this->weights[idx_v];
      }
      term3s[i] = term3 / (total_weight * total_weight);
      total_weights[i] = total_weight;
    }

    vector<vector<double> > norms(this->vertices.size());
    for (size_t i = 0; i < this->vertices.size(); i++) {
      double term1 = this->kernel_matrix[i][i];
      norms[i].resize(clusters.size());
      for (size_t j = 0; j < clusters.size(); j++) {
        double term2 = 0;
        const Indices &cluster = clusters[j];
        for (size_t k = 0; k < cluster.size(); k++) {
          size_t idx = cluster[k];
          double kernel = (i < idx)
            ? this->kernel_matrix[idx][i] : this->kernel_matrix[i][idx];
          term2 += this->weights[idx] * kernel;
        }
        term2 = -2 * term2 / total_weights[j];
        norms[i][j] = term1 + term2 + term3s[j];
      }
    }

    return norms;
  }

  double compute_score(const vector<Indices> &clusters,
                       const vector<vector<double> > &norms) const {
    double score = 0;
    for (size_t i = 0; i < clusters.size(); i++) {
      const Indices &cluster = clusters[i];
      for (size_t j = 0; j < cluster.size(); j++) {
        score += this->weights[cluster[j]] * norms[cluster[j]][i];
      }
    }
    return score;
  }

  vector<Indices> init_clusters(size_t k, bool shuffle) const {
    vector<size_t> indices(this->vertices.size());
    for (size_t i = 0; i < indices.size(); i++) { indices[i] = i; }
    if (shuffle) { std::random_shuffle(indices.begin(), indices.end()); }
    size_t cluster_size = static_cast<size_t>(
      floor(static_cast<double>(this->vertices.size()) / k)
    );
    vector<Indices> clusters(k);
    for (size_t i = 0; i < k; i++) {
      Indices &cluster = clusters[i];
      cluster.resize(
        (i == k - 1) ? this->vertices.size() - cluster_size * i : cluster_size
      );
      for (size_t j = 0; j < cluster.size(); j++) {
        cluster[j] = indices[i * cluster_size + j];
      }
    }
    return clusters;
  }

  vector<Indices> step(const vector<Indices> &clusters,
                       const vector<vector<double> > &norms) const {
    vector<Indices> new_clusters(clusters.size());
    for (size_t i = 0; i < this->vertices.size(); i++) {
      size_t nearest_cluster = 0;
      double min_norm = norms[i][0];
      for (size_t j = 0; j < clusters.size(); j++) {
        if (norms[i][j] < min_norm) {
          min_norm = norms[i][j];
          nearest_cluster = j;
        }
      }
      new_clusters[nearest_cluster].push_back(i);
    }

    for (vector<Indices>::iterator cluster = new_clusters.begin();
         cluster != new_clusters.end();
         cluster++) {
      if (cluster->size() == 0) { cluster = new_clusters.erase(cluster); }
    }
    return new_clusters;
  }

public:
  template <typename T>
  Clusterer(const vector<Vertex> &vertices,
            const vector<double> &weights,
            T kernel) throw (std::invalid_argument)
    : vertices(vertices),
      weights(weights),
      kernel_matrix(this->compute_kernel_matrix(kernel)) {
    if (this->vertices.size() != this->weights.size()) {
      throw std::invalid_argument("Vertices and weights are must be same size");
    }
  }
  Clusterer(const vector<Vertex> &vertices,
            const vector<double> &weights,
            const KernelMatrix &kernel_matrix) throw (std::invalid_argument)
    : vertices(vertices),
      weights(weights),
      kernel_matrix(kernel_matrix) {
    if (this->vertices.size() != this->weights.size()) {
      throw std::invalid_argument("Vertices and weights are must be same size");
    }
    // It's assumed that kernel_matrix is lower triangle
    if ((this->kernel_matrix.size() < this->vertices.size())
        || (this->kernel_matrix[this->kernel_matrix.size() - 1].size()
            < this->vertices.size())) {
      throw std::invalid_argument("Kernel matrix is too small");
    }
  }
  ~Clusterer() {}

  template <typename T>
  vector<Cluster> run(size_t k, T converged, bool shuffle = true) const {
    return this->run(k, k, converged, shuffle);
  }

  template <typename T>
  vector<Cluster> run(size_t k, size_t k_min,
                      T converged, bool shuffle = true) const
    throw (NumberOfClusterError) {
    vector<Indices> clusters = this->init_clusters(k, shuffle);
    vector<vector<double> > norms = this->compute_norms(clusters);
    double score;
    double new_score = this->compute_score(clusters, norms);

    do {
      clusters = this->step(clusters, norms);
      if (clusters.size() < k_min) {
        std::ostringstream errmsg;
        errmsg << "Number of clusters became less than k_min="
               << k_min
               << std::endl;
        throw NumberOfClusterError(errmsg.str());
      }
      norms = this->compute_norms(clusters);
      score = new_score;
      new_score = this->compute_score(clusters, norms);
    } while (!converged(score, new_score));

    vector<Cluster> result(clusters.size());
    for (size_t i = 0; i < clusters.size(); i++) {
      Indices &cluster = clusters[i];
      result[i].resize(cluster.size());
      for (size_t j = 0; j < cluster.size(); j++) {
        result[i][j] = this->vertices[cluster[j]];
      }
    }

    return result;
  }
};

} // namespace

#endif
