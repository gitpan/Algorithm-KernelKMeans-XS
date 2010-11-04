#include "wkkm.hxx"

#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "ppport.h"
#ifdef __cplusplus
}
#endif

#define XS_STATE(type, x) \
    INT2PTR(type, SvROK(x) ? SvIV(SvRV(x)) : SvIV(x))

#define XS_STRUCT2OBJ(sv, klass, obj) \
    if (obj == NULL) { \
        sv_setsv(sv, &PL_sv_undef); \
    } else { \
        sv_setref_pv(sv, klass, (void *) obj); \
    }

inline bool is_arrayref(const SV *ref) {
  return SvROK(ref) && SvTYPE(SvRV(ref)) == SVt_PVAV;
}

inline bool is_hashref(const SV *ref) {
  return SvROK(ref) && SvTYPE(SvRV(ref)) == SVt_PVHV;
}

inline bool is_coderef(const SV *ref) {
  return SvROK(ref) && SvTYPE(SvRV(ref)) == SVt_PVCV;
}

inline bool is_vector_array(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  for (size_t i = 0; i < array_len; i++) {
    if (!looks_like_number(*av_fetch(ary, i, 0))) { return false; }
  }
  return true;
}

inline bool is_matrix_array(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  for (size_t i = 0; i < array_len; i++) {
    SV *ref = *av_fetch(ary, i, 0);
    if (!(is_arrayref(ref) && is_vector_array((AV *)SvRV(ref)))) {
      return false;
    }
  }
  return true;
}

inline AV *vector2array(const std::vector<double> &v) {
  AV *ary = newAV();
  av_extend(ary, v.size() - 1);
  for (size_t i = 0; i < v.size(); i++) { av_store(ary, i, newSVnv(v[i])); }
  return ary;
}

inline void array2vector(AV *ary, std::vector<double> &v) {
  size_t array_len = av_len(ary) + 1;
  v.resize(array_len);
  for (size_t i = 0; i < array_len; i++) {
    v[i] = SvNV(*av_fetch(ary, i, 0));
  }
}

inline AV *matrix2array(const std::vector<std::vector<double> > &matrix) {
  AV *mat = newAV();
  av_extend(mat, matrix.size() - 1);
  for (size_t i = 0; i < matrix.size(); i++) {
    SV *vec = newRV_noinc((SV *)vector2array(matrix[i]));
    av_store(mat, i, vec);
  }
  return mat;
}

inline void array2matrix(AV *ary, std::vector<std::vector<double> > &matrix) {
  size_t array_len = av_len(ary) + 1;
  matrix.resize(array_len);
  for (size_t i = 0; i < array_len; i++) {
    AV *row_ary = (AV *)SvRV(*av_fetch(ary, i, 0));
    array2vector(row_ary, matrix[i]);
  }
}

inline AV *clusters2array(const std::vector<WKKM::Cluster> &clusters) {
  AV *ary = newAV();
  av_extend(ary, clusters.size() - 1);
  for (size_t i = 0; i < clusters.size(); i++) {
    SV *mat = newRV_noinc((SV *)matrix2array(clusters[i]));
    av_store(ary, i, mat);
  }
  return ary;
}

class KernelSubWrapper {
private:
  SV *coderef;
  KernelSubWrapper &operator=(const KernelSubWrapper &);
public:
  KernelSubWrapper(SV *coderef) throw (std::invalid_argument)
    : coderef(coderef) {
    if (!is_coderef(this->coderef)) {
      throw std::invalid_argument("Not a code reference");
    }
    SvREFCNT_inc(this->coderef);
  }
  KernelSubWrapper(const KernelSubWrapper &wrapper) : coderef(wrapper.coderef) {
    SvREFCNT_inc(this->coderef);
  }
  ~KernelSubWrapper() {
    SvREFCNT_dec(this->coderef);
  }
  double operator()(const WKKM::Vertex &v, const WKKM::Vertex &u) const {
    dSP;
    ENTER;
    SAVETMPS;

    PUSHMARK(SP);
    XPUSHs(sv_2mortal(newRV_noinc((SV *)vector2array(v))));
    XPUSHs(sv_2mortal(newRV_noinc((SV *)vector2array(u))));
    PUTBACK;
    call_sv(this->coderef, G_SCALAR);
    SPAGAIN;
    double kernel = POPn;
    PUTBACK;

    FREETMPS;
    LEAVE;

    return kernel;
  }
};

class PredictSubWrapper {
private:
  SV *coderef;
  PredictSubWrapper &operator=(const PredictSubWrapper &);
public:
  PredictSubWrapper(SV *coderef) throw (std::invalid_argument)
    : coderef(coderef) {
    if (!is_coderef(this->coderef)) {
      throw std::invalid_argument("Not a code reference");
    }
    SvREFCNT_inc(this->coderef);
  }
  PredictSubWrapper(const PredictSubWrapper &wrapper)
    : coderef(wrapper.coderef) {
    SvREFCNT_inc(this->coderef);
  }
  ~PredictSubWrapper() {
    SvREFCNT_dec(this->coderef);
  }
  bool operator()(double score, double new_score) const {
    dSP;
    ENTER;
    SAVETMPS;

    PUSHMARK(SP);
    XPUSHs(sv_2mortal(newSVnv(score)));
    XPUSHs(sv_2mortal(newSVnv(new_score)));
    PUTBACK;
    call_sv(this->coderef, G_SCALAR);
    SPAGAIN;
    bool converged = SvTRUEx(POPs);
    PUTBACK;

    FREETMPS;
    LEAVE;

    return converged;
  }
};

bool converged_default(double score, double new_score) {
  return score == new_score;
}

/* This macro takes named parameter list and put it into one HV.
   This macro intent to be used in class/instance method.
   Since ST(0) points class name string or $self, we start taking from ST(1).
   If only 1 parameter is given, the parameter should be a hashref.
   Note that "items % 2 != 0" is true when number of parameters is even.
   Because items count ST(0) in. */
#define TAKE_NAMED_PARAMETERS(hash)                        \
  if (items == 2) {                                        \
    if (is_hashref(ST(1))) {                               \
      hash = (HV *)SvRV(ST(1));                            \
    } else {                                               \
      Perl_croak(aTHX_ "Named parameters are required");   \
    }                                                      \
  } else if (items % 2 != 0) {                             \
    hash = (HV *)sv_2mortal((SV *)newHV());                \
    for (size_t i = 1; i < items; i += 2) {                \
      if (!SvPOK(ST(i))) {                                 \
        Perl_croak(aTHX_ "Named parameters are required"); \
      }                                                    \
      size_t key_len;                                      \
      const char *key = SvPV(ST(i), key_len);              \
      SV *value = newSVsv(ST(i + 1));                      \
      hv_store(hash, key, key_len, value, 0);              \
    }                                                      \
  } else {                                                 \
    Perl_croak(aTHX_ "Named parameters are required");     \
  }

#define CROAK_IF_UNKNOWN_PARAMETER_WAS_REST(hash)     \
  {                                                   \
    if (HvUSEDKEYS(hash) != 0) {                      \
      hv_iterinit(hash);                              \
      HE *entry = hv_iternext(hash);                  \
      size_t key_len;                                 \
      const char *key = HePV(entry, key_len);         \
      Perl_croak(aTHX_ "Unknown parameter: %s", key); \
    }                                                 \
  }

MODULE = Algorithm::KernelKMeans::XS  PACKAGE = Algorithm::KernelKMeans::XS

PROTOTYPES: DISABLE

WKKM::Clusterer *
WKKM::Clusterer::new(...)
CODE:
    HV *args;
    TAKE_NAMED_PARAMETERS(args);

    if (!hv_exists(args, "vertices", 8)) {
      Perl_croak(aTHX_ "Missing required parameter");
    }
    SV *verts_ref = hv_delete(args, "vertices", 8, 0);
    if (!(is_arrayref(verts_ref) && is_matrix_array((AV *)SvRV(verts_ref)))) {
      Perl_croak(aTHX_ "Vertices must be an array of vectors");
    }
    std::vector<WKKM::Vertex> verts;
    array2matrix((AV *)SvRV(verts_ref), verts);

    std::vector<double> weights(verts.size(), 1);
    if (hv_exists(args, "weights", 7)) {
      SV *weights_ref = hv_delete(args, "weights", 7, 0);
      if (!(is_arrayref(weights_ref)
            && is_vector_array((AV *)SvRV(weights_ref)))) {
        Perl_croak(aTHX_ "Weights must be a real vector");
      }
      array2vector((AV *)SvRV(weights_ref), weights);
    }

    try {
      if (hv_exists(args, "kernel_matrix", 13)) {
        SV *kmat_ref = hv_delete(args, "kernel_matrix", 13, 0);
        if (!(is_arrayref(kmat_ref) && is_matrix_array((AV *)SvRV(kmat_ref)))) {
          Perl_croak(aTHX_ "Kernel matrix must be an array of vectors");
        }
        WKKM::KernelMatrix kmat;
        array2matrix((AV *)SvRV(kmat_ref), kmat);

        CROAK_IF_UNKNOWN_PARAMETER_WAS_REST(args);
        RETVAL = new WKKM::Clusterer(verts, weights, kmat);
      } else {
        if (hv_exists(args, "kernel", 6)) {
          SV *kernel = hv_delete(args, "kernel", 6, 0);
          KernelSubWrapper kernel_wrapped(kernel);
          CROAK_IF_UNKNOWN_PARAMETER_WAS_REST(args);
          RETVAL = new WKKM::Clusterer(verts, weights, kernel_wrapped);
        } else {
          CROAK_IF_UNKNOWN_PARAMETER_WAS_REST(args);
          RETVAL = new WKKM::Clusterer(verts, weights, PolynominalKernel(1, 2));
        }
      }
    } catch (const std::invalid_argument &e) {
      Perl_croak(aTHX_ e.what());
    } catch (...) { throw; }
OUTPUT:
    RETVAL

SV *
WKKM::Clusterer::run(...)
CODE:
  HV *args;
  TAKE_NAMED_PARAMETERS(args);

  if (!hv_exists(args, "k", 1)) {
    Perl_croak(aTHX_ "Missing required parameter");
  }

  SV *k_sv = hv_delete(args, "k", 1, 0);
  if (!(looks_like_number(k_sv) && SvUV(k_sv) > 0)) {
    Perl_croak(aTHX_ "Cluster size must be a positive integer (> 0)");
  }
  UV k = SvUV(k_sv);

  UV k_min = k;
  if (hv_exists(args, "k_min", 5)) {
    SV *k_min_sv = hv_delete(args, "k_min", 5, 0);
    if (!(looks_like_number(k_min_sv) && SvUV(k_min_sv) > 0)) {
      Perl_croak(aTHX_ "Cluster size must be a positive integer (> 0)");
    }
    k_min = SvUV(k_min_sv);
  }

  bool shuffle = true;
  if (hv_exists(args, "shuffle", 7)) {
    shuffle = SvTRUEx(hv_delete(args, "shuffle", 7, 0));
  }

  try {
    std::vector<Cluster> clusters;
    if (hv_exists(args, "converged", 9)) {
      SV *converged = hv_delete(args, "converged", 9, 0);
      CROAK_IF_UNKNOWN_PARAMETER_WAS_REST(args);
      clusters = THIS->run(k, k_min, PredictSubWrapper(converged), shuffle);
    } else {
      CROAK_IF_UNKNOWN_PARAMETER_WAS_REST(args);
      clusters = THIS->run(k, k_min, converged_default, shuffle);
    }
    AV *clusters_av = clusters2array(clusters);
    RETVAL = newRV_noinc((SV *)clusters_av);
  } catch (const WKKM::NumberOfClusterError &e) {
    Perl_croak(aTHX_ e.what());
  } catch (...) { throw; }
OUTPUT:
  RETVAL

void
WKKM::Clusterer::DESTROY()
