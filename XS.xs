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

inline bool is_vector(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  for (size_t i = 0; i < array_len; i++) {
    if (!looks_like_number(*av_fetch(ary, i, FALSE))) { return false; }
  }
  return true;
}

inline bool is_matrix(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  for (size_t i = 0; i < array_len; i++) {
    SV *ref = *av_fetch(ary, i, FALSE);
    if (!(is_arrayref(ref) && is_vector((AV *)SvRV(ref)))) { return false; }
  }
  return true;
}

inline bool is_vertices(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  for (size_t i = 0; i < array_len; i++) {
    if (!is_hashref(*av_fetch(ary, i, FALSE))) { return false; }
  }
  return true;
}

inline AV *vector2array(const std::vector<double> &v) {
  AV *ary = newAV();
  av_extend(ary, v.size() - 1);
  for (size_t i = 0; i < v.size(); i++) { av_store(ary, i, newSVnv(v[i])); }
  return ary;
}

inline std::vector<double> array2vector(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  std::vector<double> v(array_len);
  for (size_t i = 0; i < array_len; i++) {
    v[i] = SvNV(*av_fetch(ary, i, FALSE));
  }
  return v;
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

inline std::vector<std::vector<double> > array2matrix(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  std::vector<std::vector<double> > matrix(array_len);
  for (size_t i = 0; i < array_len; i++) {
    AV *row_ary = (AV *)SvRV(*av_fetch(ary, i, FALSE));
    matrix[i] = array2vector(row_ary);
  }
  return matrix;
}

inline HV *vertex2hash(const WKKM::Vertex &v) {
  WKKM::Vertex &v_ = const_cast<WKKM::Vertex &>(v);
  HV *hash = newHV();
  for (WKKM::Vertex::iterator iter = v_.begin(); iter != v_.end(); iter++) {
    const std::string &key = iter->first;
    hv_store(hash, key.c_str(), key.size(), newSVnv(iter->second), 0);
  }
  return hash;
}

inline WKKM::Vertex hash2vertex(HV *hash) {
  WKKM::Vertex vert;
  hv_iterinit(hash);
  char *key;
  I32 key_len;
  SV *val;
  while ((val = hv_iternextsv(hash, &key, &key_len)) != 0) {
    vert[std::string(key, key_len)] = SvNV(val);
  }
  return vert;
}

inline AV *vertices2array(const std::vector<WKKM::Vertex> &verts) {
  AV *ary = newAV();
  av_extend(ary, verts.size() - 1);
  for (size_t i = 0; i < verts.size(); i++) {
    av_store(ary, i, newRV_noinc((SV *)vertex2hash(verts[i])));
  }
  return ary;
}

inline std::vector<WKKM::Vertex> array2vertices(AV *ary) {
  size_t array_len = av_len(ary) + 1;
  std::vector<WKKM::Vertex> verts(array_len);
  for (size_t i = 0; i < array_len; i++) {
    HV *vert_hv = (HV *)SvRV(*av_fetch(ary, i, FALSE));
    verts[i] = hash2vertex(vert_hv);
  }
  return verts;
}

inline AV *clusters2array(const std::vector<WKKM::Cluster> &clusters) {
  AV *ary = newAV();
  av_extend(ary, clusters.size() - 1);
  for (size_t i = 0; i < clusters.size(); i++) {
    av_store(ary, i, newRV_noinc((SV *)vertices2array(clusters[i])));
  }
  return ary;
}

class KernelSubWrapper {
private:
  SV *coderef;
  KernelSubWrapper();
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
    XPUSHs(sv_2mortal(newRV_noinc((SV *)vertex2hash(v))));
    XPUSHs(sv_2mortal(newRV_noinc((SV *)vertex2hash(u))));
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
  PredictSubWrapper();
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

#define CROAK_UNKNOWN_PARAMETER(hash)                 \
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
  if (!(is_arrayref(verts_ref) && is_vertices((AV *)SvRV(verts_ref)))) {
    Perl_croak(aTHX_ "Vertices must be an array of vectors");
  }
  std::vector<WKKM::Vertex> verts = array2vertices((AV *)SvRV(verts_ref));

  std::vector<double> weights(verts.size(), 1);
  if (hv_exists(args, "weights", 7)) {
    SV *weights_ref = hv_delete(args, "weights", 7, 0);
    if (!(is_arrayref(weights_ref) && is_vector((AV *)SvRV(weights_ref)))) {
      Perl_croak(aTHX_ "Weights must be a real vector");
    }
    weights = array2vector((AV *)SvRV(weights_ref));
  }

  try {
    if (hv_exists(args, "kernel_matrix", 13)) {
      SV *kmat_ref = hv_delete(args, "kernel_matrix", 13, 0);
      if (!(is_arrayref(kmat_ref) && is_matrix((AV *)SvRV(kmat_ref)))) {
        Perl_croak(aTHX_ "Kernel matrix must be an array of vectors");
      }
      WKKM::KernelMatrix kmat = array2matrix((AV *)SvRV(kmat_ref));

      CROAK_UNKNOWN_PARAMETER(args);
      RETVAL = new WKKM::Clusterer(verts, weights, kmat);
    } else {
      if (hv_exists(args, "kernel", 6)) {
        KernelSubWrapper kernel(hv_delete(args, "kernel", 6, 0));
        CROAK_UNKNOWN_PARAMETER(args);
        RETVAL = new WKKM::Clusterer(verts, weights, kernel);
      } else {
        CROAK_UNKNOWN_PARAMETER(args);
        WKKM::PolynominalKernel kernel(1.0, 2.0);
        RETVAL = new WKKM::Clusterer(verts, weights, kernel);
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

  bool shuffle = hv_exists(args, "shuffle", 7)
    ? SvTRUEx(hv_delete(args, "shuffle", 7, 0)) : true;

  try {
    std::vector<WKKM::Cluster> clusters;
    if (hv_exists(args, "converged", 9)) {
      PredictSubWrapper converged(hv_delete(args, "converged", 9, 0));
      CROAK_UNKNOWN_PARAMETER(args);
      clusters = THIS->run(k, k_min, converged, shuffle);
    } else {
      CROAK_UNKNOWN_PARAMETER(args);
      clusters = THIS->run(k, k_min, converged_default, shuffle);
    }
    AV *clusters_av = clusters2array(clusters);
    RETVAL = newRV_noinc((SV *)clusters_av);
  } catch (const std::runtime_error &e) {
    Perl_croak(aTHX_ e.what());
  } catch (...) { throw; }
OUTPUT:
  RETVAL

void
WKKM::Clusterer::DESTROY()
