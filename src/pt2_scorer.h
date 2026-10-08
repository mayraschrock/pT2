#ifndef PT2_SCORER_H
#define PT2_SCORER_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "pt2.h"
#include "rootReader.h"

// ONNX NN that scores a pT2 candidate as real (-> 1) or fake (-> 0)
class Pt2Scorer
{
public:
    static constexpr size_t kNBaseFeatures = 21;
    static constexpr size_t kNLstFeatures = 8;    // LST pT3-style variables + z residual validity flag, train_v7.py --lst_vars
    static constexpr size_t kNLayerFeatures = 13; // one-hot LS layer connection (= kNCat), train_v7.py --layers
    static constexpr size_t kNFeatures = kNBaseFeatures + kNLstFeatures + kNLayerFeatures;
    // All inputs: base, then LST, then layer flags; a model takes the base inputs plus either or both of the other groups
    using FeatureVector = std::array<float, kNFeatures>;

    // meanPath / stdPath: .npy normalization arrays saved by the training script
    Pt2Scorer(const std::string &modelPath, const std::string &meanPath, const std::string &stdPath);

    // Model inputs, in the order of FEATURE_NAMES in pt2_ml/train_*.py (with --lst_vars --layers)
    static FeatureVector features(const rootReader &reader, const pT2 &pt2);

    // Score many candidates in as few ONNX calls as possible; one score per row, same order
    std::vector<float> score(const std::vector<FeatureVector> &raw) const;
    float score(const FeatureVector &raw) const;
    float score(const rootReader &reader, const pT2 &pt2) const;

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memInfo_;

    std::vector<float> mean_, std_;
    size_t nFeatures_ = kNBaseFeatures; // inputs the model takes
    bool useLst_ = false, useLayers_ = false;
    std::vector<size_t> cols_; // FeatureVector index of each model input
    std::string inputName_, outputName_;
    bool dynamicBatch_ = false; // model accepts N rows per call (else 1 row per call)

    // Run the model on n already-normalized rows
    void run(float *normed, size_t n, float *out) const;
};

#endif
