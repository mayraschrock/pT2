#include "pt2_scorer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "histograms.h"

static_assert(Pt2Scorer::kNLayerFeatures == kNCat, "one layer flag per layer-connection category");

namespace
{

// Minimal .npy reader: 1D float32 / float64 arrays only
std::vector<float> loadNpy(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    char magic[6];
    f.read(magic, 6);
    if (!f || std::strncmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a .npy file: " + path);

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char *>(&major), 1);
    f.read(reinterpret_cast<char *>(&minor), 1);

    uint32_t hlen = 0;
    if (major == 1)
    {
        uint16_t h16 = 0;
        f.read(reinterpret_cast<char *>(&h16), 2);
        hlen = h16;
    }
    else
    {
        f.read(reinterpret_cast<char *>(&hlen), 4);
    }

    std::string header(hlen, ' ');
    f.read(&header[0], hlen);

    bool isDouble = header.find("<f8") != std::string::npos || header.find("float64") != std::string::npos;

    size_t lp = header.find('(', header.find("shape"));
    size_t rp = header.find(')', lp);
    size_t n = std::stoul(header.substr(lp + 1, rp - lp - 1));

    std::vector<float> out(n);
    if (isDouble)
    {
        std::vector<double> tmp(n);
        f.read(reinterpret_cast<char *>(tmp.data()), n * sizeof(double));
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(tmp[i]);
    }
    else
    {
        f.read(reinterpret_cast<char *>(out.data()), n * sizeof(float));
    }
    if (!f) throw std::runtime_error("truncated .npy file: " + path);

    return out;
}

} // namespace

Pt2Scorer::Pt2Scorer(const std::string &modelPath, const std::string &meanPath, const std::string &stdPath)
    : env_(ORT_LOGGING_LEVEL_WARNING, "pt2scorer"),
      memInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);

    mean_ = loadNpy(meanPath);
    std_ = loadNpy(stdPath);
    nFeatures_ = mean_.size();
    if (std_.size() != nFeatures_) throw std::runtime_error("NN mean and std sizes differ");
    // Which input groups the model was trained with, from its number of inputs
    if (nFeatures_ == kNBaseFeatures) {}
    else if (nFeatures_ == kNBaseFeatures + kNLayerFeatures) useLayers_ = true;
    else if (nFeatures_ == kNBaseFeatures + kNLstFeatures) useLst_ = true;
    else if (nFeatures_ == kNFeatures) useLst_ = useLayers_ = true;
    else throw std::runtime_error("NN has " + std::to_string(nFeatures_) + " inputs, which matches no combination of --layers / --lst_vars");
    for (size_t i = 0; i < kNBaseFeatures; ++i) cols_.push_back(i);
    if (useLst_)
        for (size_t i = 0; i < kNLstFeatures; ++i) cols_.push_back(kNBaseFeatures + i);
    if (useLayers_)
        for (size_t i = 0; i < kNLayerFeatures; ++i) cols_.push_back(kNBaseFeatures + kNLstFeatures + i);

    Ort::AllocatorWithDefaultOptions allocator;
    inputName_ = session_->GetInputNameAllocated(0, allocator).get();
    outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

    // Batch dimension is -1 if the model was exported with a dynamic batch size
    std::vector<int64_t> shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape[1] != static_cast<int64_t>(nFeatures_))
        throw std::runtime_error("NN input shape does not match its mean/std size");
    dynamicBatch_ = shape[0] < 0;

    std::cout << "ONNX model loaded: " << modelPath << " (input: " << inputName_ << ", output: " << outputName_
              << ", batch: " << (dynamicBatch_ ? "dynamic" : "fixed at 1, scoring one row per call")
              << ", " << nFeatures_ << " inputs" << (useLst_ ? " with LST vars" : "") << (useLayers_ ? " with layers" : "") << ")\n";
}

Pt2Scorer::FeatureVector Pt2Scorer::features(const rootReader &reader, const pT2 &pt2)
{
    size_t p = pt2.pls_idx, l = pt2.ls_idx;
    FeatureVector f = {
        reader.ls_pt->at(l),
        reader.ls_eta->at(l),
        std::sin(reader.ls_phi->at(l)),
        std::cos(reader.ls_phi->at(l)),
        reader.pls_pt->at(p),
        reader.pls_eta->at(p),
        std::sin(reader.pls_phi->at(p)),
        std::cos(reader.pls_phi->at(p)),
        static_cast<float>(reader.pls_charge->at(p)),
        static_cast<float>(reader.pls_nhit->at(p)),
        pt2.delta_pt,
        pt2.delta_eta,
        pt2.delta_phi,
        pt2.delta_r,
        static_cast<float>(pt2.heli[0]),
        static_cast<float>(pt2.heli[1]),
        static_cast<float>(pt2.heli[2]),
        static_cast<float>(pt2.heli[3]),
        static_cast<float>(pt2.rz_simple.first),
        static_cast<float>(pt2.rz_simple.second),
        std::log(std::abs(static_cast<float>(pt2.heli[0])) + 1e-6f),
    }; // LST inputs and layer flags: set below

    // LST inputs, as lst_columns() in pt2_ml/nn_common.py: the z residuals are 0 when either one
    // is invalid (sentinel <= -99), with a flag saying whether they are valid
    constexpr float kLstSentinel = -99.0f;
    float zGeo = pt2.z_res_geo, zKin = pt2.z_res_kin;
    bool zValid = zGeo > kLstSentinel && zKin > kLstSentinel;
    float *lst = f.data() + kNBaseFeatures;
    lst[0] = pt2.lst_delta_phi;
    lst[1] = pt2.beta_in;
    lst[2] = pt2.beta_out;
    lst[3] = pt2.delta_beta;
    lst[4] = zValid ? zGeo : 0.0f;
    lst[5] = zValid ? zKin : 0.0f;
    lst[6] = pt2.delta_angle;
    lst[7] = zValid ? 1.0f : 0.0f;

    // Layer flags: 0 except the pT2's layer connection
    if (pt2.combo_idx >= 0 && pt2.combo_idx < static_cast<int>(kNLayerFeatures))
        f[kNBaseFeatures + kNLstFeatures + pt2.combo_idx] = 1.0f;
    return f;
}

void Pt2Scorer::run(float *normed, size_t n, float *out) const
{
    const int64_t shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(nFeatures_)};
    Ort::Value input = Ort::Value::CreateTensor<float>(memInfo_, normed, n * nFeatures_, shape, 2);

    const char *inputNames[] = {inputName_.c_str()};
    const char *outputNames[] = {outputName_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 1);
    if (outputs.empty() || !outputs[0].IsTensor() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != n)
        throw std::runtime_error("invalid ONNX output");

    // Model outputs logits
    const float *logits = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-logits[i]));
}

std::vector<float> Pt2Scorer::score(const std::vector<FeatureVector> &raw) const
{
    size_t n = raw.size();

    // Normalize; non-finite inputs are treated as 0
    std::vector<float> normed(n * nFeatures_);
    for (size_t r = 0; r < n; ++r)
    {
        for (size_t i = 0; i < nFeatures_; ++i)
        {
            float raw_i = raw[r][cols_[i]];
            float val = std::isfinite(raw_i) ? raw_i : 0.0f;
            normed[r * nFeatures_ + i] = (val - mean_[i]) / std_[i];
        }
    }

    std::vector<float> scores(n);
    if (n == 0) return scores;
    if (dynamicBatch_)
        run(normed.data(), n, scores.data());
    else
        for (size_t r = 0; r < n; ++r) run(normed.data() + r * nFeatures_, 1, scores.data() + r);
    return scores;
}

float Pt2Scorer::score(const FeatureVector &raw) const
{
    return score(std::vector<FeatureVector>{raw})[0];
}

float Pt2Scorer::score(const rootReader &reader, const pT2 &pt2) const
{
    return score(features(reader, pt2));
}
