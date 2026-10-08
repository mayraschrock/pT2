#ifndef PROCESS_H
#define PROCESS_H

#include <vector>

#include "config.h"
#include "histograms.h"
#include "pt2.h"
#include "pt2_scorer.h"
#include "rootReader.h"

// Event loop: build pT2s, fill histograms into cfg.histFile, optionally write the pT2 ntuple
int runProcess(const Config& cfg);

// Per-pT2 steps of the event loop, shared with `pt2 mlcut` (ml_cut.cc).
// computeFeatures needs reader.ls_isUsed / pls_isUsed set for the event (buildUsedMask).
void computeFeatures(const rootReader& reader, pT2& pt2);
void scoreBatch(const Pt2Scorer& scorer, const rootReader& reader, const std::vector<pT2*>& batch);
void fillHistograms(const HistogramManager& hists, const pT2& pt2);

#endif
