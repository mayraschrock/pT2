#ifndef ML_CUT_H
#define ML_CUT_H

#include "config.h"

// `pt2 mlcut`: read the pT2s stored in an LSTNtuple_with_pT2.root (from `process -r`), score them with the
// NN in cfg.nnModelDir, keep those with score >= cfg.nnCut, and write cfg.histFile (passing pT2s only)
// plus the before/after cut study in <outputDir>/cut_study
int runMLCut(const Config& cfg);

#endif
