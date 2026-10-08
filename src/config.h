#ifndef CONFIG_H
#define CONFIG_H

#include <string>

enum class Mode { Process, MLCut, Scan, Plot };

struct Config {
    Mode mode = Mode::Process;

    // process
    bool lowPT = false;
    bool writeRoot = false;
    int nEvents = -1;
    std::string inputFile;       // defaults depend on lowPT
    std::string pixelMapDir;     // defaults depend on lowPT
    std::string nnModelDir;      // holds model.onnx, mean.npy, std.npy
    bool useNN = true;           // -X turns the NN off, leaving nn_score at -1
    bool applyCuts = false;      // -c applies the thresholds in pt2_cuts.h
    int keepPerPls = 0;          // -a N keeps only the N best pT2s per (pLS, zone)

    // mlcut (also uses inputFile, nnModelDir, nEvents); process uses nnCut and split too
    double nnCut = -1;           // keep pT2s with NN score >= nnCut (-1: no cut)
    std::string split = "all";   // events to use: all, or train,val,test (same split as train_v7.py)

    // scan
    double targetPercent = 90;

    // all modes
    std::string outputDir = "output";
    std::string histFile;        // defaults to <outputDir>/pt2_hists.root
};

// Parse "pt2 <process|mlcut|scan|plot> [options]". Returns false (after printing usage) on bad input.
bool parseArgs(int argc, char** argv, Config& cfg);

void printConfig(const Config& cfg);

#endif
