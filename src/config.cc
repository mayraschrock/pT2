#include "config.h"

#include <getopt.h>
#include <iostream>

namespace {

// Alternative inputs, kept for reference:
//   /blue/avery/aaponteutani/CMSSW_16_1_0/src/RecoTracker/LSTCore/standalone/LSTNtuple_pionGun_0p3.root  (0.5 GeV)
//   /cmsuf/data/store/user/t2/users/matthew.dittrich/PT2_DATA/ROOT_FILES/LSTNtuple_LowPT.root            (0.6 GeV)
//   /blue/avery/aaponteutani/CMSSW_16_1_0/src/RecoTracker/LSTCore/standalone/LSTNtuple_PU200_0p5.root     (0.5 GeV)
const char* kInputLowPT = "/blue/avery/aaponteutani/CMSSW_16_1_0/src/RecoTracker/LSTCore/standalone/LSTNtuple_PU200_0p6.root";
const char* kInput      = "/cmsuf/data/store/user/t2/users/matthew.dittrich/PT2_DATA/ROOT_FILES/LSTNtuple.root";

// Alternative pixel maps, kept for reference:
//   /cmsuf/data/store/user/t2/users/matthew.dittrich/PT2_DATA/PIXEL_MAPS/Pixel_Maps_0p6GeV/  (0.6 GeV)
//   /blue/p.chang/aaponteutani/LSTGeometry/output_0p4/pixelmap/  (0.4 GeV)
//   /blue/p.chang/aaponteutani/LSTGeometry/output_0p3/pixelmap/  (0.5 GeV)
const char* kPixelMapLowPT = "/blue/p.chang/aaponteutani/LSTGeometry/output/pixelmap/";
const char* kPixelMap      = "/cmsuf/data/store/user/t2/users/matthew.dittrich/PT2_DATA/PIXEL_MAPS/Pixel_Maps_0p8GeV/";

const char* kNNModelDir = "/blue/p.chang/mayrasilva/lst_pt2/pT2/NN_model";

void printUsage(const char* prog)
{
    std::cerr << "Usage: " << prog << " <process|mlcut|scan|plot> [options]\n"
              << "\n"
              << "  process   Run the event loop, write histograms (and optionally the pT2 ntuple)\n"
              << "    -i  Input file path\n"
              << "    -k  Run low pT\n"
              << "    -n  Number of events\n"
              << "    -r  Also write LSTNtuple_with_pT2.root (with NN scores) and pt2_training_data.root\n"
              << "    -N  NN model directory with model.onnx, mean.npy, std.npy (\"none\" to skip NN scoring)\n"
              << "    -X  Turn the NN off (no model is loaded, nn_score stays -1); same as -N none\n"
              << "    -c  Apply the tuned per-zone threshold cuts in src/pt2_cuts.h\n"
              << "    -a  Ambiguity: keep only the N best pT2s per (pLS, zone, charge)\n"
              << "    -t  NN score threshold: only pT2s with score >= this go into the LST ntuple (needs -r;\n"
              << "        pt2_training_data.root is then not written)\n"
              << "    -s  Events to process and write: all (default), or train, val, test separated by commas\n"
              << "        (e.g. val,test; same event split as train_v7.py, see event_split.h)\n"
              << "\n"
              << "  mlcut     Apply an NN cut to the pT2s of an existing LSTNtuple_with_pT2.root (from process -r)\n"
              << "    -i  LSTNtuple_with_pT2.root (required)\n"
              << "    -N  NN model directory (the --output_dir of train_v7.py)\n"
              << "    -t  NN score threshold: keep pT2s with score >= this (required; see the model's metrics.json)\n"
              << "    -n  Number of events\n"
              << "    -s  Events to use: all (default), or train, val, test separated by commas (same event split as train_v7.py)\n"
              << "    Writes the histogram file (passing pT2s) and <output dir>/cut_study: before/after plots,\n"
              << "    per_layer.csv (analyze_pT2_cuts format plus true/fake counts before the cut)\n"
              << "\n"
              << "  scan      Compute cut values from the histogram file\n"
              << "    -e  Target efficiency percent (default 90)\n"
              << "\n"
              << "  plot      Make plots from the histogram file\n"
              << "\n"
              << "  common\n"
              << "    -o  Output directory (default: output)\n"
              << "    -H  Histogram file (default: <output dir>/pt2_hists.root)\n";
}

} // namespace

bool parseArgs(int argc, char** argv, Config& cfg)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return false;
    }

    std::string mode = argv[1];
    if (mode == "process") cfg.mode = Mode::Process;
    else if (mode == "mlcut") cfg.mode = Mode::MLCut;
    else if (mode == "scan") cfg.mode = Mode::Scan;
    else if (mode == "plot") cfg.mode = Mode::Plot;
    else {
        printUsage(argv[0]);
        return false;
    }

    // Parse options after the subcommand
    optind = 2;
    int opt;
    while ((opt = getopt(argc, argv, "rkXci:o:n:e:H:N:t:s:a:")) != -1) {
        switch (opt) {
        case 'r': cfg.writeRoot = true; break;
        case 'k': cfg.lowPT = true; break;
        case 'i': cfg.inputFile = optarg; break;
        case 'o': cfg.outputDir = optarg; break;
        case 'n': cfg.nEvents = std::stoi(optarg); break;
        case 'e': cfg.targetPercent = std::stod(optarg); break;
        case 'H': cfg.histFile = optarg; break;
        case 'N': cfg.nnModelDir = optarg; break;
        case 't': cfg.nnCut = std::stod(optarg); break;
        case 's': cfg.split = optarg; break;
        case 'X': cfg.useNN = false; break;
        case 'c': cfg.applyCuts = true; break;
        case 'a': cfg.keepPerPls = std::stoi(optarg); break;
        default:
            printUsage(argv[0]);
            return false;
        }
    }

    // mlcut has no default input: it needs an LSTNtuple_with_pT2.root
    if (cfg.inputFile.empty() && cfg.mode == Mode::Process) cfg.inputFile = cfg.lowPT ? kInputLowPT : kInput;
    if (cfg.pixelMapDir.empty()) cfg.pixelMapDir = cfg.lowPT ? kPixelMapLowPT : kPixelMap;
    if (cfg.nnModelDir.empty()) cfg.nnModelDir = kNNModelDir;
    if (cfg.histFile.empty()) cfg.histFile = cfg.outputDir + "/pt2_hists.root";

    return true;
}

void printConfig(const Config& cfg)
{
    std::cout << "\n=== Configuration ===\n";
    switch (cfg.mode) {
    case Mode::Process:
        std::cout << "Mode:              process\n";
        std::cout << "Input file:        " << cfg.inputFile << "\n";
        std::cout << "Pixel maps:        " << cfg.pixelMapDir << "\n";
        std::cout << "Use Low pT:        " << (cfg.lowPT ? "yes" : "no") << "\n";
        std::cout << "Write ROOT file:   " << (cfg.writeRoot ? "yes" : "no") << "\n";
        if (cfg.writeRoot) {
            if (cfg.useNN) std::cout << "NN model dir:      " << cfg.nnModelDir << "\n";
            else           std::cout << "NN scoring:        off (-X)\n";
        }
        if (cfg.nnCut >= 0) std::cout << "NN cut:            score >= " << cfg.nnCut << " (LST ntuple only)\n";
        std::cout << "Threshold cuts:    " << (cfg.applyCuts ? "on (pt2_cuts.h)" : "off") << "\n";
        if (cfg.keepPerPls > 0)
            std::cout << "Ambiguity:         keep " << cfg.keepPerPls << " best per (pLS, zone)\n";
        std::cout << "Events:            " << cfg.split << "\n";
        if (cfg.nEvents > 0) std::cout << "Number of events:  " << cfg.nEvents << "\n";
        break;
    case Mode::MLCut:
        std::cout << "Mode:              mlcut\n";
        std::cout << "Input file:        " << cfg.inputFile << "\n";
        std::cout << "NN model dir:      " << cfg.nnModelDir << "\n";
        std::cout << "NN cut:            score >= " << cfg.nnCut << "\n";
        std::cout << "Events:            " << cfg.split << "\n";
        if (cfg.nEvents > 0) std::cout << "Number of events:  " << cfg.nEvents << "\n";
        break;
    case Mode::Scan:
        std::cout << "Mode:              scan\n";
        std::cout << "Target percent:    " << cfg.targetPercent << "\n";
        break;
    case Mode::Plot:
        std::cout << "Mode:              plot\n";
        break;
    }
    std::cout << "Output directory:  " << cfg.outputDir << "\n";
    std::cout << "Histogram file:    " << cfg.histFile << "\n";
    std::cout << "=====================\n\n";
}
