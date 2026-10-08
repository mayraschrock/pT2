#include "ml_cut.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cut_study.h"
#include "event_split.h"
#include "gator.h"
#include "histograms.h"
#include "process.h"
#include "pt2.h"
#include "pt2_scorer.h"
#include "rootReader.h"
#include "tools.h"

namespace
{
    // pT2s are scored by the NN in chunks of this size
    constexpr size_t kNNBatchSize = 4096;

} // namespace

int runMLCut(const Config &cfg)
{
    if (cfg.inputFile.empty()) throw std::runtime_error("mlcut needs -i <LSTNtuple_with_pT2.root> (made by `pt2 process -r`)");
    if (cfg.nnCut < 0) throw std::runtime_error("mlcut needs -c <NN score threshold> (from the model's metrics.json)");
    if (cfg.nnModelDir == "none") throw std::runtime_error("mlcut needs an NN model (-N <model dir>)");
    if (!event_split::isValid(cfg.split))
        throw std::runtime_error("mlcut -s must be all, or train, val, test separated by commas (got " + cfg.split + ")");

    std::filesystem::create_directories(cfg.outputDir);

    rootReader reader;
    if (!reader.Init(cfg.inputFile, "tree")) throw std::runtime_error("could not open ROOT file or TTree: " + cfg.inputFile);

    // The pT2s written by `process -r`
    std::vector<int> *plsIdx = nullptr, *lsIdx = nullptr;
    if (!reader.inputTree->GetBranch("pT2_plsIdx") || !reader.inputTree->GetBranch("pT2_lsIdx"))
        throw std::runtime_error(cfg.inputFile + " has no pT2_plsIdx / pT2_lsIdx branches; make it with `pt2 process -r`");
    reader.inputTree->SetBranchAddress("pT2_plsIdx", &plsIdx);
    reader.inputTree->SetBranchAddress("pT2_lsIdx", &lsIdx);

    // An ntuple written by `process -s` keeps only some events; pT2_sourceEntry is each event's
    // entry in the original ntuple, which is what the event split is defined on
    Long64_t sourceEntry = -1;
    const bool hasSourceEntry = reader.inputTree->GetBranch("pT2_sourceEntry") != nullptr;
    if (hasSourceEntry) reader.inputTree->SetBranchAddress("pT2_sourceEntry", &sourceEntry);

    Pt2Scorer scorer(cfg.nnModelDir + "/model.onnx", cfg.nnModelDir + "/mean.npy", cfg.nnModelDir + "/std.npy");

    HistogramManager hists;
    hists.init();
    CutStudy cutStudy(hists, "nn_score");

    Long64_t totalEntries = reader.GetEntries();
    Long64_t nEntries = (cfg.nEvents > 0 && cfg.nEvents < totalEntries) ? cfg.nEvents : totalEntries;

    print_creature();

    pT2Collection pt2s;
    std::vector<Long64_t> usedEvents;
    for (Long64_t ievt = 0; ievt < nEntries; ++ievt)
    {
        if (hasSourceEntry)
        {
            reader.inputTree->GetBranch("pT2_sourceEntry")->GetEntry(ievt);
            if (!event_split::selects(cfg.split, sourceEntry)) continue;
        }
        else if (!event_split::selects(cfg.split, ievt)) continue;
        usedEvents.push_back(ievt);

        reader.GetEntry(ievt);
        if (ievt % 2 == 0) printProgressBar(ievt, nEntries);

        // Same per-event state as process: which pLS / LS are already used by a track candidate.
        // The pT2s that `process -r` appended to tc_* have no pT5/pT3/T5 index, so they do not count here.
        UsedMask usedMask = buildUsedMask(reader);
        reader.ls_isUsed = std::move(usedMask.ls_isUsed);
        reader.pls_isUsed = std::move(usedMask.pls_isUsed);

        pt2s.clear();
        pt2s.reserve(plsIdx->size());
        for (size_t i = 0; i < plsIdx->size(); ++i) pt2s.emplace_back(plsIdx->at(i), lsIdx->at(i));

        std::vector<pT2 *> batch;
        auto flush = [&]()
        {
            scoreBatch(scorer, reader, batch);
            for (pT2 *pt2 : batch)
            {
                bool passed = pt2->nn_score >= cfg.nnCut;
                cutStudy.fill(*pt2, passed);
                if (passed) fillHistograms(hists, *pt2);
            }
            batch.clear();
        };

        for (auto &pt2 : pt2s)
        {
            computeFeatures(reader, pt2);
            batch.push_back(&pt2);
            if (batch.size() == kNNBatchSize) flush();
        }
        flush();

        std::cout << " npt2: " << pt2s.size() << std::endl;
    }
    std::cout << "\n";

    std::cout << "Used " << usedEvents.size() << " of " << nEntries << " events (" << cfg.split << "):";
    for (Long64_t ievt : usedEvents) std::cout << " " << ievt;
    std::cout << std::endl;

    hists.write(cfg.histFile);
    std::cout << "Saved histograms of pT2s passing the NN cut to: " << cfg.histFile << std::endl;
    cutStudy.write(cfg.outputDir);

    return 0;
}
