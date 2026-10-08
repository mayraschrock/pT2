#ifndef CUT_STUDY_H
#define CUT_STUDY_H

#include <TH1D.h>

#include <array>
#include <string>
#include <vector>

#include "histograms.h"
#include "pt2.h"

// Before/after-cut distributions and yields for one cut (used by `pt2 mlcut`).
// Every pT2 is passed to fill() with whether it passed the cut; write() then saves into <outputDir>/cut_study/:
//   cut_study.root           all histograms
//   per_layer.csv            one row per layer connection plus "all": analyze_pT2_cuts columns + counts before the cut
//   *.png, *.pdf             one plot per variable and layer (plus "all"), with an index.html gallery
class CutStudy {
public:
    // cutName labels the cut row of the CSVs (e.g. "nn_score")
    CutStudy(const HistogramManager &hists, std::string cutName);
    ~CutStudy();

    void fill(const pT2 &pt2, bool passed);
    void write(const std::string &outputDir) const;

private:
    // Layer connections that share one set of thresholds in pt2_cuts.h (and are plotted together by
    // plot_recipes.cc); each group gets its own slot
    static constexpr int kNMerged = 4;
    static constexpr int kMergedGroups[kNMerged][2] = {
        {0, 2},   // L1F_to_L2F + L1T_to_L2F
        {1, 3},   // L1F_to_L2T + L1T_to_L2T
        {5, 7},   // L2F_to_L3F + L2T_to_L3F
        {6, 8},   // L2F_to_L3T + L2T_to_L3T
    };
    // Slot 0 is all categories together, slot c+1 is category c, slot kNCat+1+g is merged group g
    static constexpr int kNSlot = kNCat + 1 + kNMerged;

    struct Var {
        std::string name, axis;
        int nbins;
        double lo, hi;
        double (*get)(const pT2 &);   // returns NAN when the value could not be computed
    };

    void writeCsv(const std::string &dir) const;
    void drawPlots(const std::string &dir) const;

    std::string cutName_;
    std::vector<std::string> slotNames_, slotTitles_;
    std::vector<Var> vars_;
    // hists_[var][slot][real][passed]; "before" plots use passed + failed
    std::vector<std::array<std::array<std::array<TH1D *, 2>, 2>, kNSlot>> hists_;
    // counts_[slot][real][passed]
    double counts_[kNSlot][2][2] = {};
};

#endif
