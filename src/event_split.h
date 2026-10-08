#ifndef EVENT_SPLIT_H
#define EVENT_SPLIT_H

#include <cstdint>
#include <sstream>
#include <string>

#include "Rtypes.h"

// Train / val / test event split, identical to event_split() in pt2_ml/train_v7.py.
// Entry i of the LST ntuple is the event with event_idx = i in pt2_training_data.root.
// The fractions must match the --val_frac / --test_frac the model was trained with
// (these are NN_model's: --val_frac 0.10 --test_frac 0.45).
namespace event_split
{
    constexpr uint64_t kSeed = 12345;
    constexpr double kValFrac = 0.10;
    constexpr double kTestFrac = 0.45;

    inline std::string splitOf(Long64_t ievt)
    {
        uint64_t u = (static_cast<uint64_t>(ievt) * 2654435761ULL + kSeed) % 1000003ULL;
        double x = static_cast<double>(u) / 1000003;
        if (x < kTestFrac) return "test";
        if (x < kTestFrac + kValFrac) return "val";
        return "train";
    }

    // spec: "all", or a comma-separated list of train, val, test (e.g. "val,test")
    inline bool isValid(const std::string &spec)
    {
        if (spec == "all") return true;
        std::stringstream ss(spec);
        std::string part;
        bool any = false;
        while (std::getline(ss, part, ','))
        {
            if (part != "train" && part != "val" && part != "test") return false;
            any = true;
        }
        return any;
    }

    inline bool selects(const std::string &spec, Long64_t ievt)
    {
        if (spec == "all") return true;
        const std::string split = splitOf(ievt);
        std::stringstream ss(spec);
        std::string part;
        while (std::getline(ss, part, ','))
            if (part == split) return true;
        return false;
    }
} // namespace event_split

#endif
