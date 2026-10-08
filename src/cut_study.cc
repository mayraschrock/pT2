#include "cut_study.h"

#include <TCanvas.h>
#include <TColor.h>
#include <TGaxis.h>
#include <TFile.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TStyle.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "plotting.h"

namespace
{
    // Same values as fillSet() in process.cc treats as "could not be computed"
    double orNan(double v, double sentinel) { return v > sentinel ? v : NAN; }
    double heli(const pT2 &p, size_t i) { return (p.heli.size() == 4 && p.heli[0] >= 0) ? p.heli[i] : NAN; }
    double rz(double v) { return v > -900 ? v : NAN; }

    double ratio(double num, double den) { return den > 0 ? num / den : NAN; }

    // Same formatting as analyze_pT2_cuts: 6 decimals, empty if nothing to divide
    std::string fraction(double num, double den)
    {
        if (den <= 0) return "";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", num / den);
        return buf;
    }

    // analyze_pT2_cuts names for the layer connections
    const char *kLayerFileNames[kNCat] = {
        "L1F_to_L2F", "L1F_to_L2T", "L1T_to_L2F", "L1T_to_L2T", "L1T_to_E1PS",
        "L2F_to_L3F", "L2F_to_L3T", "L2T_to_L3F", "L2T_to_L3T", "L2T_to_E1PS",
        "E1PS_to_E2PS", "E1PS_to_E2-2S", "E2PS_to_E3PS"};
} // namespace

CutStudy::CutStudy(const HistogramManager &hists, std::string cutName) : cutName_(std::move(cutName))
{
    slotNames_ = {"all"};
    slotTitles_ = {"All categories"};
    slotNames_.insert(slotNames_.end(), hists.catNames.begin(), hists.catNames.end());
    slotTitles_.insert(slotTitles_.end(), hists.catTitles.begin(), hists.catTitles.end());
    // Same names as the merged plots of plot_recipes.cc
    for (const auto &g : kMergedGroups)
    {
        slotNames_.push_back(hists.catNames[g[0]] + "_and_" + hists.catNames[g[1]]);
        slotTitles_.push_back(hists.catTitles[g[0]] + " + " + hists.catTitles[g[1]]);
    }

    // Binning matches HistogramManager::init(), except the eta variables: coarse, 36 bins of 0.25 from -4.5 to 4.5
    vars_ = {
        {"deltaPT", "#Delta p_{T} [GeV]", 180, -10.0, 10.0, [](const pT2 &p) -> double { return p.delta_pt; }},
        {"deltaETA", "#Delta#eta (coarse)", 36, -4.5, 4.5, [](const pT2 &p) -> double { return p.delta_eta; }},
        {"deltaPHI", "#Delta #phi [rad]", 180, -1.0, 1.0, [](const pT2 &p) -> double { return p.delta_phi; }},
        {"deltaR", "#Delta R", 180, 0.0, 1.0, [](const pT2 &p) -> double { return p.delta_r; }},
        {"deltaAngle", "#Delta angle", 180, 0.0, 0.5, [](const pT2 &p) { return orNan(p.delta_angle, -1.0); }},
        {"pls_ETA", "pLS #eta (coarse)", 36, -4.5, 4.5, [](const pT2 &p) -> double { return p.pls_eta; }},
        {"ls_ETA", "LS #eta (coarse)", 36, -4.5, 4.5, [](const pT2 &p) -> double { return p.ls_eta; }},
        {"LSTdPhi", "LST #Delta #phi", 180, -0.5, 0.5, [](const pT2 &p) { return orNan(p.lst_delta_phi, -100.0); }},
        {"LSTdBeta", "LST #Delta #beta", 180, -0.5, 0.5, [](const pT2 &p) { return orNan(p.delta_beta, -100.0); }},
        {"LSTbetaOut", "LST #beta_{out}", 180, -0.5, 0.5, [](const pT2 &p) { return orNan(p.beta_out, -100.0); }},
        {"LSTKinZRes", "LST kinematic z residual", 180, -5.0, 5.0, [](const pT2 &p) { return orNan(p.z_res_kin, -100.0); }},
        {"LSTOrgZRes", "LST geometric z residual", 180, -25.0, 25.0, [](const pT2 &p) { return orNan(p.z_res_geo, -100.0); }},
        {"MD0_dXY", "MD0 d_{xy}", 180, 0.0, 5.0, [](const pT2 &p) { return heli(p, 0); }},
        {"MD0_dZ", "MD0 d_{z}", 180, 0.0, 10.0, [](const pT2 &p) { return heli(p, 1); }},
        {"MD1_dXY", "MD1 d_{xy}", 180, 0.0, 5.0, [](const pT2 &p) { return heli(p, 2); }},
        {"MD1_dZ", "MD1 d_{z}", 180, 0.0, 10.0, [](const pT2 &p) { return heli(p, 3); }},
        {"MD0_rz_simple", "MD0 r-z pointing", 180, -10.0, 10.0, [](const pT2 &p) { return rz(p.rz_simple.first); }},
        {"MD1_rz_simple", "MD1 r-z pointing", 180, -10.0, 10.0, [](const pT2 &p) { return rz(p.rz_simple.second); }},
        {"nn_score", "NN score", 100, 0.0, 1.0, [](const pT2 &p) -> double { return p.nn_score >= 0 ? p.nn_score : NAN; }},
    };

    const char *truth[2] = {"fake", "real"};
    const char *status[2] = {"failed", "passed"};
    hists_.resize(vars_.size());
    for (size_t v = 0; v < vars_.size(); ++v)
        for (int s = 0; s < kNSlot; ++s)
            for (int r = 0; r < 2; ++r)
                for (int p = 0; p < 2; ++p)
                {
                    std::string name = std::string("cut_") + truth[r] + "_" + status[p] + "_" + vars_[v].name + "_" + slotNames_[s];
                    auto *h = new TH1D(name.c_str(), (";" + vars_[v].axis + ";pT2s").c_str(), vars_[v].nbins, vars_[v].lo, vars_[v].hi);
                    h->SetDirectory(nullptr);
                    hists_[v][s][r][p] = h;
                }
}

CutStudy::~CutStudy()
{
    for (auto &var : hists_)
        for (auto &slot : var)
            for (auto &truth : slot)
                for (TH1D *h : truth) delete h;
}

void CutStudy::fill(const pT2 &pt2, bool passed)
{
    int r = pt2.is_real ? 1 : 0, p = passed ? 1 : 0;
    int cat = (pt2.combo_idx >= 0 && pt2.combo_idx < kNCat) ? pt2.combo_idx + 1 : -1;
    int merged = -1;
    for (int g = 0; g < kNMerged; ++g)
        if (pt2.combo_idx == kMergedGroups[g][0] || pt2.combo_idx == kMergedGroups[g][1]) merged = kNCat + 1 + g;

    counts_[0][r][p] += 1;
    if (cat > 0) counts_[cat][r][p] += 1;
    if (merged > 0) counts_[merged][r][p] += 1;

    for (size_t v = 0; v < vars_.size(); ++v)
    {
        double x = vars_[v].get(pt2);
        if (std::isnan(x)) continue;
        hists_[v][0][r][p]->Fill(x);
        if (cat > 0) hists_[v][cat][r][p]->Fill(x);
        if (merged > 0) hists_[v][merged][r][p]->Fill(x);
    }
}

void CutStudy::write(const std::string &outputDir) const
{
    std::string dir = outputDir + "/cut_study";
    std::filesystem::create_directories(dir);

    std::unique_ptr<TFile> f(TFile::Open((dir + "/cut_study.root").c_str(), "RECREATE"));
    if (!f || f->IsZombie()) throw std::runtime_error("could not create " + dir + "/cut_study.root");
    for (auto &var : hists_)
        for (auto &slot : var)
            for (auto &truth : slot)
                for (TH1D *h : truth) h->Write();
    f->Close();

    writeCsv(dir);
    drawPlots(dir);
    std::cout << "Saved cut study to: " << dir << std::endl;
}

void CutStudy::writeCsv(const std::string &dir) const
{
    // One row per layer connection (both charges together) plus "all": the analyze_pT2_cuts columns,
    // plus how many true / fake pT2s the layer had before the cut. With a single cut, eff and cumulative eff are the same
    const std::string path = dir + "/per_layer.csv";
    std::ofstream csv(path);
    if (!csv) throw std::runtime_error("could not create " + path);
    csv << "Layer,True pT2 before cut,True pT2 passing cut,True eff,True cumulative eff,"
           "Fake pT2 before cut,Fake pT2 passing cut,Fake eff,Fake cumulative eff\n";

    auto row = [&](const std::string &name, const double (&c)[2][2])
    {
        csv << name;
        for (int r : {1, 0})
        {
            double atStart = c[r][0] + c[r][1], passing = c[r][1];
            csv << "," << (long long)atStart << "," << (long long)passing << "," << fraction(passing, atStart) << ","
                << fraction(passing, atStart);
        }
        csv << "\n";
    };

    // A merged group replaces its two layer connections: one row, where the first of them would be
    for (int cat = 0; cat < kNCat; ++cat)
    {
        int slot = cat + 1;
        std::string name = kLayerFileNames[cat];
        bool second = false;
        for (int g = 0; g < kNMerged; ++g)
        {
            if (cat == kMergedGroups[g][1]) second = true;
            if (cat == kMergedGroups[g][0])
            {
                slot = kNCat + 1 + g;
                name += std::string("_and_") + kLayerFileNames[kMergedGroups[g][1]];
            }
        }
        if (second) continue;
        const double(&c)[2][2] = counts_[slot];
        if (c[0][0] + c[0][1] + c[1][0] + c[1][1] > 0) row(name, c);
    }
    row("all", counts_[0]);
}

void CutStudy::drawPlots(const std::string &dir) const
{
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(1);
    // Same real/fake colors as analyze_pT2_cuts.cpp
    const int realColor = TColor::GetColor("#3a7abf"), fakeColor = TColor::GetColor("#bf3a3a");

    for (size_t v = 0; v < vars_.size(); ++v)
    {
        for (int s = 0; s < kNSlot; ++s)
        {
            const auto &h = hists_[v][s];
            if (h[0][0]->GetEntries() + h[0][1]->GetEntries() + h[1][0]->GetEntries() + h[1][1]->GetEntries() == 0) continue;

            std::string file = "cut_" + vars_[v].name + "_" + slotNames_[s];
            TCanvas canvas(("c_" + file).c_str(), "", 900, 700);
            canvas.SetLogy();
            canvas.SetLeftMargin(0.12);
            canvas.SetRightMargin(0.05);
            canvas.SetBottomMargin(0.13);

            // before = passed + failed, after = passed
            std::unique_ptr<TH1D> before[2], after[2];
            double ymax = 0;
            for (int r = 0; r < 2; ++r)
            {
                int color = r ? realColor : fakeColor;
                before[r].reset((TH1D *)h[r][1]->Clone((file + (r ? "_rb" : "_fb")).c_str()));
                before[r]->Add(h[r][0]);
                before[r]->SetLineColor(color);
                before[r]->SetLineStyle(2);
                before[r]->SetLineWidth(2);
                after[r].reset((TH1D *)h[r][1]->Clone((file + (r ? "_ra" : "_fa")).c_str()));
                after[r]->SetLineColor(color);
                after[r]->SetLineWidth(2);
                after[r]->SetFillColorAlpha(color, 0.35);
                // Underflow into the first bin, overflow into the last, so every pT2 is drawn
                for (TH1D *hh : {before[r].get(), after[r].get()})
                {
                    const int n = hh->GetNbinsX();
                    hh->SetBinContent(1, hh->GetBinContent(1) + hh->GetBinContent(0));
                    hh->SetBinContent(n, hh->GetBinContent(n) + hh->GetBinContent(n + 1));
                    hh->SetBinContent(0, 0.0);
                    hh->SetBinContent(n + 1, 0.0);
                }
                ymax = std::max(ymax, before[r]->GetMaximum());
            }

            TH1D *frame = before[0].get();
            frame->SetTitle((vars_[v].axis + " before / after cuts (" + slotTitles_[s] + ")").c_str());
            frame->SetMaximum(ymax * 50);
            frame->SetMinimum(0.5);
            frame->GetXaxis()->SetTitleSize(0.05);
            const bool coarseEta = vars_[v].name.find("ETA") != std::string::npos;
            if (coarseEta)   // labels and ticks at the integers; the bin-edge ticks are drawn below
                frame->GetXaxis()->SetNdivisions(9, 0, 0, true);
            frame->GetYaxis()->SetTitleSize(0.05);
            frame->Draw("HIST");
            before[1]->Draw("HIST SAME");
            after[0]->Draw("HIST SAME");
            after[1]->Draw("HIST SAME");

            // Coarse eta: a small unlabelled tick on every bin edge (ROOT's own minor ticks do not
            // follow the binning), drawn as an extra axis along the bottom of the frame
            std::unique_ptr<TGaxis> binTicks;
            if (coarseEta)
            {
                canvas.Update();   // so the pad knows its y range (log10 on a log pad)
                binTicks.reset(new TGaxis(vars_[v].lo, gPad->GetUymin(), vars_[v].hi, gPad->GetUymin(),
                                          vars_[v].lo, vars_[v].hi, vars_[v].nbins, "UN"));
                binTicks->SetTickSize(0.015);
                binTicks->Draw();
            }

            const double(&c)[2][2] = counts_[s];
            auto label = [](const char *what, double n) { return std::string(what) + " (" + std::to_string((long long)n) + ")"; };
            TLegend legend(0.55, 0.70, 0.94, 0.89);
            legend.SetTextSize(0.032);
            legend.AddEntry(before[1].get(), label("Real before", c[1][0] + c[1][1]).c_str(), "l");
            legend.AddEntry(after[1].get(), label("Real after", c[1][1]).c_str(), "f");
            legend.AddEntry(before[0].get(), label("Fake before", c[0][0] + c[0][1]).c_str(), "l");
            legend.AddEntry(after[0].get(), label("Fake after", c[0][1]).c_str(), "f");
            legend.Draw();

            TLatex latex;
            latex.SetNDC();
            latex.SetTextSize(0.032);
            double eff = ratio(c[1][1], c[1][0] + c[1][1]), kept = ratio(c[0][1], c[0][0] + c[0][1]);
            latex.DrawLatex(0.15, 0.85, Form("Real eff: %.2f%%", 100 * eff));
            latex.DrawLatex(0.15, 0.80, Form("Fakes kept: %.3f%%", 100 * kept));

            canvas.SaveAs((dir + "/" + file + ".png").c_str());
            canvas.SaveAs((dir + "/" + file + ".pdf").c_str());
        }
    }
    Plotting::generateHTMLGallery(dir);
}
