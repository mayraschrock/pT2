#ifndef PLOTTING_H
#define PLOTTING_H

#include <string>
#include <vector>
#include "plot_recipes.h"
#include "histograms.h"

class Plotting {
public:
    void plotRecipes(const std::vector<PlotRecipe>& recipes,
                     const std::string& outputDir);

    // index.html with every .png in outputDir
    static void generateHTMLGallery(const std::string& outputDir);
};

#endif
