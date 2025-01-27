/* internal includes */
#include <hamming.cuh>
#include <runner.hpp>

#include <fstream>

void Runner::Run(const BinSequencePack &data, const std::string &filename) {
    auto result = FindHamming1PairsCUDA(data);

    /* save results to file */
    const auto output_name = filename + ".out";
    std::ofstream output_file(output_name);

    for (const auto &[l, r]: result) {
        output_file << l << ' ' << r << '\n';
    }

    output_file.close();
}
