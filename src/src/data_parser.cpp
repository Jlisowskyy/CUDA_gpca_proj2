/* internal includes */
#include <data_parser.hpp>
#include <defines.hpp>

/* external includes */
#include <fstream>
#include <stdexcept>

enum ParseStage : int {
    SequenceStage,
    SolutionStage,
};

BinSequencePack DataParser::ParseData(const char *path) const {
    std::ifstream inputFile(path);

    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    BinSequencePack data{};
    std::string line{};
    ParseStage stage = SequenceStage;
    size_t line_counter{};

    try {
        while (std::getline(inputFile, line)) {
            switch (stage) {
                case SequenceStage:
                    if (!line.empty()) {
                        ParseSequence_(line, data);
                    } else {
                        stage = SolutionStage;
                    }
                    break;
                case SolutionStage:
                    ParseSolution_(line, data);
                    break;
            }
            ++line_counter;
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(
            std::string("Failed to parse line ") + std::to_string(line_counter) + ": " + e.what());
    }

    if (data.sequences.size() < kMinNumSequences) {
        throw std::runtime_error(
            "Number of sequences is too low. Expected minimum: " + std::to_string(kMinNumSequences));
    }

    size_t total_mem;
    for (const auto &sequence: data.sequences) {
        total_mem += sequence.GetSizeWords() * sizeof(uint64_t);
    }

    return data;
}

void DataParser::SaveData(const BinSequencePack &data, const char *path) const {
    std::ofstream outputFile(path, std::ios::trunc);

    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    for (const auto &sequence: data.sequences) {
        for (size_t idx = 0; idx < sequence.GetSizeBits(); ++idx) {
            outputFile << (sequence.GetBit(idx) ? '1' : '0');
        }
        outputFile << std::endl;
    }

    outputFile << std::endl;

    for (const auto &[left, right]: data.solution) {
        outputFile << left << ',' << right << std::endl;
    }
}

void DataParser::ParseSequence_(const std::string &line, BinSequencePack &data) const {
    if (line.length() < kMinSequenceLength) {
        throw std::runtime_error("Sequence is too short");
    }

    data.sequences.emplace_back(line.length());
    auto &sequence = data.sequences.back();

    size_t cursor = 0;
    for (const char c: line) {
        bool value;
        switch (c) {
            case '1':
                value = true;
                break;
            case '0':
                value = false;
                break;
            default:
                throw std::runtime_error("Invalid character in sequence");
        }

        sequence.SetBit(cursor++, value);
    }

    data.max_seq_size_bits = std::max(data.max_seq_size_bits, sequence.GetSizeBits());
}

void DataParser::ParseSolution_(const std::string &line, BinSequencePack &data) const {
    const size_t comma_pos = line.find(',');

    if (comma_pos == std::string::npos) {
        throw std::runtime_error("Invalid solution format, missing comma");
    }

    const std::string left = line.substr(0, comma_pos);
    const std::string right = line.substr(comma_pos + 1);

    const size_t left_idx = std::stoul(left);
    const size_t right_idx = std::stoul(right);

    data.solution.emplace_back(left_idx, right_idx);
}
