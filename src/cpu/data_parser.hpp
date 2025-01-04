#ifndef DATA_PARSER_HPP
#define DATA_PARSER_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <string>

class DataParser {
    // ------------------------------
    // Class creation
    // ------------------------------
public:
    DataParser() = default;

    ~DataParser() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    [[nodiscard]] BinSequencePack ParseData(const char *path) const;

    void SaveData(const BinSequencePack &data, const char *path) const;

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    void ParseSequence_(const std::string &line, BinSequencePack &data) const;

    void ParseSolution_(const std::string &line, BinSequencePack &data) const;

    // ------------------------------
    // class fields
    // ------------------------------
};

#endif //DATA_PARSER_HPP
